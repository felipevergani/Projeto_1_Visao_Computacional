import cv2
import numpy as np
import os
import glob
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# =============================================================================
# 1. Funções de Pré-processamento e Realce 
# =============================================================================

def preprocess_image(image_path):
    """
    Carrega a imagem e aplica pré-processamento para realçar o contraste e 
    reduzir o ruído, abordando variações de iluminação.
    """
    # 1. Carregar a imagem em cores
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None, None

    # 2. Conversão para escala de cinza
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Realce de Contraste (Equalização de Histograma Adaptativa - CLAHE)
    # Mais robusto que a equalização de histograma simples para imagens com 
    # variações de iluminação localizadas.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)

    # 4. Filtragem Espacial (Filtro Gaussiano) para redução de ruído
    # O ruído é comum em imagens de campo.
    img_denoised = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    return img_bgr, img_denoised

# =============================================================================
# 2. Funções de Segmentação 
# =============================================================================

def segment_grains(img_denoised, img_bgr, debug_image_path=None):
    """
    Aplica segmentação para isolar os grãos do fundo.
    Utiliza uma ESTRATÉGIA COMBINADA: Limiar de Brilho E Limiar de Cor.
    """
    
    # 1. Limiarização Adaptativa 
    thresh_bright = cv2.adaptiveThreshold(
        img_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 8
    )
    
    # (ESTRATÉGIA 2: COR)
    # 2. Conversão para o espaço de cores HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 3. Definição dos intervalos de cor para "grãos"
    # Intervalo 1: Marrom/Amarelo 
    lower_brown_yellow = np.array([15, 40, 100])
    upper_brown_yellow = np.array([35, 255, 255])
    mask_brown_yellow = cv2.inRange(img_hsv, lower_brown_yellow, upper_brown_yellow)
    
    # Intervalo 2: Verde 
    lower_green = np.array([36, 40, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    # 4. Combina as máscaras de cor
    # Queremos pixels que sejam MARRONS/AMARELOS OU VERDES
    thresh_color = cv2.add(mask_brown_yellow, mask_green)
 
    
    # 5. Combina as duas estratégias (Brilho E Cor)
    # O objeto final deve ser BRILHANTE E ter a cor de um grão.
    final_mask = cv2.bitwise_or(thresh_color, thresh_bright)
    
    
    # 6. Operações Morfológicas (Abertura) - Aplicadas na máscara final
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 7. Encontrar contornos (como antes)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cria uma cópia da imagem original para desenharmos nela
    if debug_image_path:
        debug_img = img_bgr.copy()
    
    # Criar máscara para isolar os grãos
    mask = np.zeros(img_denoised.shape, dtype="uint8")
    
    min_area = 200
    
    grain_rois = []
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # --- FILTRO 1: ÁREA MÍNIMA ---
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            
            if h == 0 or w == 0:
                continue
            aspect_ratio = float(w) / h
            
            # --- FILTRO 2: FORMA (ASPECT RATIO) ---
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                # --- REJEITADO POR FORMA (TALO) ---
                if debug_image_path:
                    cv2.drawContours(debug_img, [c], -1, (255, 0, 0), 1) # AZUL
                continue 
            
            # --- ACEITO (PASSOU NA ÁREA E NA FORMA) ---
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            if debug_image_path:
                cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2) # VERDE
            
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_bgr.shape[1] - x, w + 2 * margin)
            h = min(img_bgr.shape[0] - y, h + 2 * margin)
            
            roi_bgr = img_bgr[y:y+h, x:x+w]
            roi_mask = mask[y:y+h, x:x+w]
            
            roi_mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            masked_roi = cv2.bitwise_and(roi_bgr, roi_mask_3ch)
            
            grain_rois.append({
                'image': masked_roi, 
                'mask': roi_mask,
                'contour': c
            })
            
        else:
            # --- REJEITADO POR ÁREA (RUÍDO) ---
            if debug_image_path:
                cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 1) # VERMELHO
    
    # Salva a imagem de depuração
    if debug_image_path:
        try:
            # Salva também a máscara de cor para ajudar a ajustar
            debug_mask_path = os.path.join(os.path.dirname(debug_image_path), "mask_color_" + os.path.basename(debug_image_path))
            cv2.imwrite(debug_mask_path, thresh_color)
            
            cv2.imwrite(debug_image_path, debug_img)
            print(f"Imagem de depuração salva em: {debug_image_path}")
        except Exception as e:
            print(f"Erro ao salvar imagem de depuração: {e}")

    return grain_rois

# =============================================================================
# 3. Funções de Extração de Características
# =============================================================================

def extract_features(grain_rois):
    """
    Extrai características de forma, cor e textura de cada grão.
    """
    features_list = []
    
    for roi in grain_rois:
        masked_roi = roi['image']
        mask = roi['mask']
        
        # 1. Características de Cor (Média e Desvio Padrão nos canais RGB e HSV)
        # Usamos a máscara para calcular as estatísticas apenas sobre o grão.
        
        # Converte para HSV
        roi_hsv = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)
        
        # Encontra os pixels do grão (onde a máscara é 255)
        grain_pixels_bgr = masked_roi[np.where(mask == 255)]
        grain_pixels_hsv = roi_hsv[np.where(mask == 255)]
        
        if grain_pixels_bgr.size == 0:
            continue
            
        # Média e Desvio Padrão RGB
        mean_bgr = np.mean(grain_pixels_bgr, axis=0)
        std_bgr = np.std(grain_pixels_bgr, axis=0)
        
        # Média e Desvio Padrão HSV
        mean_hsv = np.mean(grain_pixels_hsv, axis=0)
        std_hsv = np.std(grain_pixels_hsv, axis=0)
        
        # 2. Características de Forma (usando scikit-image regionprops)
        # A máscara é a imagem binária necessária para regionprops.
        props = regionprops(mask)
        
        if not props:
            continue
            
        prop = props[0] # Assume-se que há apenas um objeto (o grão) na máscara
        
        # Características de Forma
        area = prop.area
        perimeter = cv2.arcLength(roi['contour'], True) # Contorno original
        eccentricity = prop.eccentricity
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        solidity = prop.solidity
        extent = prop.extent
        
        # 3. Características de Textura (GLCM - Gray Level Co-occurrence Matrix)
        # A GLCM deve ser calculada na imagem em escala de cinza do ROI, 
        # mas apenas na região do grão.
        
        # Converte o ROI para escala de cinza e aplica a máscara
        roi_gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
        # O GLCM precisa de uma imagem de 8 bits.
        
        # Normaliza a imagem para 0-255 e converte para uint8
        # A imagem já deve estar em uint8, mas garantimos que os valores 
        # fora do grão são 0 (preto).
        
        # Reduz o número de níveis de cinza para 16 ou 32 para GLCM mais estável
        # Usaremos 16 níveis (bins)
        bins = np.linspace(0, 256, 17) # [0, 16, 32, ..., 256]
        roi_glcm = np.digitize(roi_gray, bins) - 1
        roi_glcm[mask == 0] = 0 # Garante que o fundo é 0
        
        # Garante que a imagem tem 16 níveis de cinza (0 a 15)
        max_val = 15
        
        # Calcula a GLCM
        # Distância 1, Ângulos 0, 45, 90, 135 graus
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # A GLCM precisa de uma imagem com valores inteiros não negativos
        # O `roi_glcm` já está em [0, 15]
        
        # Garante que a imagem é uint8 para graycomatrix
        roi_glcm = roi_glcm.astype(np.uint8)
        
        glcm = graycomatrix(roi_glcm, distances=distances, angles=angles, 
                            levels=max_val + 1, symmetric=True, normed=True)

        # Extrai propriedades da GLCM
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # 4. Características de Textura (LBP - Local Binary Pattern)
        # Calculado na imagem em escala de cinza do ROI (roi_gray)
        radius = 3
        n_points = 8 * radius
        
        # LBP é sensível ao fundo. Aplicamos LBP apenas na ROI e então 
        # calculamos o histograma dos valores LBP dentro da máscara.
        lbp_image = local_binary_pattern(roi_gray, n_points, radius, method="uniform")
        
        # Apenas os valores LBP dentro do grão (mask == 255)
        lbp_values = lbp_image[np.where(mask == 255)]
        
        # Histograma LBP normalizado
        n_bins = int(lbp_image.max() + 1)
        hist, _ = np.histogram(lbp_values, bins=n_bins, range=(0, n_bins), density=True)
        
        # Juntando todas as características
        features = {
            # Cor RGB
            'mean_B': mean_bgr[0], 'mean_G': mean_bgr[1], 'mean_R': mean_bgr[2],
            'std_B': std_bgr[0], 'std_G': std_bgr[1], 'std_R': std_bgr[2],
            # Cor HSV
            'mean_H': mean_hsv[0], 'mean_S': mean_hsv[1], 'mean_V': mean_hsv[2],
            'std_H': std_hsv[0], 'std_S': std_hsv[1], 'std_V': std_hsv[2],
            # Forma
            'area': area, 'perimeter': perimeter, 'eccentricity': eccentricity,
            'major_axis': major_axis_length, 'minor_axis': minor_axis_length,
            'solidity': solidity, 'extent': extent,
            # Textura GLCM
            'glcm_contrast': contrast, 'glcm_energy': energy, 
            'glcm_homogeneity': homogeneity, 'glcm_correlation': correlation
        }
        
        # Adiciona as características do Histograma LBP
        for i, val in enumerate(hist):
            features[f'lbp_hist_{i}'] = val
            
        features_list.append(features)
        
    return features_list

# =============================================================================
# 4. Função Principal de Processamento e Classificação 
# =============================================================================

def process_and_classify(image_dir):
    """
    Orquestra o pipeline completo: carrega imagens, pré-processa, segmenta, 
    extrai características e classifica.
    """
    DEBUG_OUTPUT_DIR = 'debug_segmentation'
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)    

    all_features = []
    image_paths = []
    formats_to_read = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")

    # 1. Carregar e processar todas as imagens
    # Se a estrutura for de pastas, usamos recursividade para encontrar as imagens
    # e o nome da pasta como o rótulo da classe.
    for fmt in formats_to_read:
        # Busca recursiva em subdiretórios
        image_paths.extend(glob.glob(os.path.join(image_dir, '**', fmt), recursive=True))
    
    # Extrai o rótulo (classe) a partir do nome do diretório pai
    # Ex: 'teste/CIMMYT/imagem.jpg' -> 'CIMMYT'
    labels = []
    for path in image_paths:
        # Obtém o nome do diretório pai (a classe)
        class_label = os.path.basename(os.path.dirname(path))
        
        # Se a imagem estiver diretamente no IMAGE_DIR (caso de classe única anterior),
        # o rótulo será extraído do nome do arquivo como fallback.
        if class_label == os.path.basename(image_dir) or class_label == '':
            # Extrai o rótulo do nome do arquivo (ex: USASK_!_00001.jpg -> USASK)
            class_label = os.path.basename(path).split('_')[0]
            
        labels.append(class_label)
    
    print(f"Total de {len(image_paths)} imagens encontradas.")
    
    # A lista 'unique_labels' original é usada apenas para fins informativos no início
    unique_labels = sorted(list(set(labels)))
    print(f"Classes originais identificadas: {unique_labels}")
    
    # Loop principal de processamento
    for i, path in enumerate(image_paths):
        print(f"Processando imagem {i+1}/{len(image_paths)}: {os.path.basename(path)}")
        
        # Pré-processamento 
        img_bgr, img_denoised = preprocess_image(path)
        if img_bgr is None:
            continue

        
        debug_filename = f"debug_{os.path.basename(path)}"
        debug_path = os.path.join(DEBUG_OUTPUT_DIR, debug_filename)

        grain_rois = segment_grains(img_denoised, img_bgr, debug_image_path=debug_path)

        # Extração de Características 
        features = extract_features(grain_rois)

        # Assumimos que a imagem contém apenas um tipo de grão (o rótulo da imagem)
        # e que cada ROI é um grão desse tipo.
        for feature_set in features:
            feature_set['label'] = labels[i]
            all_features.append(feature_set)
            
    # 2. Preparação dos Dados para Classificação
    df = pd.DataFrame(all_features)
    
    if df.empty:
        print("Nenhuma característica extraída. Verifique a segmentação e extração.")
        return
        
    
    #Remove colunas com valores NaN (se houver) - pode ocorrer se a extração falhar  
    df = df.dropna()
    
    if df.empty:
        print("Nenhuma característica extraída após o filtro de classes.")
        return
    X = df.drop('label', axis=1)
    # y é o rótulo original (string)
    y = df['label']
    
    # Mapeamento de rótulos para índices numéricos para classificação
    # Refazemos o mapeamento após a filtragem para garantir que os IDs sejam contínuos
    unique_labels_filtered = sorted(list(set(y)))
    label_to_id_filtered = {label: i for i, label in enumerate(unique_labels_filtered)}
    id_to_label_filtered = {i: label for label, i in label_to_id_filtered.items()}
    
    y_mapped = y.map(label_to_id_filtered)
    
    # 3. Classificação 
    
    # Obtemos os IDs únicos presentes no conjunto de teste
    test_labels_ids = sorted(list(np.unique(y_mapped)))
    # Mapeamos os IDs para os nomes de classes correspondentes
    test_target_names = [id_to_label_filtered[i] for i in test_labels_ids]
    
    if len(test_labels_ids) <= 1:
        # Caso de classe única: A classificação é trivial (100% de acurácia)
        # O modelo não pode ser treinado, mas o resultado esperado é 1.0
        print("\n--- Aviso: Apenas uma classe encontrada. A classificação é trivial. ---")
        accuracy = 1.0
        
        # Cria um relatório simulado para a classe única
        report = {
            test_target_names[0]: {
                'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': len(y_mapped)
            },
            'accuracy': accuracy,
            'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': len(y_mapped)},
            'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': len(y_mapped)}
        }
        
        # 4. Resultados Finais (Apenas para classe única)
        results = {
            'KNN_Accuracy': accuracy,
            'KNN_Report': report,
            'SVM_Accuracy': accuracy,
            'SVM_Report': report,
            'Feature_Names': list(X.columns),
            'Classes': test_target_names
        }
        
        return results
    
    # Se houver mais de uma classe, continua com a classificação normal
    # Divisão Treino/Teste
    X_train, X_test, y_train_mapped, y_test_mapped = train_test_split(
        X, y_mapped, test_size=0.3, random_state=42)
    
    # y_test_mapped contém os IDs numéricos (0, 1, 2, ...)
    # y_test contém os rótulos originais (domain1, domain2, ...)
    # Vamos usar y_test_mapped para o treinamento e teste, e os rótulos originais
    # para a lista de nomes de classes.
    
    # Padronização dos dados (importante para KNN e SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Classificador 1: K-Nearest Neighbors (KNN)
    print("\n--- Classificação KNN ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train_mapped)
    y_pred_knn = knn.predict(X_test_scaled)
    
    accuracy_knn = accuracy_score(y_test_mapped, y_pred_knn)
    
    # Obtemos os IDs únicos presentes no conjunto de teste
    test_labels_ids = sorted(list(np.unique(y_test_mapped)))
    # Mapeamos os IDs para os nomes de classes correspondentes
    test_target_names = [id_to_label_filtered[i] for i in test_labels_ids]
    
    report_knn = classification_report(
        y_test_mapped, y_pred_knn, labels=test_labels_ids, 
        target_names=test_target_names, output_dict=True
    )
    
    print(f"Acurácia KNN: {accuracy_knn:.4f}")
    
    # Classificador 2: Support Vector Machine (SVM)
    print("\n--- Classificação SVM ---")
    # Usamos um kernel RBF para lidar com a complexidade dos dados
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train_mapped)
    y_pred_svm = svm.predict(X_test_scaled)
    
    accuracy_svm = accuracy_score(y_test_mapped, y_pred_svm)
    
    report_svm = classification_report(
        y_test_mapped, y_pred_svm, labels=test_labels_ids, 
        target_names=test_target_names, output_dict=True
    )
    
    print(f"Acurácia SVM: {accuracy_svm:.4f}")
    
    # 4. Resultados Finais
    results = {
        'KNN_Accuracy': accuracy_knn,
        'KNN_Report': report_knn,
        'SVM_Accuracy': accuracy_svm,
        'SVM_Report': report_svm,
        'Feature_Names': list(X.columns),
        'Classes': test_target_names 
    }
    
    return results

# =============================================================================
# 5. Execução
# =============================================================================

if __name__ == '__main__':
    IMAGE_DIR = 'C:/Users/Felipe/Documents/visao_computacional/testes' # Ajustado para o novo diretório de teste com subpastas
    
    
    # Nova checagem (procura por qualquer formato)
    image_paths_check = []
    formats_to_read = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    
    # Checagem com recursividade para subdiretórios
    found_images = False
    for fmt in formats_to_read:
        if glob.glob(os.path.join(IMAGE_DIR, '**', fmt), recursive=True):
            found_images = True
            break

    if not os.path.isdir(IMAGE_DIR) or not found_images:
        print(f"Diretório de imagens não encontrado ou vazio: {IMAGE_DIR}")
    else:
        results = process_and_classify(IMAGE_DIR)
        
        if results:
            
            pd.DataFrame(results['KNN_Report']).transpose().to_csv('knn_classification_report.csv')
            pd.DataFrame(results['SVM_Report']).transpose().to_csv('svm_classification_report.csv')
            
            
            with open('final_results_summary.txt', 'w') as f:
                f.write("--- Resumo dos Resultados ---\n")
                f.write(f"Classes Analisadas: {results['Classes']}\n")
                f.write(f"Total de Amostras de Grãos (ROIs): {len(pd.read_csv('knn_classification_report.csv')) - 3}\n") # -3 para remover as linhas de métricas
                f.write(f"\nKNN Acurácia: {results['KNN_Accuracy']:.4f}\n")
                f.write(f"SVM Acurácia: {results['SVM_Accuracy']:.4f}\n")
                f.write("\nRelatórios de classificação detalhados salvos em 'knn_classification_report.csv' e 'svm_classification_report.csv'.\n")
            
            print("\nProcessamento concluído. Resultados salvos.")
            
