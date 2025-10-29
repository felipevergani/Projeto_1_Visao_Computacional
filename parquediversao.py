from vpython import *
import math # Para movimentos como o do Barco Viking

# Configuração da Cena 
scene.width = 1024
scene.height = 768
scene.title = "Simulação de Parque de Diversões" 
scene.background = color.cyan

# 1. Funções de Infraestrutura
def criar_chao_e_caminhos():
    """ Cria o chão e os caminhos do parque  """
    print("Criando chão...")
    
    # Chão com Textura
    tint_color = vector(0.5, 0.6, 0.3) 
    ground = box(pos=vector(0, -0.1, 0), 
                 size=vector(200, 0.2, 200), 
                 color=tint_color, 
                 texture=textures.rough,
                 receive_shadow=True) 
    
    # Caminho Principal
    cor_caminho = color.gray(0.2)
    # Caminhos não precisam projetar ou receber sombras
    caminho_principal = box(pos=vector(0, 0.05, 0), 
                            size=vector(100, 0.1, 5), 
                            color=cor_caminho)
    
    # 1. Caminho para o Carrossel 
    z_start_carrossel = 2.5
    z_end_carrossel = 10 
    z_len_carrossel = z_end_carrossel - z_start_carrossel
    z_pos_carrossel = z_start_carrossel + (z_len_carrossel / 2)
    caminho_carrossel = box(pos=vector(0, 0.05, z_pos_carrossel), 
                          size=vector(3, 0.1, z_len_carrossel), 
                          color=cor_caminho)

    # 2. Caminho para a Roda-Gigante
    z_start_roda = 2.5
    z_end_roda = 5
    z_len_roda = z_end_roda - z_start_roda
    z_pos_roda = z_start_roda + (z_len_roda / 2)
    caminho_roda = box(pos=vector(40, 0.05, z_pos_roda), 
                       size=vector(3, 0.1, z_len_roda), 
                       color=cor_caminho)

    # 3. Caminho para o Barco Viking 
    caminho_barco = box(pos=vector(-45, 0.05, -2.5), 
                      size=vector(4, 0.1, 1), 
                      color=cor_caminho)
    
    # 4. Caminho para a Montanha-Russa
    z_start_mr = -2.5
    z_end_mr = -73.25 # Posição Z aproximada da base da escada
    z_len_mr = abs(z_end_mr - z_start_mr)
    z_pos_mr = z_start_mr - (z_len_mr / 2) # Calcula o centro Z do caminho
    caminho_mr = box(pos=vector(0, 0.05, z_pos_mr), 
                   size=vector(3, 0.1, z_len_mr), 
                   color=cor_caminho)

def criar_arvore(posicao):
    """ Cria uma árvore na posição base (chão) especificada. """
    cor_marrom = vector(0.6, 0.4, 0.2)
    # Posição é a base do tronco
    tronco = cylinder(pos=posicao, axis=vector(0, 5, 0), radius=0.5, color=cor_marrom, cast_shadow=True)
    copa = sphere(pos=posicao + vector(0, 5, 0), radius=3, color=color.green, cast_shadow=True)
    return {'tronco': tronco, 'copa': copa}

def criar_poste(posicao):
    """ Cria um poste de luz na posição base (chão) especificada. """
    # Posição é a base do poste
    poste = cylinder(pos=posicao, axis=vector(0, 4, 0), radius=0.1, color=color.gray(0.5), cast_shadow=True)
    luz_poste = sphere(pos=posicao + vector(0, 4, 0), radius=0.3, color=color.yellow, emissive=True)
    return {'poste': poste, 'luz': luz_poste}

def criar_banco(posicao):
    """ Cria um banco na posição central especificada (nível do assento). """
    cor_banco = vector(0.6, 0.4, 0.2)
    
    # Posição é o centro do assento
    assento = box(pos=posicao, 
                  size=vector(2, 0.1, 0.5), 
                  color=cor_banco, cast_shadow=True)
    # Encosto
    encosto = box(pos=posicao + vector(0, 0.3, -0.2), 
                  size=vector(2, 0.6, 0.1), 
                  color=cor_banco, cast_shadow=True)
    # Pés (relativos ao centro do assento)
    pe1 = box(pos=posicao + vector(-0.9, -0.2, 0), size=vector(0.1, 0.4, 0.1), color=cor_banco, cast_shadow=True)
    pe2 = box(pos=posicao + vector(0.9, -0.2, 0), size=vector(0.1, 0.4, 0.1), color=cor_banco, cast_shadow=True)
    return {'assento': assento, 'encosto': encosto, 'pe1': pe1, 'pe2': pe2}

def criar_bebedouro(posicao):
    """ Cria um bebedouro na posição base (chão) especificada. """
    cor_bebedouro = color.gray(0.6)
    altura_bebedouro = 0.8
    # Posição é a base
    bebedouro = cylinder(pos=posicao, 
                         axis=vector(0, altura_bebedouro, 0), 
                         radius=0.2, 
                         color=cor_bebedouro, cast_shadow=True)
    return bebedouro

# FUNÇÕES DE INFRAESTRUTURA 

def criar_paisagismo():
    """ Cria e posiciona os elementos de paisagismo. """
    print("Criando paisagismo...")
    
    criar_arvore(posicao=vector(-30, 0, 20))
    criar_banco(posicao=vector(-30, 0.3, 22)) 

    criar_arvore(posicao=vector(25, 0, 25))
    criar_banco(posicao=vector(25, 0.3, 23))
    
    criar_arvore(posicao=vector(28, 0, 22))
    criar_banco(posicao=vector(23, 0.3, 27)) 

    criar_poste(posicao=vector(20, 0, 3)) 
    criar_poste(posicao=vector(20, 0, -3)) 
    criar_poste(posicao=vector(23, 0, 3)) 
    criar_poste(posicao=vector(23, 0, -3)) 
    criar_poste(posicao=vector(17, 0, 3)) 
    criar_poste(posicao=vector(17, 0, -3))
    criar_bebedouro(posicao=vector(-13, 0, 3))
    criar_arvore(posicao=vector(-20, 0, 3))
    criar_arvore(posicao=vector(13, 0, 3))
    

    criar_arvore(posicao=vector(-30, 0, -15))
    criar_banco(posicao=vector(-30, 0.3, -17))

    criar_arvore(posicao=vector(30, 0, -25))
    criar_banco(posicao=vector(30, 0.3, -23))


def criar_estruturas_basicas():
    """ Cria quiosques, restaurantes, banheiros e bebedouros. """
    print("Criando estruturas...")
    # Exemplo de quiosque (box simples)
    quiosque = box(pos=vector(-10, 1, 5), 
                   size=vector(4, 2, 3), 
                   color=color.orange, cast_shadow=True)
    # Exemplo de restaurante 
    base_rest = box(pos=vector(-15, 2.5, -10), 
                    size=vector(10, 5, 8), 
                    color=color.white, cast_shadow=True)
    porta_rest = box(pos=vector(-12, 1, -6), 
                     size=vector(1.5, 2, 0.1), 
                     color=color.blue, cast_shadow=True) 
    
    # Bebedouros
    criar_bebedouro(posicao=vector(-10, 0, 7))
    criar_bebedouro(posicao=vector(38, 0, 7))

# 2. Funções dos Brinquedos Obrigatórios
def criar_carrossel(posicao):

    print("Criando carrossel...")
    base = cylinder(pos=posicao, axis=vector(0, 0.5, 0), radius=5, color=color.red, cast_shadow=True)
    eixo = cylinder(pos=posicao + vector(0, 0.5, 0), axis=vector(0, 4, 0), radius=0.3, color=color.white, cast_shadow=True)
    
    # Posição do topo (onde as hastes se prendem)
    pos_topo = posicao + vector(0, 4.5, 0) 
    topo = cylinder(pos=pos_topo, axis=vector(0, 0.5, 0), radius=5.5, color=color.blue, cast_shadow=True)
    
    cavalos = []
    hastes = [] 
    
    num_cavalos = 6
    raio_cavalos = 4.0 # O raio onde os cavalos giram
    altura_inicial_cavalos = 1.5 # Altura média

    for i in range(num_cavalos):
        angulo = 2 * math.pi * i / num_cavalos
        
        #  Posição do Cavalo 
        x = posicao.x + raio_cavalos * math.cos(angulo)
        z = posicao.z + raio_cavalos * math.sin(angulo)
        pos_cavalo = vector(x, altura_inicial_cavalos, z)
        cavalo = sphere(pos=pos_cavalo, radius=0.5, color=color.yellow, cast_shadow=True)
        cavalos.append(cavalo)
        
        #  Posição da Haste 
        # A haste se prende no topo, no mesmo ângulo do cavalo
        x_haste = posicao.x + raio_cavalos * math.cos(angulo)
        z_haste = posicao.z + raio_cavalos * math.sin(angulo)
        # A altura Y é a do topo
        pos_haste_inicio = vector(x_haste, pos_topo.y, z_haste) 
        
        # Cria a haste (cilindro que liga o topo ao cavalo)
        haste = cylinder(pos=pos_haste_inicio, 
                         axis=pos_cavalo - pos_haste_inicio, # Aponta do topo para o cavalo
                         radius=0.1, 
                         color=color.gray(0.7), cast_shadow=True)
        hastes.append(haste)

    # Ponto de Câmera para o primeiro cavalo 
    # Posição "sentada" (um pouco acima do centro do cavalo 0)
    pos_camera_car = cavalos[0].pos + vector(0, 0.5, 0)
    camera_point_car = sphere(pos=pos_camera_car, radius=0.1, visible=False)

    # Retorna tudo que precisa ser animado
    return {
        'eixo': eixo, 'topo': topo, 
        'cavalos': cavalos, 'hastes': hastes, 
        'pos_central': posicao,
        'camera_point': camera_point_car 
    }

def criar_roda_gigante(posicao):
    """ Cria uma roda-gigante """
    print("Criando Roda-Gigante...")
    cor_estrutura = vector(0.4, 0.4, 0.5) 
    altura_eixo = 10
    raio_roda = 8
    num_assentos = 8
    
    # Posição central do eixo
    pos_eixo = posicao + vector(0, altura_eixo, 0)
    
    # 1. Estrutura de suporte (Duas vigas em "A" simples)
    viga1_1 = cylinder(pos=posicao + vector(-2, 0, 1), axis=pos_eixo - (posicao + vector(-2, 0, 1)), radius=0.3, color=cor_estrutura, cast_shadow=True)
    viga1_2 = cylinder(pos=posicao + vector(2, 0, 1), axis=pos_eixo - (posicao + vector(2, 0, 1)), radius=0.3, color=cor_estrutura, cast_shadow=True)
    viga2_1 = cylinder(pos=posicao + vector(-2, 0, -1), axis=pos_eixo - (posicao + vector(-2, 0, -1)), radius=0.3, color=cor_estrutura, cast_shadow=True)
    viga2_2 = cylinder(pos=posicao + vector(2, 0, -1), axis=pos_eixo - (posicao + vector(2, 0, -1)), radius=0.3, color=cor_estrutura, cast_shadow=True)
    
    # 2. Eixo Central
    # (O eixo visual que gira)
    eixo = cylinder(pos=pos_eixo - vector(0, 0, 1.5), axis=vector(0, 0, 3), radius=0.4, color=color.red, cast_shadow=True)
    
    # 3. A "Roda" (Anéis e Raios)
    # Usamos o eixo Z como eixo de rotação
    roda_frente = ring(pos=pos_eixo + vector(0, 0, 0.5), axis=vector(0, 0, 1), 
                       radius=raio_roda, thickness=0.2, color=color.white, cast_shadow=True)
    roda_tras = ring(pos=pos_eixo + vector(0, 0, -0.5), axis=vector(0, 0, 1), 
                     radius=raio_roda, thickness=0.2, color=color.white, cast_shadow=True)
    
    # Listas para guardar as partes que precisam girar
    partes_giratorias = [eixo, roda_frente, roda_tras]
    assentos = []
    
    # 4. Assentos/Gôndolas e Raios
    for i in range(num_assentos):
        angulo = 2 * math.pi * i / num_assentos
        
        # Posição do raio/ponto de fixação na roda
        x = pos_eixo.x + raio_roda * math.cos(angulo)
        y = pos_eixo.y + raio_roda * math.sin(angulo)
        pos_fixacao = vector(x, y, pos_eixo.z) # Gira nos eixos X e Y

        # Cria os Raios (ligando o centro à borda)
        raio1 = cylinder(pos=pos_eixo + vector(0,0,0.5), axis=pos_fixacao - (pos_eixo + vector(0,0,0.5)), 
                         radius=0.1, color=color.gray(0.7), cast_shadow=True)
        raio2 = cylinder(pos=pos_eixo + vector(0,0,-0.5), axis=pos_fixacao - (pos_eixo + vector(0,0,-0.5)), 
                         radius=0.1, color=color.gray(0.7), cast_shadow=True)
        partes_giratorias.extend([raio1, raio2])
        
        # Posição do assento (um pouco abaixo do ponto de fixação)
        pos_assento = pos_fixacao + vector(0, -1, 0)
        
        # Assento (uma caixa simples)
        assento = box(pos=pos_assento, 
                      size=vector(1.5, 1, 1.2), 
                      color=color.blue, cast_shadow=True)
        
        # Haste do assento (liga o ponto de fixação ao assento)
        haste_assento = cylinder(pos=pos_fixacao, axis=pos_assento - pos_fixacao, 
                                 radius=0.05, color=color.gray(0.8), cast_shadow=True)
        
        assentos.append(assento)
        partes_giratorias.append(haste_assento) # A haste gira com a roda

    # Ponto de Câmera para o primeiro assento 
    # Posição "sentada" (um pouco acima e à frente do centro do assento 0)
    pos_camera_rg = assentos[0].pos + vector(0, 0.2, 0.5) 
    camera_point_rg = sphere(pos=pos_camera_rg, radius=0.1, visible=False)

    # Retorna os objetos que precisam ser animados
    return {
        'partes_giratorias': partes_giratorias, 
        'assentos': assentos, 
        'eixo_pos': pos_eixo,
        'camera_point': camera_point_rg 
    }

def criar_barco_viking(posicao):
    """ Cria um barco viking com movimento pendular """
    print("Criando Barco Viking...")
    
    cor_marrom = vector(0.6, 0.4, 0.2)
    cor_estrutura = color.gray(0.5) # Cor para os suportes
    
    #  Configurações de Altura 
    altura_plataforma = 1.0 # Altura da plataforma que já existe
    altura_barco_fundo = 1.2 # O fundo do barco, um pouco acima da plataforma
    
    # O barco tem 1.5 de altura (size.y), então seu centro é 0.75
    altura_barco_centro = altura_barco_fundo + 0.75 # 1.2 + 0.75 = 1.95
    
    # O eixo de rotação, 1 unidade acima do centro do barco
    altura_eixo_y = altura_barco_centro + 1.0 # 1.95 + 1.0 = 2.95
    #  FIM DAS CONFIGURAÇÕES 
    
    
    #  Posição do eixo (onde o barco gira) 
    # Usamos a nova altura 'altura_eixo_y' em vez de 10
    pos_eixo = posicao + vector(0, altura_eixo_y, 0) 

    #  Estrutura de Suporte 
    # 4 pilares que seguram o eixo
    base1 = posicao + vector(-3, 0, 2)
    base2 = posicao + vector(3, 0, 2)
    base3 = posicao + vector(-3, 0, -2)
    base4 = posicao + vector(3, 0, -2)
    
    # O topo dos pilares agora usa a 'pos_eixo' rebaixada (automático)
    topo1 = pos_eixo + vector(0, 0, 2)  
    topo2 = pos_eixo + vector(0, 0, -2) 
    
    pilar1 = cylinder(pos=base1, axis=topo1 - base1, radius=0.3, color=cor_estrutura, cast_shadow=True)
    pilar2 = cylinder(pos=base2, axis=topo1 - base2, radius=0.3, color=cor_estrutura, cast_shadow=True)
    pilar3 = cylinder(pos=base3, axis=topo2 - base3, radius=0.3, color=cor_estrutura, cast_shadow=True)
    pilar4 = cylinder(pos=base4, axis=topo2 - base4, radius=0.3, color=cor_estrutura, cast_shadow=True)

    #  Eixo de rotação 
    # Viga horizontal que os pilares suportam
    eixo = cylinder(pos=topo2, 
                    axis=topo1 - topo2,
                    radius=0.3, 
                    color=color.red, cast_shadow=True)
    
    #  Posição do Barco 
    barco = box(pos=posicao + vector(0, altura_barco_centro, 0), 
                size=vector(5, 1.5, 3), 
                color=cor_marrom, cast_shadow=True)
                
    #  Plataforma de Carregamento e Escada 
    cor_plataforma = color.gray(0.6)
    cor_degrau = color.gray(0.4)
    
    plataforma = box(pos=posicao + vector(0, altura_plataforma - 0.1, 4), # pos.y = 0.9
                     size=vector(5, 0.2, 2), 
                     color=cor_plataforma, cast_shadow=True)

    y_chao_barco = -0.1 # Nível do chão (definido em criar_chao_e_caminhos)
    y_base_plataforma_barco = plataforma.pos.y - (plataforma.size.y / 2)
    altura_haste_barco = y_base_plataforma_barco - y_chao_barco
    axis_haste_barco = vector(0, altura_haste_barco, 0)
    raio_haste_barco = 0.1
    
    # Posições X e Z dos cantos
    x_min_barco = plataforma.pos.x - plataforma.size.x / 2
    x_max_barco = plataforma.pos.x + plataforma.size.x / 2
    z_min_barco = plataforma.pos.z - plataforma.size.z / 2
    z_max_barco = plataforma.pos.z + plataforma.size.z / 2
    
    # Base (no chão) das 4 hastes
    pos_haste1 = vector(x_min_barco, y_chao_barco, z_min_barco)
    pos_haste2 = vector(x_max_barco, y_chao_barco, z_min_barco)
    pos_haste3 = vector(x_min_barco, y_chao_barco, z_max_barco)
    pos_haste4 = vector(x_max_barco, y_chao_barco, z_max_barco)
    
    # Criar cilindros
    cylinder(pos=pos_haste1, axis=axis_haste_barco, radius=raio_haste_barco, color=cor_plataforma, cast_shadow=True)
    cylinder(pos=pos_haste2, axis=axis_haste_barco, radius=raio_haste_barco, color=cor_plataforma, cast_shadow=True)
    cylinder(pos=pos_haste3, axis=axis_haste_barco, radius=raio_haste_barco, color=cor_plataforma, cast_shadow=True)
    cylinder(pos=pos_haste4, axis=axis_haste_barco, radius=raio_haste_barco, color=cor_plataforma, cast_shadow=True)

    num_degraus = 5
    largura_degrau = 3.0
    profundidade_degrau = 0.4
    altura_degrau = altura_plataforma / num_degraus 
    
    z_frente_plataforma = plataforma.pos.z + (plataforma.size.z / 2) 
    
    for i in range(num_degraus):
        y_pos = -0.1 + (i * altura_degrau) + (altura_degrau / 2)
        z_pos = z_frente_plataforma + ((num_degraus - 1 - i) * profundidade_degrau) + (profundidade_degrau / 2)
        
        degrau = box(pos=vector(posicao.x, y_pos, z_pos),
                     size=vector(largura_degrau, altura_degrau, profundidade_degrau),
                     color=cor_degrau, cast_shadow=True)

    return {'barco': barco, 'eixo_pos': pos_eixo}


def criar_montanha_russa(posicao):
    """ 
    Cria uma montanha-russa com uma estação no nível do chão.
    A "ida" usa uma função seno (para subidas/descidas).
    A "volta" usa uma função elíptica (para retornar ao início).
    """
    print("Criando Montanha-Russa...")
    

    pontos_trilho = []
    cor_trilho = color.gray(0.8)
    raio_trilho = 0.2
    
    #  Configurações das Hastes de Suporte  
    cor_hastes = color.gray(0.5)
    raio_hastes = 0.15
    espacamento_hastes = 25 # Coloca uma haste a cada 25 pontos
    contador_pontos = 0
    altura_chao = -0.1 # Posição Y do seu 'ground'
    
    #  Definindo a altura da estação 
    altura_estacao = 0.1 # Nível do chão (relativo à 'posicao' base)
    amplitude_vertical = 6 # Altura MÁXIMA que o trilho alcançará
    
    #  A "Ida" (Função Senoidal) 
    z_start = -30
    z_end = 30
    comprimento_ida = z_end - z_start
    num_oscilacoes = 3
    
    passos_ida = 200
    for i in range(passos_ida + 1):
        
        frac_ida = i / passos_ida
        z = z_start + (comprimento_ida * frac_ida)
        x = 0
        
        #  Ajustando a função Seno 
        fator_seno = (sin(frac_ida * num_oscilacoes * 2 * math.pi - (math.pi / 2)) + 1) / 2
        y = altura_estacao + amplitude_vertical * fator_seno
        
        # Adiciona o ponto (posição absoluta)
        ponto_atual = posicao + vector(x, y, z)
        pontos_trilho.append(ponto_atual)
        
        #  ADICIONA HASTE DE SUPORTE  
        if contador_pontos % espacamento_hastes == 0:
            # A haste vai do chão (altura_chao) até a altura do ponto
            base_haste = vector(ponto_atual.x, altura_chao, ponto_atual.z)
            # O eixo (axis) é a altura do trilho menos a altura do chão
            cylinder(pos=base_haste, 
                     axis=vector(0, ponto_atual.y - altura_chao, 0), 
                     radius=raio_hastes, 
                     color=cor_hastes, cast_shadow=True) # <-- Sombra
        contador_pontos += 1
        #  Fim da Haste 
    
    #  A "Volta" (Função Elíptica/Circular) 
    raio_volta_x = 25 
    raio_volta_z = (z_end - z_start) / 2 # = 30
    centro_volta_z = (z_start + z_end) / 2 # = 0
    
    passos_volta = 150
    for i in range(1, passos_volta + 1): 
        
        frac_volta = i / passos_volta
        t = frac_volta * math.pi
        
        z = centro_volta_z + raio_volta_z * cos(t)
        x = -raio_volta_x * sin(t) 
        
        #  A volta é plana e na altura da estação 
        y = altura_estacao 
        
        # (posição absoluta)      
        ponto_atual = posicao + vector(x, y, z)
        pontos_trilho.append(ponto_atual)
         
        if contador_pontos % espacamento_hastes == 0:
            # A haste vai do chão (altura_chao) até a altura do ponto
            base_haste = vector(ponto_atual.x, altura_chao, ponto_atual.z)
            # O eixo (axis) é a altura do trilho menos a altura do chão
            cylinder(pos=base_haste, 
                     axis=vector(0, ponto_atual.y - altura_chao, 0), 
                     radius=raio_hastes, 
                     color=cor_hastes, cast_shadow=True) # <-- Sombra
        contador_pontos += 1
        #  Fim da Haste 
            

    #  Cria os trilhos e o carrinho 
    trilho_visual = curve(pos=pontos_trilho, color=cor_trilho, radius=raio_trilho, cast_shadow=True)

    carrinho = box(pos=pontos_trilho[0], 
                   size=vector(1.5, 0.8, 1), 
                   color=color.red, 
                   make_trail=True,
                   trail_type="curve",
                   trail_color=color.orange,
                   trail_radius=0.1,
                   cast_shadow=True) # <-- Sombra
    
    #  Plataforma da Estação 
    
    # Posição absoluta da estação (ponto 0 do trilho)
    pos_estacao = pontos_trilho[0] 
    
    # O fundo do carrinho está em y = 1.6 - (0.8 / 2) = 1.2
    y_plataforma = 1.2
    y_chao = -0.1 # Nível do chão
    altura_total_subida = y_plataforma - y_chao # Altura que a escada precisa subir
    
    largura_plataforma = 3.0
    profundidade_plataforma = 3.0
    
    cor_plataforma = color.gray(0.6)
    cor_degrau = color.gray(0.4)
    
    # 1. Plataforma Plana (Onde o carrinho para)
    # Posicionada um pouco na frente (Z+) do ponto de parada
    z_plataforma = pos_estacao.z + profundidade_plataforma / 2 + 1.0 # z = -80 + 1.5 + 1.0 = -77.5
    
    plataforma_topo = box(pos=vector(pos_estacao.x, y_plataforma - 0.1, z_plataforma),
                          size=vector(largura_plataforma, 0.2, profundidade_plataforma),
                          color=cor_plataforma, cast_shadow=True)

    # Hastes de Suporte para a Plataforma 
    y_chao_mr = y_chao # Reutiliza o y_chao = -0.1 da escada
    y_base_plataforma_mr = plataforma_topo.pos.y - (plataforma_topo.size.y / 2)
    altura_haste_mr = y_base_plataforma_mr - y_chao_mr
    axis_haste_mr = vector(0, altura_haste_mr, 0)
    raio_haste_mr = 0.1
    
    # Posições X e Z dos cantos
    x_min_mr = plataforma_topo.pos.x - plataforma_topo.size.x / 2
    x_max_mr = plataforma_topo.pos.x + plataforma_topo.size.x / 2
    z_min_mr = plataforma_topo.pos.z - plataforma_topo.size.z / 2
    z_max_mr = plataforma_topo.pos.z + plataforma_topo.size.z / 2
    
    # Base (no chão) das 4 hastes
    pos_haste1_mr = vector(x_min_mr, y_chao_mr, z_min_mr)
    pos_haste2_mr = vector(x_max_mr, y_chao_mr, z_min_mr)
    pos_haste3_mr = vector(x_min_mr, y_chao_mr, z_max_mr)
    pos_haste4_mr = vector(x_max_mr, y_chao_mr, z_max_mr)
    
    # Criar cilindros
    cylinder(pos=pos_haste1_mr, axis=axis_haste_mr, radius=raio_haste_mr, color=cor_plataforma, cast_shadow=True)
    cylinder(pos=pos_haste2_mr, axis=axis_haste_mr, radius=raio_haste_mr, color=cor_plataforma, cast_shadow=True)
    cylinder(pos=pos_haste3_mr, axis=axis_haste_mr, radius=raio_haste_mr, color=cor_plataforma, cast_shadow=True)
    cylinder(pos=pos_haste4_mr, axis=axis_haste_mr, radius=raio_haste_mr, color=cor_plataforma, cast_shadow=True)
    #  FIM DA ADIÇÃO 
                          
    # 2. Escada (que leva do chão até a plataforma)
    num_degraus = 6 # 6 degraus para subir
    largura_degrau = largura_plataforma * 0.8 # Um pouco mais estreita que a plataforma
    altura_degrau = altura_total_subida / num_degraus
    profundidade_degrau = 0.5 # Profundidade de cada degrau
    
    # Posição Z da frente da plataforma (onde a escada começa)
    z_frente_plataforma = plataforma_topo.pos.z + (plataforma_topo.size.z / 2)
    
    for i in range(num_degraus):
        # Posição Y do degrau (começa do chão -0.1)
        y_pos = y_chao + (i * altura_degrau) + (altura_degrau / 2)
        
        # Posição Z do degrau (o mais baixo fica mais à frente)
        z_pos = z_frente_plataforma + ((num_degraus - 1 - i) * profundidade_degrau) + (profundidade_degrau / 2)
        
        degrau = box(pos=vector(plataforma_topo.pos.x, y_pos, z_pos),
                     size=vector(largura_degrau, altura_degrau, profundidade_degrau),
                     color=cor_degrau, cast_shadow=True)
    #  FIM DA ADIÇÃO 

    
    return {'carrinho': carrinho, 'pontos': pontos_trilho}


#  3. Elemento Especial 
def criar_helicoptero(posicao):
    """ Cria um helicóptero animado com spotlight """
    print("Criando Helicóptero...")
    # Corpo (box), hélice (cylinder/box fina), cauda (box)
    corpo = box(pos=posicao, size=vector(5, 2, 2), color=color.gray(0.3), cast_shadow=True)
    helice_topo = cylinder(pos=posicao + vector(0, 1, 0), axis=vector(0, 0.2, 0), radius=4, color=color.white, opacity=0.8, cast_shadow=True)
    
    # Luz local (spotlight)
    # Esta luz se moverá *com* o helicóptero
    spot = local_light(pos=posicao, color=color.white, direction=vector(0, -1, 0))
    # Nota: A local_light também pode ter .shadows=True, 
    # mas isso faria o helicóptero projetar uma sombra de sua própria luz.
    
    # Ponto de Câmera (na "cabine") 
    # Começa 3 unidades à frente (eixo x) da posição inicial
    camera_point_heli = sphere(pos=posicao + vector(3, 0, 0), radius=0.1, visible=False)
    
    # Vamos agrupar (não é um 'compound' real, mas um dicionário para animar)
    return {'corpo': corpo, 'helice': helice_topo, 'luz': spot, 'camera_point': camera_point_heli}

#  4. Iluminação Global e Câmera 
def setup_luzes_e_camera():
    """ Configura luz ambiente, direcional e câmera """
    # Luz direcional (simula o sol) 
    distant_light(direction=vector(1, -0.5, 1), 
                  color=color.gray(0.7),
                  shadows=True) 
                  
    # Luz ambiente (para não ter sombras totalmente pretas)
    scene.ambient = color.gray(0.6)
    
    # Posição inicial da câmera
    scene.camera.pos = vector(0, 50, 70)
    scene.camera.axis = vector(0, -50, -70) # Aponta para a origem
    
    #  MENSAGEM DE CONTROLES 
    # Instruções impressas no console
    print("\n--- Controles da Simulação ---")
    print("Mouse (Exploração Manual):")
    print(" - Arrastar com Botão Direito: Rotacionar a câmera")
    print(" - Scroll (Roda do Mouse): Zoom In/Out")
    print(" - Arrastar com Botão do Meio (ou Ctrl+Arrastar): Mover a cena (Pan)")
    print("\nTeclado (Vistas Pré-definidas):")
    print(" - Tecla '1': Visão Geral do Parque")
    print(" - Tecla '2': Seguir Carrinho da Montanha-Russa")
    print(" - Tecla '3': Câmera no Carrossel (Cavalo 1)")
    print(" - Tecla '4': Câmera na Roda-Gigante (Assento 1)")
    print(" - Tecla '5': Câmera no Helicóptero")
    print(" - Tecla '0': Resetar Câmera (Visão Geral)")
    print("---------------------------------")


#  Chamada das Funções de Criação 
criar_chao_e_caminhos()
criar_paisagismo()
criar_estruturas_basicas() 
setup_luzes_e_camera() 

# Posições dos brinquedos
carrossel = criar_carrossel(posicao=vector(0, 0, 15))
roda_gigante = criar_roda_gigante(posicao=vector(40, 0, 5))
barco_viking = criar_barco_viking(posicao=vector(-45, 0, -9))
montanha_russa = criar_montanha_russa(posicao=vector(0, 1.5, -50))
helicoptero = criar_helicoptero(posicao=vector(30, 40, 0))


# Função para Lidar com Teclado  
def handle_keys(evt):
    """ Gerencia as teclas pressionadas para mudar a câmera. """
    
    # 'scene.camera.follow(None)' é importante para
    # destravar a câmera antes de movê-la manualmente.
    
    if evt.key == '1' or evt.key == '0': # Visão Geral / Resetar
        scene.camera.follow(None)
        scene.camera.pos = vector(0, 50, 70)
        scene.camera.axis = vector(0, -50, -70)
        
    elif evt.key == '2': # Seguir Montanha-Russa
        # A câmera agora está "presa" ao carrinho
        scene.camera.follow(montanha_russa['carrinho'])
        
    elif evt.key == '3': # Câmera no Carrossel (NOVO)
        scene.camera.follow(carrossel['camera_point'])

    elif evt.key == '4': # Câmera na Roda-Gigante (NOVO)
        scene.camera.follow(roda_gigante['camera_point'])
        
    elif evt.key == '5': # Câmera no Helicóptero (NOVO)
        scene.camera.follow(helicoptero['camera_point'])
    

# "Liga" a função handle_keys para responder a eventos de teclado
scene.bind('keydown', handle_keys)


#  Loop de Animação (Onde tudo se move) 
print("Iniciando loop de animação...")

# Variáveis de controle da animação
t = 0 # Tempo
angulo_carrossel = 0
velocidade_carrossel = 0.01

angulo_barco = 0
velocidade_barco = 0.009
amplitude_barco = 1.0 # Em radianos (aprox. 57 graus)

indice_trilho = 0
angulo_helice = 0

velocidade_roda_gigante = 0.005
delay_montanha_russa = 0
delay_frames_total = 300

while True:
    rate(100) # Limita o loop a 100 execuções por segundo
    
    # 1. Animação do Carrossel
    angulo_carrossel += velocidade_carrossel
    
    # Posição central do carrossel (base)
    centro_x = carrossel['pos_central'].x
    centro_z = carrossel['pos_central'].z
    
    # Posição Y do topo (onde as hastes se prendem)
    altura_topo = carrossel['topo'].pos.y
    
    raio_cavalos = 4.0 # Mesmo raio usado na criação
    amplitude_vertical = 0.5 # O quanto o cavalo sobe e desce
    altura_media_cavalos = 1.5 # Mesma altura inicial

    #  Pega o ponto da câmera do carrossel 
    cam_point_car = carrossel['camera_point']

    # Recalcula a posição dos cavalos E HASTES
    for i, cavalo in enumerate(carrossel['cavalos']):
        
        haste = carrossel['hastes'][i] # Pega a haste correspondente
        
        angulo_base = 2 * math.pi * i / len(carrossel['cavalos'])
        novo_angulo = angulo_base + angulo_carrossel
        
        #  1. Atualiza Posição do Cavalo 
        # Posição X e Z (circular)
        cavalo.pos.x = centro_x + raio_cavalos * math.cos(novo_angulo)
        cavalo.pos.z = centro_z + raio_cavalos * math.sin(novo_angulo)
        
        # Posição Y (sobe e desce) - REINTRODUZIDA
        cavalo.pos.y = altura_media_cavalos + amplitude_vertical * math.sin(novo_angulo * 5) 
        
        
        #  2. Atualiza Posição da Haste 
        # Calcula o ponto de fixação no topo (que também gira)
        pos_haste_inicio_x = centro_x + raio_cavalos * math.cos(novo_angulo)
        pos_haste_inicio_z = centro_z + raio_cavalos * math.sin(novo_angulo)
        
        nova_pos_haste = vector(pos_haste_inicio_x, altura_topo, pos_haste_inicio_z)
        
        # Atualiza a haste para conectar o topo ao cavalo
        haste.pos = nova_pos_haste             # Ponto inicial (no topo)
        haste.axis = cavalo.pos - nova_pos_haste # Eixo (aponta para o cavalo)

        #  Atualiza Ponto da Câmera 
        # Se este é o cavalo 0, atualize o ponto da câmera
        # para ter a mesma posição X, Z, mas um Y "sentado".
        if i == 0:
            cam_point_car.pos.x = cavalo.pos.x
            cam_point_car.pos.z = cavalo.pos.z
            # O Y da câmera sobe e desce *com* o cavalo
            cam_point_car.pos.y = cavalo.pos.y + 0.5 # 0.5 acima do centro do cavalo

    
    # Gira o eixo e o topo (elementos visuais centrais)
    carrossel['eixo'].rotate(angle=velocidade_carrossel, axis=vector(0, 1, 0))
    carrossel['topo'].rotate(angle=velocidade_carrossel, axis=vector(0, 1, 0))

    # 2. Animação da Roda-Gigante 
    eixo_pos = roda_gigante['eixo_pos']
    #  Pega o ponto da câmera da roda-gigante 
    cam_point_rg = roda_gigante['camera_point']
    
    # Gira todas as partes estruturais da roda (anéis, raios, hastes)
    for parte in roda_gigante['partes_giratorias']:
        parte.rotate(angle=velocidade_roda_gigante, 
                     axis=vector(0, 0, 1), # Gira em torno do eixo Z
                     origin=eixo_pos)

    # Gira os assentos E aplica contra-rotação
    for i, assento in enumerate(roda_gigante['assentos']):
        
        # 1. Gira o assento junto com a roda
        assento.rotate(angle=velocidade_roda_gigante, 
                       axis=vector(0, 0, 1), 
                       origin=eixo_pos)
        
        #  Gira o Ponto da Câmera (junto com o assento 0) 
        if i == 0:
            cam_point_rg.rotate(angle=velocidade_roda_gigante, 
                                axis=vector(0, 0, 1), 
                                origin=eixo_pos)
        
        # 2. Gira o assento no sentido oposto (contra-rotação)
        #    em torno do seu próprio centro, para ficar sempre "de pé"
        assento.rotate(angle=-velocidade_roda_gigante, 
                       axis=vector(0, 0, 1), 
                       origin=assento.pos) # O 'origin' é o próprio assento
                       
        #  Contra-rotação do Ponto de Câmera (junto com o assento 0) 
        if i == 0:
            cam_point_rg.rotate(angle=-velocidade_roda_gigante, 
                                axis=vector(0, 0, 1), 
                                origin=assento.pos) # A origem é o assento
    
    # 3. Animação do Barco Viking
    # Movimento pendular usando seno
    angulo_barco_atual = amplitude_barco * math.sin(t * velocidade_barco)
    # Gira o barco em torno do eixo
    barco_viking['barco'].rotate(angle=angulo_barco_atual - angulo_barco, # Gira a diferença
                                 axis=vector(0, 0, 1), 
                                 origin=barco_viking['eixo_pos'])
    angulo_barco = angulo_barco_atual # Atualiza o ângulo anterior

    # 4. Animação da Montanha-Russa
    
    # Primeiro, verificamos se o timer do delay está ativo
    if delay_montanha_russa > 0:
        # Se estiver, apenas diminuímos o timer e não movemos o carrinho
        delay_montanha_russa -= 1
        
        # O carrinho fica parado na estação (índice 0)
        montanha_russa['carrinho'].pos = montanha_russa['pontos'][0]
        
    else:
        # Se o delay for 0, o carrinho se move normalmente
        pontos = montanha_russa['pontos']
        carrinho = montanha_russa['carrinho']
        
        # Move o carrinho para o próximo ponto no trilho
        carrinho.pos = pontos[indice_trilho]
        
        # Faz o carrinho "olhar" para a frente
        proximo_indice = (indice_trilho + 1) % len(pontos)
        direcao = pontos[proximo_indice] - pontos[indice_trilho]
        
        if direcao.mag > 0:
            carrinho.axis = direcao.norm()
            
        # Avança para o próximo ponto do trilho
        indice_trilho += 1
        
        # VERIFICAÇÃO: O carrinho chegou ao fim do percurso?
        if indice_trilho >= len(pontos):
            indice_trilho = 0 # Reseta para o início (estação)
            
            #  ACIONA O DELAY 
            delay_montanha_russa = delay_frames_total

    # 5. Animação do Helicóptero 
    angulo_helice += 0.5
    helicoptero['helice'].rotate(angle=0.5, axis=vector(0, 1, 0))
    
    # Pega o ponto da câmera
    cam_point_heli = helicoptero['camera_point']

    # Movimento circular sobrevoando o parque 
    angle_heli = t * 0.0005
    novo_x_heli = 30 * math.cos(angle_heli)
    novo_z_heli = 30 * math.sin(angle_heli)
    nova_pos_heli = vector(novo_x_heli, 40, novo_z_heli)
    
    # Calcula a direção do movimento (tangente ao círculo)
    direcao_heli = vector(-math.sin(angle_heli), 0, math.cos(angle_heli)).norm()

    # Move todas as partes do helicóptero
    helicoptero['corpo'].pos = nova_pos_heli
    helicoptero['corpo'].axis = direcao_heli # Orienta o helicóptero
    
    # A hélice fica no Y=1 relativo ao corpo
    helicoptero['helice'].pos = nova_pos_heli + vector(0, 1, 0)
    
    # A luz fica no centro do corpo
    helicoptero['luz'].pos = nova_pos_heli
    # Faz a luz balançar ou apontar para o centro
    helicoptero['luz'].direction = vector(-novo_x_heli, -40, -novo_z_heli).norm()

    # Move o ponto da câmera para a "frente" do helicóptero
    # (3 unidades à frente do centro do corpo)
    cam_point_heli.pos = nova_pos_heli + direcao_heli * 3


    t += 2 # Incrementa o tempo