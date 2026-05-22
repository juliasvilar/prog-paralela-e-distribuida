import numpy as np # o numpy nos permite manipular as matrizes das imagens de forma rapida e eficiente
from scipy.signal import convolve2d # s scipy possui uma funcao matematica nativa altamente otimizada em C para realizar a operacao de convolucao em duas dimensoes (2D). usaremos ela para aplicar os kernels na CPU

def traditional_convolution_cpu(image, kernel): # define a funcao que recebe a imagem original carregada (um array tridimensional) e o kernel de convolucao (uma matriz bidimensional, como o Sobel 3 x 3)
    """
    Simula uma convolução 2D tradicional pesada.
    """
    h, w, channels = image.shape # Extrai as dimensões da imagem: Altura ($h$), Largura ($w$) e a quantidade de canais (3, por ser RGB).
    kernel_size = kernel.shape[0] # Obtém o tamanho do kernel (ex: se for $3 \times 3$, kernel_size será 3).
    output = np.zeros_like(image) # Cria uma matriz preenchida com zeros que possui exatamente o mesmo tamanho e tipo da imagem original. É aqui que salvaremos o resultado final para não alterar os pixels originais durante o cálculo.
    
    # Aplica o filtro canal por canal de forma padrão
    # Um laço que vai rodar 3 vezes (uma vez para o Canal 0/Vermelho, uma para o Canal 1/Verde e uma para o Canal 2/Azul).
    for c in range(channels):
        output[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
        
    return output # Retorna a imagem RGB completamente filtrada.

def depthwise_separable_convolution_cpu(image, spatial_kernel): # Define a função que recebe a imagem e o kernel espacial (ex: o kernel do Desfoque Gaussiano ). Extrai as dimensões da imagem idêntico ao método anterior.
    """
    Implementação da convolução separável para o projeto.
    """
    h, w, channels = image.shape
    
    # 1. DEPTHWISE STEP (Filtragem Espacial)
    # Aplica o kernel (ex: Gaussiano ou Sobel) separadamente em cada canal RGB
    depthwise_out = np.zeros_like(image)
    for c in range(channels):
        depthwise_out[:, :, c] = convolve2d(image[:, :, c], spatial_kernel, mode='same', boundary='symm')
        
    # 2. POINTWISE STEP (Fusão de Canais - kernel 1x1)
    # Exemplo: um filtro 1x1 que faz a média dos 3 canais (R, G, B)
    # Na prática de Deep Learning, essa matriz seria aprendida. Aqui usamos pesos fixos.
    pointwise_kernel = np.array([0.33, 0.33, 0.33]) 
    
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            # O pixel recebe a combinação dos 3 canais baseada no kernel 1x1
            pixel_val = np.dot(depthwise_out[i, j, :], pointwise_kernel)
            output[i, j, :] = pixel_val
            
    return output