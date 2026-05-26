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

def depthwise_separable_convolution_cpu_otimizada(image, spatial_kernel):
    h, w, channels = image.shape
    
    # 1. DEPTHWISE STEP (Continua rápido via SciPy)
    depthwise_out = np.zeros_like(image)
    for c in range(channels):
        depthwise_out[:, :, c] = convolve2d(image[:, :, c], spatial_kernel, mode='same', boundary='symm')
        
    # 2. POINTWISE STEP OTIMIZADO (Sem loops!)
    # Multiplica a matriz inteira pelos pesos usando transmissão (broadcasting) do NumPy
    pointwise_kernel = np.array([0.33, 0.33, 0.33], dtype=np.float32)
    
    # O np.dot ao longo do último eixo (axis=-1) substitui perfeitamente o loop duplo
    output_grayscale = np.dot(depthwise_out, pointwise_kernel)
    
    # Reconverte para 3 canais idênticos (RGB) para manter a compatibilidade do seu projeto
    output = np.stack([output_grayscale] * channels, axis=-1)
            
    return output