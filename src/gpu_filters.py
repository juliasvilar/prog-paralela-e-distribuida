from numba import cuda
import numpy as np
import math

@cuda.jit
def traditional_conv_gpu_kernel(image, kernel, output):
    """
    Kernel CUDA: Cada thread processa UM pixel da imagem.
    """
    # 1. Identifica a posição (x, y) do pixel que esta thread vai processar
    x, y = cuda.grid(2)
    
    h, w, channels = image.shape
    k_size = kernel.shape[0]
    pad = k_size // 2

    # 2. Garante que a thread não tente processar algo fora da imagem
    if x < w and y < h:
        # 3. Aplica o filtro nos 3 canais (RGB)
        for c in range(channels):
            pixel_value = 0.0
            
            # Percorre a matriz do filtro (ex: 3x3)
            for i in range(k_size):
                for j in range(k_size):
                    img_y = y + i - pad
                    img_x = x + j - pad
                    
                    # Checa as bordas para não dar erro
                    if 0 <= img_y < h and 0 <= img_x < w:
                        pixel_value += image[img_y, img_x, c] * kernel[i, j]
            
            # Salva o resultado na imagem de saída
            output[y, x, c] = pixel_value

def run_traditional_gpu(image, kernel):
    """
    Função auxiliar para preparar a GPU, enviar os dados e executar o kernel.
    """
    # Prepara a matriz de saída com zeros
    output = np.zeros_like(image)
    
    # Transfere os dados da Memória RAM (CPU) para a Memória de Vídeo (GPU)
    d_image = cuda.to_device(image)
    d_kernel = cuda.to_device(kernel)
    d_output = cuda.to_device(output)
    
    # Configura a grade de execução da GPU (blocos e threads)
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(image.shape[1] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(image.shape[0] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Dispara o processamento na GPU
    traditional_conv_gpu_kernel[blocks_per_grid, threads_per_block](d_image, d_kernel, d_output)
    
    # Espera a GPU terminar e copia o resultado de volta para a RAM (CPU)
    return d_output.copy_to_host()

@cuda.jit
def depthwise_kernel_gpu(image, spatial_kernel, depthwise_out):
    """
    Aplica o kernel espacial em cada canal independentemente.
    """
    x, y = cuda.grid(2)
    h, w, channels = image.shape
    k_size = spatial_kernel.shape[0]
    pad = k_size // 2

    if x < w and y < h:
        # Cada thread processa os canais de forma independente para o seu pixel (x, y)
        for c in range(channels):
            pixel_value = 0.0
            for i in range(k_size):
                for j in range(k_size):
                    img_y = y + i - pad
                    img_x = x + j - pad
                    
                    # Tratamento de borda (Zero Padding simplificado para GPU)
                    if 0 <= img_y < h and 0 <= img_x < w:
                        pixel_value += image[img_y, img_x, c] * spatial_kernel[i, j]
            
            depthwise_out[y, x, c] = pixel_value

# --- KERNEL 2: POINTWISE STEP ---
@cuda.jit
def pointwise_kernel_gpu(depthwise_out, pointwise_weights, output):
    """
    Combina os canais de cada pixel usando uma operação 1x1 (combinação linear).
    """
    x, y = cuda.grid(2)
    h, w, channels = depthwise_out.shape

    if x < w and y < h:
        # Realiza o produto escalar (dot product) entre os canais do pixel e os pesos
        pixel_val = 0.0
        for c in range(channels):
            pixel_val += depthwise_out[y, x, c] * pointwise_weights[c]
        
        # Replica o resultado nos 3 canais para manter a saída RGB
        for c in range(channels):
            output[y, x, c] = pixel_val

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---
def run_depthwise_separable_gpu(image, spatial_kernel):
    """
    Função auxiliar para coordenar a execução da Convolução Separada na GPU.
    """
    h, w, channels = image.shape
    output = np.zeros_like(image)
    
    # Pesos fixos para a fusão de canais (idêntico ao seu modelo original)
    pointwise_weights = np.array([0.33, 0.33, 0.33], dtype=np.float32)
    
    # Allocando espaço e transferindo dados para a GPU
    d_image = cuda.to_device(image)
    d_spatial_kernel = cuda.to_device(spatial_kernel)
    d_pointwise_weights = cuda.to_device(pointwise_weights)
    
    # Matrizes intermediária e final alocadas diretamente na memória da GPU
    d_depthwise_out = cuda.device_array_like(image)
    d_output = cuda.device_array_like(image)
    
    # Configuração da Grid CUDA (Blocos de 16x16 threads)
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(w / threads_per_block[0])
    blocks_per_grid_y = math.ceil(h / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Passo 1: Executa a Filtragem Espacial (Depthwise)
    depthwise_kernel_gpu[blocks_per_grid, threads_per_block](d_image, d_spatial_kernel, d_depthwise_out)
    
    # Sincroniza para garantir que o passo 1 terminou antes do passo 2 começar
    cuda.synchronize()
    
    # Passo 2: Executa a Fusão de Canais 1x1 (Pointwise)
    pointwise_kernel_gpu[blocks_per_grid, threads_per_block](d_depthwise_out, d_pointwise_weights, d_output)
    
    # Transfere apenas o resultado final de volta para a memória RAM (CPU)
    return d_output.copy_to_host()