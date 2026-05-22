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