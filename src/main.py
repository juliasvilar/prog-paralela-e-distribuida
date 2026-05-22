import time
import numpy as np
from utils import load_image_as_array, save_array_as_image
from cpu_filters import traditional_convolution_cpu
from gpu_filters import run_traditional_gpu

def calculate_metrics(time_cpu, time_gpu, image_shape):
    """Calcula Speedup e Throughput."""
    pixels_totais = image_shape[0] * image_shape[1]
    megapixels = pixels_totais / 1_000_000
    
    # Speedup: Quantas vezes a GPU foi mais rápida
    speedup = time_cpu / time_gpu if time_gpu > 0 else 0
    
    # Throughput: Megapixels processados por segundo
    throughput_cpu = megapixels / time_cpu
    throughput_gpu = megapixels / time_gpu
    
    return speedup, throughput_cpu, throughput_gpu

def main():
    print("Iniciando testes de paralelismo...")
    
    # 1. Carrega a imagem (Coloque uma imagem grande na pasta data/input)
    image_path = r"C:\Users\julia\OneDrive\Documentos\7 periodo\Programacao Paralela e Distribuida\projeto\data\input\ellie.jpg" 
    image = load_image_as_array(image_path)
    
    # 2. Define um Kernel (Exemplo: Filtro de Sobel Horizontal)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    print(f"\nResolução da Imagem: {image.shape[1]}x{image.shape[0]} pixels")
    
    # --- TESTE CPU ---
    print("\n[1] Executando na CPU (Sequencial)...")
    start_cpu = time.perf_counter()
    out_cpu = traditional_convolution_cpu(image, sobel_x)
    end_cpu = time.perf_counter()
    tempo_cpu = end_cpu - start_cpu
    print(f"Tempo CPU: {tempo_cpu:.4f} segundos")
    
    # --- TESTE GPU ---
    print("\n[2] Executando na GPU (Paralelo)...")
    # Warm-up (a primeira execução no Numba sempre demora mais por causa da compilação)
    _ = run_traditional_gpu(image, sobel_x) 
    
    start_gpu = time.perf_counter()
    out_gpu = run_traditional_gpu(image, sobel_x)
    end_gpu = time.perf_counter()
    tempo_gpu = end_gpu - start_gpu
    print(f"Tempo GPU: {tempo_gpu:.4f} segundos")
    
    # --- MÉTRICAS ---
    speedup, tp_cpu, tp_gpu = calculate_metrics(tempo_cpu, tempo_gpu, image.shape)
    
    print("\n" + "="*30)
    print("RESULTADOS FINAIS:")
    print("="*30)
    print(f"Speedup (Aceleração): {speedup:.2f}x mais rápido na GPU")
    print(f"Throughput CPU: {tp_cpu:.2f} Megapixels/segundo")
    print(f"Throughput GPU: {tp_gpu:.2f} Megapixels/segundo")
    
    # Salva o resultado para comprovar que funcionou
    save_array_as_image(out_gpu, "../data/output/sam_sobel_gpu.jpg")

if __name__ == "__main__":
    main()