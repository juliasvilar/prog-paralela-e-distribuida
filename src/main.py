import time
import numpy as np
from utils import load_image_as_array, save_array_as_image
from cpu_filters import traditional_convolution_cpu, depthwise_separable_convolution_cpu_otimizada
from gpu_filters import run_traditional_gpu, run_depthwise_separable_gpu

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

import matplotlib.pyplot as plt

def plotar_graficos_finais(tempo_cpu_trad, tempo_cpu_sep, tempo_gpu_trad, tempo_gpu_sep, 
                          tp_cpu_trad, tp_cpu_sep, tp_gpu_trad, tp_gpu_sep):
    """
    Gera três gráficos de linha: Isolado CPU, Isolado GPU e o Comparativo Geral com os 4 cenários.
    """
    # Configuração de estilo para os gráficos ficarem limpos e modernos
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # --- GRÁFICO 1: Isolado CPU (Tempo de Execução - Gráfico de Linha) ---
    plt.figure(figsize=(7, 4.5))
    algoritmos_cpu = ['CPU Tradicional', 'CPU Separável']
    tempos_cpu = [tempo_cpu_trad, tempo_cpu_sep]
    
    plt.plot(algoritmos_cpu, tempos_cpu, marker='o', color='#C0504D', linewidth=2, markersize=8, label='Tempo CPU')
    plt.ylabel('Tempo (segundos)')
    plt.title('Isolado CPU: Tradicional vs Separável (Vetorizada)', fontsize=12, fontweight='bold')
    
    # Adiciona os valores de tempo acima dos pontos da CPU
    for i, v in enumerate(tempos_cpu):
        plt.text(i, v + (max(tempos_cpu) * 0.04), f"{v:.4f}s", ha='center', fontweight='bold')
        
    plt.ylim(0, max(tempos_cpu) * 1.20)
    plt.tight_layout()
    plt.savefig('grafico_isolado_cpu_linha.png', dpi=300)
    plt.close()

    # --- GRÁFICO 2: Isolado GPUs (Tempo de Execução - Gráfico de Linha) ---
    plt.figure(figsize=(7, 4.5))
    algoritmos_gpu = ['GPU Tradicional', 'GPU Separável']
    tempos_gpu = [tempo_gpu_trad, tempo_gpu_sep]
    
    plt.plot(algoritmos_gpu, tempos_gpu, marker='o', color='#9BBB59', linewidth=2, markersize=8, label='Tempo GPU')
    plt.ylabel('Tempo (segundos)')
    plt.title('Isolado GPU: Tradicional vs Separável (Depthwise)', fontsize=12, fontweight='bold')
    
    # Adiciona os valores de tempo acima dos pontos da GPU
    for i, v in enumerate(tempos_gpu):
        plt.text(i, v + (max(tempos_gpu) * 0.04), f"{v:.4f}s", ha='center', fontweight='bold')
        
    plt.ylim(0, max(tempos_gpu) * 1.20)
    plt.tight_layout()
    plt.savefig('grafico_isolado_gpu_linha.png', dpi=300)
    plt.close()

    # --- GRÁFICO 3: Comparativo Geral (Throughput - Gráfico de Linha) ---
    plt.figure(figsize=(9, 5.5))
    
    # Organizado do menor para o maior Throughput esperado
    categorias_geral = ['CPU Separável', 'CPU Tradicional', 'GPU Tradicional', 'GPU Separável']
    throughputs_geral = [tp_cpu_sep, tp_cpu_trad, tp_gpu_trad, tp_gpu_sep]
    
    # Plota a linha comparativa principal
    plt.plot(categorias_geral, throughputs_geral, marker='s', color='#4F81BD', linewidth=2.5,
             markersize=10, markerfacecolor='#C0504D', markeredgewidth=2, label='Vazão de Processamento')
    
    plt.ylabel('Throughput (Megapixels / segundo)')
    plt.title('Comparativo de Performance Geral: CPU vs GPU (4 Cenários)', fontsize=12, fontweight='bold')
    
    # Adiciona as labels com os valores de MP/s em cada um dos 4 pontos
    for i, v in enumerate(throughputs_geral):
        plt.text(i, v + (max(throughputs_geral) * 0.04), f"{v:.2f} MP/s", ha='center', fontweight='bold')
        
    plt.ylim(0, max(throughputs_geral) * 1.15)
    plt.tight_layout()
    plt.savefig('grafico_comparativo_geral_4_cenarios.png', dpi=300)
    plt.close()
    
    print("\n[Sucesso] Todos os 3 gráficos foram gerados:")
    print(" - 'grafico_isolado_cpu_linha.png'")
    print(" - 'grafico_isolado_gpu_linha.png'")
    print(" - 'grafico_comparativo_geral_4_cenarios.png'")

def main():
    print("Iniciando testes de paralelismo...")
    
    # Importar o filtro separável que estava faltando
    from cpu_filters import depthwise_separable_convolution_cpu_otimizada
    
    image_path = r"/content/prog-paralela-e-distribuida/data/input/ellie.jpg" 
    image = load_image_as_array(image_path)
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    print(f"\nResolução da Imagem: {image.shape[1]}x{image.shape[0]} pixels")
    
    # 1. TESTE CPU TRADICIONAL
    print("\n[1] Executando CPU Tradicional...")
    start = time.perf_counter()
    out_cpu_trad = traditional_convolution_cpu(image, sobel_x)
    tempo_cpu_trad = time.perf_counter() - start
    
    # 2. TESTE CPU SEPARÁVEL (DEPTHWISE)
    print("[2] Executando CPU Separável (Depthwise)...")
    start = time.perf_counter()
    out_cpu_sep = depthwise_separable_convolution_cpu_otimizada(image, sobel_x)
    tempo_cpu_sep = time.perf_counter() - start
    
    # 3. TESTE GPU TRADICIONAL
    print("[3] Executando GPU Tradicional...")
    _ = run_traditional_gpu(image, sobel_x) # Warm-up
    start = time.perf_counter()
    out_gpu = run_traditional_gpu(image, sobel_x)
    tempo_gpu = time.perf_counter() - start

    # 4. TESTE GPU SEPARÁVEL 
    print("[4] Executando GPU Separável (Depthwise)...")
    _ = run_depthwise_separable_gpu(image, sobel_x) # Warm-up

    start_gpu_sep = time.perf_counter()
    out_gpu_sep = run_depthwise_separable_gpu(image, sobel_x)
    tempo_gpu_sep = time.perf_counter() - start_gpu_sep
    
    # Métricas
    _, tp_cpu_trad, _ = calculate_metrics(tempo_cpu_trad, 1.0, image.shape)
    _, tp_cpu_sep, _  = calculate_metrics(tempo_cpu_sep, 1.0, image.shape)
    _, tp_gpu_trad, _ = calculate_metrics(tempo_gpu, 1.0, image.shape)      # Seu tempo_gpu antigo
    _, tp_gpu_sep, _  = calculate_metrics(tempo_gpu_sep, 1.0, image.shape)

    speedup_trad = tempo_cpu_trad / tempo_gpu if tempo_gpu > 0 else 0
    speedup_sep  = tempo_cpu_trad / tempo_gpu_sep if tempo_gpu_sep > 0 else 0

    print("\n" + "="*30)
    print("RESULTADOS FINAIS:")
    print("="*30)
    print(f"Throughput CPU Tradicional: {tp_cpu_trad:.2f} MP/s")
    print(f"Throughput CPU Separável:   {tp_cpu_sep:.2f} MP/s")
    print(f"Throughput GPU Tradicional: {tp_gpu_trad:.2f} MP/s")
    print(f"Throughput GPU Separável:   {tp_gpu_sep:.2f} MP/s")
    print("-" * 30)
    print(f"Speedup GPU Tradicional: {speedup_trad:.2f}x mais rápido")
    print(f"Speedup GPU Separável:  {speedup_sep:.2f}x mais rápido")
    print("="*30)

    # Salva imagem e plota os gráficos
    save_array_as_image(out_gpu, "/content/prog-paralela-e-distribuida/data/output/Sam.jpg")
    plotar_graficos_finais(
        tempo_cpu_trad, tempo_cpu_sep, tempo_gpu, tempo_gpu_sep,
        tp_cpu_trad, tp_cpu_sep, tp_gpu_trad, tp_gpu_sep
    )
if __name__ == "__main__":
    main()