# Processamento de Imagens: Convolução Tradicional vs. Separável (CPU vs. GPU)

Este projeto foi desenvolvido para a disciplina de **Programação Paralela e Distribuída**. O objetivo principal é realizar uma análise de performance prática entre diferentes abordagens de processamento de imagens (filtragem por convolução 2D), comparando execuções sequenciais e vetorizadas na CPU com o processamento paralelo massivo na GPU utilizando **CUDA (Numba)**.

O projeto avalia o comportamento de dois algoritmos distintos aplicado à detecção de bordas (Filtro de Sobel):
1. **Convolução 2D Tradicional**
2. **Convolução Separável em Profundidade (Depthwise Separable Convolution)**

---

## 🛠️ Arquitetura do Projeto e Algoritmos

A aplicação está dividida em 4 cenários de teste principais para mapear o impacto de otimizações de código e de hardware:

### 1. CPU Tradicional
Utiliza a função nativa `convolve2d` da biblioteca `scipy.signal`. Essa função possui suas rotinas internas escritas e otimizadas diretamente em C, servindo como uma linha de base sólida para processamento sequencial em CPU.

### 2. CPU Separável (Vetorizada)
A Convolução Separável divide o processo matemático em duas etapas:
* **Depthwise Step:** Onde o canal espacial (ex: Sobel $3 \times 3$) é aplicado a cada canal de cor (R, G, B) de forma isolada.
* **Pointwise Step:** Uma convolução espacial de tamanho $1 \times 1$ que realiza a fusão/combinação linear dos canais resultantes. 

A implementação na CPU utiliza a **vetorização do NumPy** para evitar loops `for` nativos do Python que geram gargalos de interpretação, tirando proveito real da redução de operações matemáticas do algoritmo.

### 3. GPU Tradicional
Mapeia a imagem diretamente na memória de vídeo (VRAM) e dispara um Kernel CUDA genérico. Cada thread da GPU fica responsável por calcular a vizinhança inteira $3 \times 3$ de um único pixel para os 3 canais de cor de forma paralela.

### 4. GPU Separável
Aplica o conceito de separabilidade diretamente na arquitetura massivamente paralela. Dispara dois Kernels CUDA em sequência: o primeiro realiza a filtragem espacial por canal (`depthwise_kernel_gpu`) e o segundo faz a combinação linear $1 \times 1$ (`pointwise_kernel_gpu`), otimizando o uso dos registradores e blocos de threads da GPU.

---

## 📁 Estrutura de Arquivos

* `src/cpu_filters.py`: Implementação dos algoritmos executados na CPU.
* `src/gpu_filters.py`: Kernels CUDA e funções auxiliares gerenciadoras do Numba.
* `src/utils.py`: Funções utilitárias para carregamento, conversão e salvamento de imagens via PIL (Pillow).
* `src/main.py`: Script principal que gerencia o fluxo de execução, faz o warm-up da GPU, coleta as métricas e plota os gráficos.

---

## 📊 Métricas Avaliadas

O sistema mede e gera automaticamente relatórios baseados em:
* **Tempo de Execução:** Medido de forma isolada em segundos via `time.perf_counter()`.
* **Throughput (Vazão):** Quantidade de Megapixels processados por segundo ($\text{Megapixels} / \text{segundo}$), o que permite uma comparação justa de escala entre a CPU e a GPU.
* **Speedup:** Fator de aceleração obtido ao migrar o processamento para o ambiente paralelo.

Ao final da execução, o script exporta três gráficos de linha automáticos:
1. Comparativo isolado de tempo na CPU.
2. Comparativo isolado de tempo na GPU.
3. Gráfico geral de Throughput englobando os 4 cenários de teste.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
* Python 3.10+
* Placa de vídeo NVIDIA compatível com CUDA.
* Drivers CUDA e Kit Toolkit instalados corretamente no sistema.

### Instalação das Dependências
Instale as bibliotecas necessárias utilizando o `pip`:

```bash
pip install numpy scipy pillow numba matplotlib
