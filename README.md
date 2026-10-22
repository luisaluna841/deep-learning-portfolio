# Visão computacional eficiente para dispositivos móveis

**Do backpropagation escrito à mão até um classificador otimizado para inferência em hardware limitado:** *transfer learning*, ONNX, quantização, pruning e análise de performance.

📄 **[Relatório técnico (artigo)](RELATORIO.md)**: metodologia, resultados e discussão completos.

| | |
|---|---|
| 🎯 **Acurácia** | **90,2% top-1 / 99,2% top-5** em 37 raças de cães e gatos (Oxford-IIIT Pet) |
| ⚡ **Latência** | **5,6× mais rápida** trocando o runtime (36,9 → 6,6 ms, CPU, 1 thread) |
| 🗜️ **Compressão** | **3,6× menor** sem perda de acurácia (W8A16) |
| 🔍 **Diagnóstico** | quantização INT8 falhou (2,6%); causa localizada em **outliers de ativação** |

---

## Projeto principal: classificador de raças de pets otimizado para mobile

Uma **MobileNetV3-Large** pré-treinada no ImageNet foi ajustada para classificar **37 raças** ([Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)) e depois convertida, comprimida e medida em várias configurações de execução.

| Notebook | Conteúdo |
|---|---|
| [03 · Transfer learning](notebooks/03_projeto_transfer_learning_mobilenetv3.ipynb) | *Fine-tuning* em duas fases, LR por grupo de camadas, AMP, avaliação top-1/top-5, análise de erros, **Grad-CAM** implementado do zero e **registro de experimentos** (configuração, TensorBoard, melhor modelo) |
| [04 · Otimização para mobile](notebooks/04_projeto_otimizacao_mobile.ipynb) | **TorchScript**, exportação **ONNX**, quantização **INT8 dinâmica e estática (QDQ, por canal)**, **diagnóstico de outliers de ativação** e quantização mista / **W8A16**, **pruning** L1 global e benchmark de latência, tamanho e acurácia. **Análise de performance:** MACs × latência em 6 arquiteturas, escalabilidade com threads e o trade-off entre batch, latência e vazão em CPU × GPU (FP32/FP16) |

### Resultados

> Medido no Google Colab (CPU Intel Xeon @ 2,0 GHz, **batch 1, 1 thread**, mediana de 100 execuções). Acurácia no conjunto de teste completo (3.669 imagens). Dados brutos em [`results/`](results/).

| Variante | Tamanho (MB) | Latência p50 (ms) | Top-1 | Speedup | Δ top-1 (p.p.) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 (*eager*) | 17,2 | 36,9 | 90,2% | 1,0× | 0,0 |
| TorchScript FP32 | 17,0 | 12,7 | 90,2% | 2,9× | 0,0 |
| **ONNX Runtime FP32** | 17,0 | **6,6** | 90,2% | **5,6×** | 0,0 |
| ONNX INT8 dinâmico | 4,5 | 32,2 | 54,6% | 1,1× | −35,7 |
| ONNX INT8 estático (convencional) | 4,8 | 9,7 | **2,6%** | 3,8× | **−87,6** |
| ONNX INT8 misto (51 nós em FP32) | 8,9 | 9,2 | 65,2% | 4,0× | −25,1 |
| **ONNX W8A16** | **4,8** | 31,6 | **90,3%** | 1,2× | **0,0** |

![Benchmark: tamanho, latência e acurácia](results/benchmark.png)

**Principais conclusões:**
- **O runtime importa muito:** sem mexer no modelo, trocar o PyTorch *eager* pelo ONNX Runtime deixou a inferência **5,6× mais rápida**.
- **Quantização não é um botão.** O INT8 estático convencional **destruiu** o modelo (90% → 2,6%). Uma análise de sensibilidade por bloco e de estatísticas de ativação localizou a causa: poucos canais, nas conexões residuais e nas convoluções *depthwise*, chegam a valores até 130× acima do percentil 99,9%, e isso esmaga a resolução dos 8 bits.
- **Cada correção tem seu custo:** o **W8A16** recupera **toda a acurácia** com o arquivo 3,6× menor, mas só é rápido em hardware com suporte a 16 bits (NPUs). O INT8 misto mantém a velocidade, mas recupera só parte da acurácia.
- **Pruning não estruturado comprime, mas não acelera:** a perda é de 0,5 p.p. com 20% de esparsidade, e o modelo colapsa a partir de 50%. A latência não muda (~20 ms).

### Análise de performance

![Latência e vazão por batch, CPU × GPU](results/batch_cpu_gpu.png)

- **MACs não preveem latência:** a ResNet-50 faz **72× mais operações** que a MobileNetV3-Small, mas é só **12,5× mais lenta**. As convoluções *depthwise* fazem pouca conta por byte lido, e a memória vira o gargalo.
- **Latência × vazão:** com batch 1, a GPU T4 empata com a CPU (5,9 vs. 6,6 ms). Com batch 32, chega a **1.185 imagens/s em FP32** e **1.925 em FP16**, contra ~125–150 da CPU.
- **Threads:** no Colab, 2 threads não trouxeram ganho (1,01×). As 2 vCPUs devem ser *hyperthreads* do mesmo núcleo, o que reforça a necessidade de medir no hardware de destino.

---

## Fundamentos

| Notebook | Conteúdo |
|---|---|
| [01 · MLP do zero](notebooks/01_fundamentos_mlp_numpy_vs_pytorch.ipynb) | MLP em **NumPy puro**: forward, backprop manual, SGD com momentum, **gradient check** e comparação dos gradientes com o autograd do PyTorch (MNIST) |
| [02 · CNN residual](notebooks/02_cnn_cifar10.ipynb) | ResNet pequena treinada do zero no CIFAR-10: augmentation, OneCycleLR, label smoothing, **mixed precision**, matriz de confusão, erros de alta confiança, throughput e registro de experimentos no **TensorBoard** |

## Competências

- **Deep learning:** backpropagation, CNNs, conexões residuais, BatchNorm, regularização, agendamento de LR, *transfer learning*
- **Visão computacional:** classificação *fine-grained*, data augmentation, interpretabilidade (Grad-CAM)
- **Otimização e deploy:** TorchScript, ONNX Runtime, quantização pós-treino (dinâmica, estática, mista, W8A16), pruning
- **Análise de performance:** protocolo de benchmark (aquecimento, mediana/p90, threads fixas), latência × vazão, CPU × GPU, FP32 × FP16 × INT8, MACs × latência
- **Investigação:** análise de sensibilidade, estatísticas de ativação, validação de hipóteses
- **Experimentos reproduzíveis:** configuração versionada, TensorBoard, melhor modelo salvo pela validação
- **Ferramentas:** PyTorch, torchvision, ONNX Runtime, TensorBoard, NumPy, pandas, scikit-learn, Matplotlib, Git

## Como executar

**Google Colab (recomendado):** abra cada notebook pelo botão "Abrir no Colab" e selecione *GPU T4*. Os notebooks 03 e 04 passam o checkpoint de um para o outro pelo Google Drive (`MyDrive/dl-portfolio/artifacts`).

**Localmente:**

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Todos os notebooks aceitam a variável de ambiente `SMOKE_TEST=1`, que roda uma versão reduzida para testar o código em poucos minutos. As figuras do relatório são geradas com `python tools/make_figures.py`.

## Estrutura

```
RELATORIO.md   relatório técnico (formato de artigo)
notebooks/     notebooks 01–04
results/       tabelas (CSV) e gráficos dos experimentos
  figuras/     figuras do relatório
tools/         scripts que geram os notebooks e as figuras
```

## Desenvolvimento e uso de IA

Usei um assistente de IA (Claude, da Anthropic) como apoio para estruturar o código, investigar a falha da quantização e redigir a documentação. A execução dos experimentos, a validação dos resultados e as escolhas de escopo foram feitas por mim. Usar IA de forma transparente e crítica faz parte de desenvolver IA de forma responsável.

## Próximos passos

- Benchmark num celular real (ONNX Runtime Mobile / ExecuTorch) com aceleradores de NPU e GPU, incluindo consumo de energia
- *Quantization-aware training* e *cross-layer equalization* para obter W8A8 sem perda de acurácia
- Pruning estruturado com *fine-tuning* e *knowledge distillation*
