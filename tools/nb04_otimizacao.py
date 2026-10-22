from nbbuild import build, colab_badge
import hfdata

NAME = "04_projeto_otimizacao_mobile.ipynb"

cells = [
("md", f"""
# 04 · Projeto: otimização para mobile (parte 2: ONNX, quantização, pruning e benchmark)

{colab_badge(NAME)}

**Pergunta:** quanto dá para reduzir **tamanho** e **latência** do classificador do notebook 03 sem perder acurácia?

| Técnica | O que faz | Ferramenta |
|---|---|---|
| Exportação ONNX | Formato portátil que roda em Android/iOS/embarcados | `torch.onnx` + ONNX Runtime |
| Quantização INT8 dinâmica | Pesos em int8; ativações quantizadas em tempo de execução | `onnxruntime.quantization` |
| Quantização INT8 estática | Pesos **e** ativações em int8, com calibração por canal (QDQ) | `onnxruntime.quantization` |
| Pruning L1 não estruturado | Zera os pesos de menor magnitude | `torch.nn.utils.prune` |
| TorchScript (trace + freeze) | Grafo estático otimizado, executável sem Python (C++/mobile) | `torch.jit` |

Além disso, a seção 8 analisa a **performance em diferentes configurações de execução**: MACs vs. latência entre arquiteturas eficientes, número de threads, e o trade-off entre batch, latência e vazão em CPU e GPU.

**Protocolo de benchmark** (para a comparação ser justa):
- CPU, **batch 1** e **1 thread**, para simular um único núcleo de celular
- 20 execuções de aquecimento e 100 medidas; reporto **mediana e p90** em ms
- Acurácia top-1 no conjunto de **teste** completo, o mesmo para todos os modelos
- Quantização calibrada com imagens de **treino** (nunca do teste)

> Rode o notebook 03 antes: ele salva o checkpoint no Google Drive.
"""),
("code", """
import sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"])
"""),
("code", r"""
import os, io, time, zlib, copy, platform, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_large
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_dynamic, quantize_static, QuantType, QuantFormat,
                                      CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

SMOKE = os.environ.get("SMOKE_TEST") == "1"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ARTIFACTS = "/content/drive/MyDrive/dl-portfolio/artifacts"
    DATA_DIR = "/content/data"
else:
    ARTIFACTS, DATA_DIR = "../artifacts", "../data"
os.makedirs(ARTIFACTS, exist_ok=True)

def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

NUM_THREADS = 1
WARMUP, RUNS = (3, 10) if SMOKE else (20, 100)
print(f"CPU: {cpu_name()} ({os.cpu_count()} núcleos lógicos)")
print(f"torch {torch.__version__} | onnx {onnx.__version__} | onnxruntime {ort.__version__}")
"""),
("code", hfdata.CELL + hfdata.PETS),
("md", "## 1. Carregar o modelo treinado e os dados"),
("code", r"""
ckpt_path = f"{ARTIFACTS}/mobilenetv3_pets.pth"
assert os.path.exists(ckpt_path), f"Checkpoint não encontrado em {ckpt_path}. Rode o notebook 03 primeiro."
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
CLASSES, IMG_SIZE = ckpt["classes"], ckpt["img_size"]

def load_fp32_model():
    m = mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()

model_fp32 = load_fp32_model()
print(f"checkpoint carregado | top-1 no teste reportado pelo notebook 03: {ckpt['test_top1']:.4f}")

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
test_ds = load_pets("test", eval_tf)
trainval_ds = load_pets("trainval", eval_tf)
if SMOKE:
    test_ds = Subset(test_ds, list(range(0, len(test_ds), 30)))

n_calib = 64 if SMOKE else 256
calib_idx = torch.randperm(len(trainval_ds), generator=torch.Generator().manual_seed(SEED))[:n_calib].tolist()
calib_ds = Subset(trainval_ds, calib_idx)

test_dl = DataLoader(test_ds, batch_size=64, num_workers=NUM_WORKERS)
prune_dl = DataLoader(Subset(test_ds, list(range(min(1000, len(test_ds))))), batch_size=64, num_workers=NUM_WORKERS)
sample_input = test_ds[0][0][None]  # 1 imagem real, formato (1, 3, 224, 224)
print(f"teste: {len(test_ds)} imagens | calibração: {len(calib_ds)} imagens de treino")
"""),
("md", "## 2. Ferramentas de medição"),
("code", r"""
def benchmark(fn, x, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))

@torch.no_grad()
def accuracy_torch(model, loader):
    model = model.to(device).eval()
    correct = n = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    model.cpu()
    return correct / n

def make_session(path, threads=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def accuracy_onnx(path, loader):
    sess = make_session(path)  # todas as threads: só para medir acurácia mais rápido
    correct = n = 0
    for x, y in loader:
        logits = sess.run(None, {"input": x.numpy()})[0]
        correct += (logits.argmax(1) == y.numpy()).sum(); n += len(y)
    return correct / n

def file_mb(path):
    return os.path.getsize(path) / 1e6

results = []
def record(name, size_mb, top1, lat):
    results.append({"modelo": name, "tamanho_MB": size_mb, "latencia_ms_p50": lat[0], "latencia_ms_p90": lat[1], "top1": top1})
    print(f"{name:28s} | {size_mb:6.2f} MB | p50 {lat[0]:7.2f} ms | p90 {lat[1]:7.2f} ms | top-1 {top1:.4f}")
"""),
("md", "## 3. Baseline: PyTorch FP32"),
("code", r"""
fp32_pth = f"{ARTIFACTS}/mobilenetv3_pets_fp32_state.pth"
torch.save(model_fp32.state_dict(), fp32_pth)

torch.set_num_threads(NUM_THREADS)
with torch.inference_mode():
    lat = benchmark(model_fp32, sample_input)
torch.set_num_threads(os.cpu_count())
top1_fp32 = accuracy_torch(model_fp32, test_dl)
record("PyTorch FP32", file_mb(fp32_pth), top1_fp32, lat)
"""),
("md", """
### TorchScript

O PyTorch executa o modelo de forma *eager*: interpreta o `forward` em Python, operação por operação. Com `torch.jit.trace` o grafo é gravado a partir de uma entrada de exemplo; `freeze` transforma os pesos em constantes e funde operações (por exemplo, conv + BatchNorm). O resultado roda sem Python, em C++ ou no PyTorch Mobile.

Meço o modelo **depois de salvo e recarregado**, como um app faria. Um cuidado que aprendi testando: `torch.jit.optimize_for_inference` gera um modelo mais rápido em memória, mas que **não pode ser salvo e recarregado** (os pesos ficam pré-empacotados para a CPU local).

> O TorchScript está em modo de manutenção. O caminho atual recomendado pelo PyTorch para deploy é `torch.export` + ExecuTorch, mas a ideia (capturar o grafo e otimizá-lo antes da execução) é a mesma.
"""),
("code", r"""
import warnings
ts_path = f"{ARTIFACTS}/mobilenetv3_pets_ts.pt"
with warnings.catch_warnings(), torch.inference_mode():
    warnings.simplefilter("ignore", FutureWarning)  # avisos de depreciação do TorchScript
    torch.jit.freeze(torch.jit.trace(model_fp32, sample_input)).save(ts_path)
    scripted = torch.jit.load(ts_path).eval()
    assert torch.allclose(scripted(sample_input), model_fp32(sample_input), atol=1e-4)

torch.set_num_threads(NUM_THREADS)
with torch.inference_mode():
    lat = benchmark(scripted, sample_input)
torch.set_num_threads(os.cpu_count())
record("TorchScript FP32", file_mb(ts_path), top1_fp32, lat)  # mesmo cálculo, mesma acurácia (verificado acima)
"""),
("md", """
## 4. Exportação para ONNX

O eixo de batch é exportado como dinâmico. Antes de seguir, confiro se as saídas do ONNX Runtime batem com as do PyTorch.
"""),
("code", r"""
fp32_onnx = f"{ARTIFACTS}/mobilenetv3_pets_fp32.onnx"
export_kw = dict(input_names=["input"], output_names=["logits"],
                 dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model_fp32, sample_input, fp32_onnx, dynamo=False, **export_kw)
except TypeError:  # versões antigas do PyTorch não têm o argumento `dynamo`
    torch.onnx.export(model_fp32, sample_input, fp32_onnx, **export_kw)
onnx.checker.check_model(onnx.load(fp32_onnx))

with torch.no_grad():
    ref = model_fp32(sample_input).numpy()
out = make_session(fp32_onnx).run(None, {"input": sample_input.numpy()})[0]
print(f"diferença máxima PyTorch vs ONNX Runtime: {np.abs(ref - out).max():.2e}")
assert np.allclose(ref, out, atol=1e-3)

sess = make_session(fp32_onnx, NUM_THREADS)
x_np = sample_input.numpy()
lat = benchmark(lambda x: sess.run(None, {"input": x}), x_np)
record("ONNX Runtime FP32", file_mb(fp32_onnx), accuracy_onnx(fp32_onnx, test_dl), lat)
"""),
("md", """
## 5. Quantização INT8

**Dinâmica:** só os pesos são convertidos para int8 de antemão; a escala das ativações é calculada em cada inferência. Não precisa de dados de calibração, mas em CNNs o custo extra em tempo de execução costuma anular o ganho.

**Estática (QDQ):** pesos e ativações em int8. As escalas das ativações são definidas antes, rodando imagens de **treino** pelo modelo (*calibração*). Uso **quantização por canal** nos pesos, que importa muito em convoluções *depthwise*, com faixas de valores bem diferentes entre canais.
"""),
("code", r"""
prep_onnx = f"{ARTIFACTS}/mobilenetv3_pets_prep.onnx"
quant_pre_process(fp32_onnx, prep_onnx)  # inferência de shapes + otimizações recomendadas antes de quantizar

dyn_onnx = f"{ARTIFACTS}/mobilenetv3_pets_int8_dynamic.onnx"
quantize_dynamic(prep_onnx, dyn_onnx, weight_type=QuantType.QUInt8)
sess = make_session(dyn_onnx, NUM_THREADS)
lat = benchmark(lambda x: sess.run(None, {"input": x}), x_np)
record("ONNX INT8 dinâmico", file_mb(dyn_onnx), accuracy_onnx(dyn_onnx, test_dl), lat)
"""),
("code", r"""
class CalibReader(CalibrationDataReader):
    def __init__(self, dataset, batch_size=32):
        self.batches = iter([x.numpy() for x, _ in DataLoader(dataset, batch_size=batch_size)])

    def get_next(self):
        x = next(self.batches, None)
        return None if x is None else {"input": x}

static_onnx = f"{ARTIFACTS}/mobilenetv3_pets_int8_static.onnx"
t0 = time.perf_counter()
quantize_static(prep_onnx, static_onnx, CalibReader(calib_ds),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)
print(f"calibração + quantização: {time.perf_counter() - t0:.0f}s")
sess = make_session(static_onnx, NUM_THREADS)
lat = benchmark(lambda x: sess.run(None, {"input": x}), x_np)
record("ONNX INT8 estático (QDQ)", file_mb(static_onnx), accuracy_onnx(static_onnx, test_dl), lat)
"""),
("md", """
## 6. Pruning: esparsidade não é o mesmo que velocidade

Aplico pruning **global L1 não estruturado**: zero a fração $p$ dos pesos de menor magnitude, considerando todas as convoluções e camadas lineares juntas. Sem *fine-tuning* depois, para ver o efeito puro.

Duas coisas que quero mostrar:
1. Quanto da acurácia se perde em cada nível de esparsidade (subconjunto de 1.000 imagens de teste).
2. O arquivo **comprimido** (zlib) diminui, porque zeros comprimem bem. Mas a **latência não muda**: as matrizes continuam densas. Para ganhar velocidade é preciso pruning **estruturado** (remover canais inteiros) ou kernels esparsos.
"""),
("code", r"""
def prune_global(model, amount):
    m = copy.deepcopy(model)
    targets = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in targets:
        prune.remove(mod, name)  # torna o pruning permanente (remove máscara)
    return m

def compressed_mb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return len(zlib.compress(buf.getvalue(), 6)) / 1e6

def sparsity(model):
    ws = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    return sum((w == 0).sum().item() for w in ws) / sum(w.numel() for w in ws)

prune_rows = []
for amount in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]:
    m = prune_global(model_fp32, amount) if amount else model_fp32
    torch.set_num_threads(NUM_THREADS)
    with torch.inference_mode():
        lat = benchmark(m, sample_input, warmup=WARMUP // 2 or 1, runs=RUNS // 2 or 1)
    torch.set_num_threads(os.cpu_count())
    row = {"pruning": amount, "esparsidade": sparsity(m), "top1_subset": accuracy_torch(m, prune_dl),
           "zlib_MB": compressed_mb(m), "latencia_ms_p50": lat[0]}
    prune_rows.append(row)
    print(f"pruning {amount:.0%}: esparsidade {row['esparsidade']:.1%} | top-1 {row['top1_subset']:.4f} | "
          f"zlib {row['zlib_MB']:.2f} MB | {lat[0]:.2f} ms")
df_prune = pd.DataFrame(prune_rows)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(df_prune["pruning"] * 100, df_prune["top1_subset"], marker="o", color="tab:blue", label="top-1")
ax1.set(xlabel="pesos zerados (%)", ylabel="acurácia top-1", title="Pruning L1 global sem fine-tuning")
ax2 = ax1.twinx()
ax2.plot(df_prune["pruning"] * 100, df_prune["zlib_MB"], marker="s", color="tab:orange", label="tamanho comprimido")
ax2.set_ylabel("MB (zlib)")
fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.15)); ax1.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
("md", "## 7. Resultados"),
("code", r"""
df = pd.DataFrame(results)
base = df.iloc[0]
df["speedup"] = base["latencia_ms_p50"] / df["latencia_ms_p50"]
df["reducao_tamanho"] = base["tamanho_MB"] / df["tamanho_MB"]
df["delta_top1_pp"] = (df["top1"] - base["top1"]) * 100
display_df = df.round({"tamanho_MB": 2, "latencia_ms_p50": 2, "latencia_ms_p90": 2, "top1": 4,
                       "speedup": 2, "reducao_tamanho": 2, "delta_top1_pp": 2})
display_df
"""),
("code", r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
colors = plt.cm.tab10(range(len(df)))
axes[0].barh(df["modelo"], df["tamanho_MB"], color=colors); axes[0].set(title="Tamanho do arquivo (MB)")
axes[1].barh(df["modelo"], df["latencia_ms_p50"], xerr=df["latencia_ms_p90"] - df["latencia_ms_p50"], color=colors)
axes[1].set(title=f"Latência p50 (barra até p90), batch 1, {NUM_THREADS} thread (ms)")
for (_, r), c in zip(df.iterrows(), colors):
    axes[2].scatter(r["latencia_ms_p50"], r["top1"] * 100, s=r["tamanho_MB"] * 15 + 30, color=c, label=r["modelo"])
axes[2].set(xlabel="latência p50 (ms)", ylabel="top-1 (%)", title="Acurácia × latência (área ∝ tamanho)")
axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
for ax in axes[:2]:
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{ARTIFACTS}/benchmark.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
("code", r"""
def to_markdown(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)

table = display_df[["modelo", "tamanho_MB", "latencia_ms_p50", "top1", "speedup", "delta_top1_pp"]]
display_df.to_csv(f"{ARTIFACTS}/benchmark.csv", index=False)
df_prune.round(4).to_csv(f"{ARTIFACTS}/pruning.csv", index=False)
meta = {"cpu": cpu_name(), "threads": NUM_THREADS, "torch": torch.__version__, "onnxruntime": ort.__version__,
        "runs": RUNS, "test_images": len(test_ds), "smoke": SMOKE}
with open(f"{ARTIFACTS}/benchmark_meta.json", "w") as f:
    json.dump(meta, f, indent=1)

print("Tabela pronta para colar no README:\n")
print(to_markdown(table))
print(f"\nMedido em: {meta['cpu']}, {NUM_THREADS} thread, batch 1, mediana de {RUNS} execuções.")
"""),
("md", """
## 8. Performance em diferentes configurações de execução

A latência depende do modelo **e** de onde e como ele roda. Aqui analiso três coisas.

### 8.1 MACs não preveem latência

O número de operações (MACs, multiplicações + acúmulos) é a medida de custo mais citada, mas não basta para prever o tempo. Convoluções *depthwise* têm poucos MACs, porém pouco reuso de dados: o gargalo passa a ser o **acesso à memória**, não a aritmética. Comparo arquiteturas eficientes (MobileNet, ShuffleNet, EfficientNet) com ResNets. Os pesos são aleatórios, já que a latência não depende dos valores.
"""),
("code", r"""
from torchvision import models as tvm

def count_macs(model, img=IMG_SIZE):
    macs = 0
    def conv_hook(m, inp, out):
        nonlocal macs
        macs += (out.numel() // out.shape[0]) * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def linear_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features
    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(torch.randn(1, 3, img, img))
    for h in hooks:
        h.remove()
    return macs

archs = {
    "MobileNetV3-Small": tvm.mobilenet_v3_small,
    "MobileNetV3-Large": tvm.mobilenet_v3_large,
    "ShuffleNetV2 1.0x": tvm.shufflenet_v2_x1_0,
    "EfficientNet-B0": tvm.efficientnet_b0,
    "ResNet-18": tvm.resnet18,
    "ResNet-50": tvm.resnet50,
}
arch_rows = []
torch.set_num_threads(NUM_THREADS)
for name, fn in archs.items():
    m = fn(weights=None).eval()
    with torch.inference_mode():
        lat = benchmark(m, sample_input, warmup=WARMUP // 2 or 1, runs=RUNS // 4 or 1)
    arch_rows.append({"arquitetura": name, "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
                      "GMACs": count_macs(m) / 1e9, "latencia_ms_p50": lat[0]})
    print(f"{name:18s} | {arch_rows[-1]['params_M']:5.1f} M params | {arch_rows[-1]['GMACs']:.2f} GMACs | {lat[0]:7.1f} ms")
torch.set_num_threads(os.cpu_count())
df_arch = pd.DataFrame(arch_rows)
df_arch["ms_por_GMAC"] = df_arch["latencia_ms_p50"] / df_arch["GMACs"]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df_arch["GMACs"], df_arch["latencia_ms_p50"], s=df_arch["params_M"] * 12 + 20)
for _, r in df_arch.iterrows():
    ax.annotate(r["arquitetura"], (r["GMACs"], r["latencia_ms_p50"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set(xlabel="GMACs por imagem", ylabel=f"latência p50 (ms, {NUM_THREADS} thread)", title="Custo teórico × latência medida (área ∝ parâmetros)")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
df_arch.round(3)
"""),
("md", """
A coluna `ms_por_GMAC` mostra quanto tempo cada bilhão de operações custa de verdade. Nas redes com convoluções *depthwise* esse custo costuma ser bem maior que nas ResNets: elas fazem menos contas, mas aproveitam pior o hardware.

### 8.2 Número de threads

Celulares têm vários núcleos, mas nem sempre vale usar todos: há custo de sincronização, e núcleos *big.LITTLE* são heterogêneos. Meço como a latência escala com o número de threads no ONNX Runtime.
"""),
("code", r"""
thread_opts = sorted({t for t in (1, 2, 4, 8) if t <= os.cpu_count()} | {os.cpu_count()})
thread_rows = []
for path, name in [(fp32_onnx, "ONNX FP32"), (static_onnx, "ONNX INT8 estático")]:
    for t in thread_opts:
        sess = make_session(path, t)
        lat = benchmark(lambda x: sess.run(None, {"input": x}), x_np, warmup=WARMUP // 2 or 1, runs=RUNS // 2 or 1)
        thread_rows.append({"modelo": name, "threads": t, "latencia_ms_p50": lat[0]})
df_threads = pd.DataFrame(thread_rows)
df_threads["speedup_vs_1"] = df_threads.groupby("modelo")["latencia_ms_p50"].transform(lambda s: s.iloc[0] / s)

fig, ax = plt.subplots(figsize=(6, 4))
for name, g in df_threads.groupby("modelo"):
    ax.plot(g["threads"], g["speedup_vs_1"], marker="o", label=name)
ax.plot(thread_opts, thread_opts, ls="--", color="gray", label="ideal (linear)")
ax.set(xlabel="threads", ylabel="speedup vs. 1 thread", title="Escalabilidade com número de threads (CPU)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
df_threads.round(2)
"""),
("md", """
### 8.3 Batch: latência × vazão, CPU × GPU

**Vazão** (imagens/s) e **latência** (ms por chamada) não andam juntas. Batches maiores aproveitam melhor o paralelismo e aumentam a vazão, mas cada chamada demora mais. Num app de câmera em tempo real o que importa é a latência com batch 1; num servidor processando um lote de fotos, a vazão. Na GPU comparo também FP32 com **FP16**, que usa os Tensor Cores.
"""),
("code", r"""
batch_opts = [1, 8, 32] if not SMOKE else [1, 4]
exec_rows = []

sess_all = make_session(fp32_onnx)  # todas as threads da CPU
for bs in batch_opts:
    xb = np.random.randn(bs, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    lat = benchmark(lambda x: sess_all.run(None, {"input": x}), xb, warmup=3, runs=10 if not SMOKE else 3)
    exec_rows.append({"execução": f"CPU ONNX FP32 ({os.cpu_count()} threads)", "batch": bs, "latencia_ms": lat[0], "img_por_s": bs / lat[0] * 1000})

if torch.cuda.is_available():
    for half in (False, True):
        m = copy.deepcopy(model_fp32).cuda().eval()
        dtype = torch.float16 if half else torch.float32
        m = m.to(dtype)
        def run(x):
            with torch.inference_mode():
                m(x)
            torch.cuda.synchronize()
        for bs in batch_opts + ([128] if not SMOKE else []):
            xb = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device="cuda", dtype=dtype)
            lat = benchmark(run, xb, warmup=10, runs=30)
            exec_rows.append({"execução": f"GPU PyTorch {'FP16' if half else 'FP32'} ({torch.cuda.get_device_name(0)})",
                              "batch": bs, "latencia_ms": lat[0], "img_por_s": bs / lat[0] * 1000})
        del m; torch.cuda.empty_cache()
else:
    print("Sem GPU neste ambiente: medindo só a CPU.")

df_exec = pd.DataFrame(exec_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, g in df_exec.groupby("execução"):
    axes[0].plot(g["batch"], g["latencia_ms"], marker="o", label=name)
    axes[1].plot(g["batch"], g["img_por_s"], marker="o", label=name)
axes[0].set(xscale="log", yscale="log", xlabel="batch", ylabel="latência por chamada (ms)", title="Latência")
axes[1].set(xscale="log", yscale="log", xlabel="batch", ylabel="imagens/s", title="Vazão (throughput)")
for ax in axes: ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{ARTIFACTS}/batch_cpu_gpu.png", dpi=150, bbox_inches="tight")
plt.show()

df_arch.round(3).to_csv(f"{ARTIFACTS}/arquiteturas.csv", index=False)
df_threads.round(3).to_csv(f"{ARTIFACTS}/threads.csv", index=False)
df_exec.round(2).to_csv(f"{ARTIFACTS}/batch_cpu_gpu.csv", index=False)
df_exec.round(2)
"""),
("md", """
## Conclusões

Leia a tabela acima e, depois de rodar, ajuste este texto com os seus números. O que costuma aparecer:

- **ONNX Runtime FP32** já costuma ser mais rápido que o PyTorch *eager* com batch 1, por causa da fusão de operadores e do menor overhead por chamada. A acurácia é idêntica.
- **INT8 estático** reduz o tamanho em cerca de 4× (float32 → int8) e, em CPUs com instruções int8 (AVX2/VNNI no x86, NEON/dotprod no ARM), também reduz a latência. A perda de acurácia costuma ficar em poucos pontos percentuais, e a quantização por canal é o que mantém essa perda pequena.
- **INT8 dinâmico** diminui o arquivo, mas tende a não ganhar velocidade (ou até perder) em CNNs: calcular a escala das ativações a cada inferência custa caro. Por isso a recomendação para visão é a quantização estática.
- **Pruning não estruturado** comprime bem o arquivo, mas **não acelera** a inferência densa. A acurácia se mantém até certo nível de esparsidade e depois cai rápido, porque a MobileNetV3 já é compacta e tem pouca redundância.
- **TorchScript** elimina o overhead do Python e funde operações; compare o ganho dele com o do ONNX Runtime.
- **MACs ≠ latência:** redes com convoluções *depthwise* gastam mais tempo por operação, porque o gargalo é a memória.
- **Threads** raramente escalam de forma linear em modelos pequenos com batch 1: há overhead de sincronização.
- **Batch:** a GPU só mostra a sua vantagem com batches grandes (vazão). Com batch 1, a CPU pode ser competitiva em latência, e isso pesa na escolha de onde rodar cada parte de um sistema heterogêneo.

### Limitações e próximos passos
- Medir num **celular de verdade** (ONNX Runtime Mobile, ou ExecuTorch / TFLite) e com aceleradores (NNAPI, Core ML, GPU do celular). É aí que entram os **sistemas heterogêneos**: decidir que parte do grafo roda na CPU, na GPU ou na NPU.
- Fazer **QAT** (*quantization-aware training*) para recuperar a acurácia perdida na quantização.
- Fazer **pruning estruturado** de canais + *fine-tuning* para ganhar latência de verdade.
- Testar **distilação de conhecimento**, usando um modelo maior (ResNet-50/ViT) como professor da MobileNet.
"""),
]

if __name__ == "__main__":
    build(f"../notebooks/{NAME}", cells)
