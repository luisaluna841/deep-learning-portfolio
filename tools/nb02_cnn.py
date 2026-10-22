from nbbuild import build, colab_badge

NAME = "02_cnn_cifar10.ipynb"

cells = [
("md", f"""
# 02 · CNN residual do zero no CIFAR-10

{colab_badge(NAME)}

**Objetivo:** projetar e treinar uma CNN própria com boas práticas modernas e analisar os erros dela, sem usar modelo pré-treinado.

- Arquitetura residual pequena (estilo ResNet) com BatchNorm
- Data augmentation (crop aleatório + flip)
- SGD com Nesterov + **OneCycleLR** + label smoothing
- **Mixed precision (AMP)** na GPU
- Avaliação: matriz de confusão, acurácia por classe, erros mais "confiantes"
- Medida de custo: parâmetros, tamanho e throughput

> No Colab: *Ambiente de execução → Alterar tipo de ambiente → GPU T4*.
"""),
("code", """
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from sklearn.metrics import confusion_matrix

SMOKE = os.environ.get("SMOKE_TEST") == "1"
IN_COLAB = "google.colab" in sys.modules
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DATA_DIR = "/content/data" if IN_COLAB else "../data"
device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = device == "cuda"
print("device:", device, "| torch", torch.__version__)
if device == "cuda":
    print(torch.cuda.get_device_name(0))
"""),
("md", """
## 1. Dados e augmentation

A validação usa imagens do treino **sem** augmentation; o conjunto de teste só é usado no final.
"""),
("code", """
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

train_aug = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
train_plain = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=eval_tf)
test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)
CLASSES = train_aug.classes

perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(SEED)).tolist()
val_idx, tr_idx = perm[:5000], perm[5000:]
if SMOKE:
    tr_idx, val_idx = tr_idx[:1000], val_idx[:200]
    test_ds = Subset(test_ds, range(200))
train_ds, val_ds = Subset(train_aug, tr_idx), Subset(train_plain, val_idx)

BATCH = 128
loader_kw = dict(num_workers=2, pin_memory=device == "cuda")
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, **loader_kw)
val_dl = DataLoader(val_ds, batch_size=256, **loader_kw)
test_dl = DataLoader(test_ds, batch_size=256, **loader_kw)
print(f"treino {len(train_ds)} | validação {len(val_ds)} | teste {len(test_ds)}")

def denorm(x):
    return (x * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)

xb, yb = next(iter(train_dl))
fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for ax, img, label in zip(axes.flat, xb, yb):
    ax.imshow(denorm(img).permute(1, 2, 0)); ax.set_title(CLASSES[label], fontsize=8); ax.axis("off")
plt.suptitle("Amostras com augmentation"); plt.tight_layout(); plt.show()
"""),
("md", """
## 2. Arquitetura

Blocos residuais: $y = \\text{ReLU}(F(x) + \\text{atalho}(x))$. O atalho deixa o gradiente fluir direto pelas camadas, o que facilita treinar redes mais profundas. Quando o bloco muda a resolução ou o número de canais, o atalho vira uma convolução 1×1.
"""),
("code", """
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks, in_ch = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResidualBlock(in_ch, w, stride=1 if i == 0 else 2))
            blocks.append(ResidualBlock(w, w))
            in_ch = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

model = SmallResNet().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parâmetros: {n_params / 1e6:.2f} M  (~{n_params * 4 / 1e6:.1f} MB em float32)")
"""),
("md", """
## 3. Treino

- **OneCycleLR:** o learning rate sobe até o máximo e depois cai em cosseno. Costuma convergir em menos épocas.
- **Label smoothing (0.1):** evita que o modelo fique confiante demais.
- **AMP:** forward e backward em float16 onde é seguro. Treina mais rápido e usa menos memória na GPU.
"""),
("code", """
EPOCHS = 1 if SMOKE else 20
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=EPOCHS, steps_per_epoch=len(train_dl))
scaler = torch.amp.GradScaler(enabled=USE_AMP)

def run_epoch(model, loader, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets, all_conf = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update(); scheduler.step()
            probs = logits.float().softmax(1)
            conf, preds = probs.max(1)
            total_loss += loss.item() * len(y); correct += (preds == y).sum().item(); n += len(y)
            if not train:
                all_preds.append(preds.cpu()); all_targets.append(y.cpu()); all_conf.append(conf.cpu())
    out = {"loss": total_loss / n, "acc": correct / n}
    if not train:
        out.update(preds=torch.cat(all_preds), targets=torch.cat(all_targets), conf=torch.cat(all_conf))
    return out

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
t0 = time.perf_counter()
for ep in range(EPOCHS):
    tr = run_epoch(model, train_dl, train=True)
    va = run_epoch(model, val_dl, train=False)
    for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]), ("val_loss", va["loss"]), ("val_acc", va["acc"])]:
        history[k].append(v)
    print(f"época {ep + 1:2d}/{EPOCHS} | treino loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
          f"val loss {va['loss']:.3f} acc {va['acc']:.3f} | {time.perf_counter() - t0:.0f}s")
"""),
("code", """
ep = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(ep, history["train_loss"], label="treino"); axes[0].plot(ep, history["val_loss"], label="validação")
axes[1].plot(ep, history["train_acc"], label="treino"); axes[1].plot(ep, history["val_acc"], label="validação")
axes[0].set(title="Loss", xlabel="época"); axes[1].set(title="Acurácia", xlabel="época")
for ax in axes: ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
("md", "## 4. Avaliação no teste"),
("code", """
te = run_epoch(model, test_dl, train=False)
print(f"Acurácia no teste: {te['acc']:.4f}")

cm = confusion_matrix(te["targets"], te["preds"], labels=range(10))
per_class = cm.diagonal() / cm.sum(1).clip(min=1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks(range(10), CLASSES, rotation=45, ha="right"); axes[0].set_yticks(range(10), CLASSES)
axes[0].set(xlabel="previsto", ylabel="real", title="Matriz de confusão")
for i in range(10):
    for j in range(10):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
axes[1].barh(CLASSES, per_class); axes[1].set(xlim=(0, 1), title="Acurácia por classe"); axes[1].grid(alpha=0.3, axis="x")
plt.tight_layout(); plt.show()

off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted(((off[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:5]
print("Confusões mais frequentes:")
for count, real, pred in pairs:
    print(f"  {real:>10s} → {pred:<10s} {count}")
"""),
("md", """
### Erros cometidos com alta confiança

São os casos mais interessantes de analisar: às vezes revelam rótulos errados no dataset, às vezes vieses do modelo (por exemplo, usar o fundo da imagem em vez do objeto).
"""),
("code", """
wrong = torch.where(te["preds"] != te["targets"])[0]
worst = wrong[te["conf"][wrong].argsort(descending=True)][:12]
base_test = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
idx_map = test_ds.indices if isinstance(test_ds, Subset) else range(len(test_ds))
fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
for ax, i in zip(axes.flat, worst.tolist()):
    img, _ = base_test[idx_map[i]]
    ax.imshow(denorm(img).permute(1, 2, 0)); ax.axis("off")
    ax.set_title(f"real: {CLASSES[te['targets'][i]]}\\nprev: {CLASSES[te['preds'][i]]} ({te['conf'][i]:.0%})", fontsize=8)
plt.tight_layout(); plt.show()
"""),
("md", "## 5. Custo computacional"),
("code", """
@torch.no_grad()
def throughput(model, batch_size, n_iters=20):
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    model.eval()
    for _ in range(5):
        model(x)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=USE_AMP):
            model(x)
    if device == "cuda": torch.cuda.synchronize()
    return batch_size * n_iters / (time.perf_counter() - t0)

print(f"parâmetros: {n_params / 1e6:.2f} M")
for bs in (1, 64, 256):
    print(f"batch {bs:3d}: {throughput(model, bs):8.0f} imagens/s ({device})")

ARTIFACTS = "/content/artifacts" if IN_COLAB else "../artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)
torch.save(model.state_dict(), f"{ARTIFACTS}/small_resnet_cifar10.pth")
print("modelo salvo em", ARTIFACTS)
"""),
("md", """
## Conclusões

- Uma ResNet pequena (~2,8 M parâmetros), treinada do zero em poucos minutos numa T4, já chega a uma acurácia alta no CIFAR-10.
- As confusões se concentram em classes visualmente parecidas (gato ↔ cachorro, automóvel ↔ caminhão).
- O throughput com batch 1 é muito menor que com batch grande: em dispositivos móveis a inferência costuma ser batch 1, então **latência** é a métrica que importa. É isso que otimizo no projeto principal.

**Possíveis extensões:** ablações (sem augmentation, sem label smoothing), Mixup/CutMix, mais épocas.

➡️ Próximo: [03 · Projeto: transfer learning com MobileNetV3](03_projeto_transfer_learning_mobilenetv3.ipynb)
"""),
]

if __name__ == "__main__":
    build(f"../notebooks/{NAME}", cells)
