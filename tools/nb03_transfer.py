from nbbuild import build, colab_badge

NAME = "03_projeto_transfer_learning_mobilenetv3.ipynb"

cells = [
("md", f"""
# 03 · Projeto: classificador de raças de pets para dispositivos móveis (parte 1: treino)

{colab_badge(NAME)}

**Problema:** classificar **37 raças de cães e gatos** (dataset [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)) com um modelo leve o suficiente para rodar em **celular**.

**Por que MobileNetV3-Large?** Foi projetada para mobile: convoluções *depthwise separable*, blocos *inverted residual*, *squeeze-and-excitation* e ativação *hard-swish* (barata de calcular). Tem ~5,5 M parâmetros e ~0,22 GFLOPs, contra ~25 M e ~4 GFLOPs da ResNet-50.

**Etapas:**
1. *Transfer learning* em duas fases: primeiro só a cabeça, depois *fine-tuning* do modelo inteiro com LR diferente por grupo de camadas.
2. Avaliação: top-1, top-5, acurácia por classe e pares mais confundidos.
3. Interpretabilidade com **Grad-CAM** (implementado à mão).
4. Salvar o checkpoint para a **parte 2: otimização e benchmark** (notebook 04).

> No Colab, use GPU (T4). O checkpoint vai para o Google Drive, para o notebook 04 conseguir carregá-lo.
"""),
("code", """
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

SMOKE = os.environ.get("SMOKE_TEST") == "1"
IN_COLAB = "google.colab" in sys.modules
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = device == "cuda"

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ARTIFACTS = "/content/drive/MyDrive/dl-portfolio/artifacts"
    DATA_DIR = "/content/data"
else:
    ARTIFACTS, DATA_DIR = "../artifacts", "../data"
os.makedirs(ARTIFACTS, exist_ok=True)
print("device:", device, "| torch", torch.__version__)
"""),
("md", """
## 1. Dados

- `trainval` (3.680 imagens) é dividido em 90% treino e 10% validação. O `test` oficial (3.669 imagens) só é usado no final.
- No treino uso `RandomResizedCrop`, flip e `ColorJitter`; na avaliação, resize + crop central. A normalização usa a média e o desvio do ImageNet, porque o backbone foi pré-treinado nele.
"""),
("code", """
IMG_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

trainval_aug = datasets.OxfordIIITPet(DATA_DIR, split="trainval", download=True, transform=train_tf)
trainval_plain = datasets.OxfordIIITPet(DATA_DIR, split="trainval", download=False, transform=eval_tf)
test_ds = datasets.OxfordIIITPet(DATA_DIR, split="test", download=True, transform=eval_tf)
CLASSES = trainval_aug.classes
NUM_CLASSES = len(CLASSES)

perm = torch.randperm(len(trainval_aug), generator=torch.Generator().manual_seed(SEED)).tolist()
n_val = len(perm) // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]
if SMOKE:
    tr_idx, val_idx = tr_idx[:256], val_idx[:64]
    test_ds = Subset(test_ds, list(range(0, len(test_ds), 30)))
train_ds, val_ds = Subset(trainval_aug, tr_idx), Subset(trainval_plain, val_idx)

loader_kw = dict(num_workers=2, pin_memory=device == "cuda")
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, **loader_kw)
val_dl = DataLoader(val_ds, batch_size=128, **loader_kw)
test_dl = DataLoader(test_ds, batch_size=128, **loader_kw)
print(f"{NUM_CLASSES} classes | treino {len(train_ds)} | validação {len(val_ds)} | teste {len(test_ds)}")

def denorm(x):
    return (x.cpu() * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)).clamp(0, 1)

xb, yb = next(iter(train_dl))
fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
for ax, img, label in zip(axes.flat, xb, yb):
    ax.imshow(denorm(img).permute(1, 2, 0)); ax.set_title(CLASSES[label], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
("md", """
## 2. Modelo

Carrego a MobileNetV3-Large pré-treinada no ImageNet e troco a última camada linear (1000 → 37 classes).
"""),
("code", """
def build_model(num_classes):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(device)
count = lambda params: sum(p.numel() for p in params)
print(f"total: {count(model.parameters()) / 1e6:.2f} M parâmetros | backbone (features): {count(model.features.parameters()) / 1e6:.2f} M")
print(model.classifier)
"""),
("md", """
## 3. Treino em duas fases

**Fase 1 (linear probing):** backbone congelado, incluindo as estatísticas do BatchNorm (`features.eval()`). Treino só o classificador, com LR alto.

**Fase 2 (fine-tuning):** descongelo tudo. O backbone usa LR 10× menor que a cabeça, para não destruir as features aprendidas no ImageNet. Agendamento em cosseno.
"""),
("code", """
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def run_epoch(model, loader, optimizer=None, scheduler=None, freeze_backbone_bn=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_backbone_bn:
        model.features.eval()
    scaler = run_epoch.scaler
    total_loss, top1, top5, n = 0.0, 0, 0, 0
    preds_all, targets_all = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                if scheduler is not None:
                    scheduler.step()
            k5 = logits.topk(5, dim=1).indices
            top1 += (k5[:, 0] == y).sum().item()
            top5 += (k5 == y[:, None]).any(1).sum().item()
            total_loss += loss.item() * len(y); n += len(y)
            if not train:
                preds_all.append(k5[:, 0].cpu()); targets_all.append(y.cpu())
    out = {"loss": total_loss / n, "top1": top1 / n, "top5": top5 / n}
    if not train:
        out["preds"], out["targets"] = torch.cat(preds_all), torch.cat(targets_all)
    return out
run_epoch.scaler = torch.amp.GradScaler(enabled=USE_AMP)

history = []
def fit(epochs, optimizer, scheduler=None, phase="", freeze_backbone_bn=False):
    for ep in range(epochs):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_dl, optimizer, scheduler, freeze_backbone_bn)
        va = run_epoch(model, val_dl)
        history.append({"phase": phase, "train_loss": tr["loss"], "train_top1": tr["top1"], "val_loss": va["loss"], "val_top1": va["top1"]})
        print(f"[{phase}] época {ep + 1}/{epochs} | treino loss {tr['loss']:.3f} acc {tr['top1']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['top1']:.3f} | {time.perf_counter() - t0:.0f}s")
"""),
("code", """
EPOCHS_HEAD = 1 if SMOKE else 3
EPOCHS_FT = 1 if SMOKE else 8

# Fase 1: só o classificador
for p in model.features.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
fit(EPOCHS_HEAD, opt, phase="cabeça", freeze_backbone_bn=True)

# Fase 2: modelo inteiro, com LR diferente por grupo
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], epochs=EPOCHS_FT, steps_per_epoch=len(train_dl), pct_start=0.15)
fit(EPOCHS_FT, opt, sched, phase="fine-tuning")
"""),
("code", """
ep = range(1, len(history) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="treino")
axes[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="validação")
axes[1].plot(ep, [h["train_top1"] for h in history], marker="o", label="treino")
axes[1].plot(ep, [h["val_top1"] for h in history], marker="o", label="validação")
for ax, title in zip(axes, ["Loss", "Acurácia top-1"]):
    ax.axvline(EPOCHS_HEAD + 0.5, color="gray", ls="--", lw=1)
    ax.set(title=title, xlabel="época"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
("md", "## 4. Avaliação no conjunto de teste"),
("code", """
te = run_epoch(model, test_dl)
print(f"Teste → top-1: {te['top1']:.4f} | top-5: {te['top5']:.4f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(te["targets"], te["preds"], labels=range(NUM_CLASSES))
per_class = cm.diagonal() / cm.sum(1).clip(min=1)
order = np.argsort(per_class)
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh([CLASSES[i] for i in order], per_class[order]); ax.set(xlim=(0, 1), title="Acurácia por raça (teste)")
ax.grid(alpha=0.3, axis="x"); plt.tight_layout(); plt.show()

off = cm.copy(); np.fill_diagonal(off, 0)
flat = np.argsort(off, axis=None)[::-1][:8]
print("Pares mais confundidos (real → previsto):")
for f in flat:
    i, j = np.unravel_index(f, off.shape)
    if off[i, j]:
        print(f"  {CLASSES[i]:>28s} → {CLASSES[j]:<28s} {off[i, j]}")
"""),
("md", """
## 5. Interpretabilidade: Grad-CAM

O Grad-CAM mostra **onde** o modelo "olhou" para decidir. A ideia:
1. Pegar as ativações $A^k$ da última camada convolucional e o gradiente do score da classe em relação a elas.
2. O peso de cada canal é a média espacial do gradiente: $\\alpha_k = \\frac{1}{Z}\\sum_{i,j} \\frac{\\partial y^c}{\\partial A^k_{ij}}$.
3. O mapa é $\\text{ReLU}(\\sum_k \\alpha_k A^k)$, reamostrado para o tamanho da imagem.

Isso ajuda a verificar se o modelo usa o animal, e não o fundo, para decidir.
"""),
("code", """
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.eval(); self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx, logits.softmax(1)[0, class_idx].item()

cam = GradCAM(model, model.features[-1])
base_test = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
idx_map = test_ds.indices if isinstance(test_ds, Subset) else list(range(len(test_ds)))
rng = np.random.default_rng(0)
sample = rng.choice(len(idx_map), size=6, replace=False)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
for col, i in enumerate(sample):
    img, label = base_test[idx_map[i]]
    heat, pred, conf = cam(img[None].to(device))
    rgb = denorm(img).permute(1, 2, 0).numpy()
    axes[0, col].imshow(rgb); axes[0, col].set_title(f"real: {CLASSES[label]}", fontsize=8)
    axes[1, col].imshow(rgb); axes[1, col].imshow(heat, cmap="jet", alpha=0.45)
    axes[1, col].set_title(f"prev: {CLASSES[pred]} ({conf:.0%})", fontsize=8, color="green" if pred == label else "red")
for ax in axes.flat: ax.axis("off")
plt.tight_layout(); plt.show()
"""),
("md", "## 6. Salvar checkpoint"),
("code", """
ckpt = {
    "arch": "mobilenet_v3_large",
    "state_dict": model.state_dict(),
    "classes": CLASSES,
    "img_size": IMG_SIZE,
    "test_top1": te["top1"],
    "test_top5": te["top5"],
    "smoke": SMOKE,
}
path = f"{ARTIFACTS}/mobilenetv3_pets.pth"
torch.save(ckpt, path)
with open(f"{ARTIFACTS}/train_history.json", "w") as f:
    json.dump(history, f, indent=1)
print(f"checkpoint salvo em {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
"""),
("md", """
## Conclusões

- O *transfer learning* aproveita as features do ImageNet: com poucos milhares de imagens e poucos minutos de GPU, o modelo chega a uma acurácia alta em 37 classes de granularidade fina.
- A fase 1, com o backbone congelado, já dá uma boa base. O *fine-tuning* com LR menor no backbone melhora mais alguns pontos.
- Os erros se concentram em raças parecidas entre si, o que é esperado num problema *fine-grained*. Nos mapas de Grad-CAM, a região que mais pesa na decisão deve ser o animal (rosto, orelhas, pelagem), e não o fundo.

➡️ Próximo: [04 · Otimização para mobile: ONNX, quantização, pruning e benchmark](04_projeto_otimizacao_mobile.ipynb)
"""),
]

if __name__ == "__main__":
    build(f"../notebooks/{NAME}", cells)
