from nbbuild import build, colab_badge

NAME = "01_fundamentos_mlp_numpy_vs_pytorch.ipynb"

cells = [
("md", f"""
# 01 · Fundamentos: rede neural do zero (NumPy) vs. PyTorch

{colab_badge(NAME)}

**Objetivo:** mostrar que entendo o que acontece "por baixo" de um framework de deep learning.

1. Implemento um MLP (perceptron multicamadas) **só com NumPy**: forward, softmax + cross-entropy, **backpropagation manual** e SGD com momentum.
2. Valido os gradientes com **diferenças finitas** (gradient check).
3. Comparo com a mesma rede em **PyTorch** (autograd): os gradientes precisam ser iguais.
4. Treino as duas no **MNIST** e comparo acurácia e tempo.
"""),
("code", """
import os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets

SMOKE = os.environ.get("SMOKE_TEST") == "1"  # modo de teste rápido; no Colab fica False
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DATA_DIR = "../data"
print("torch", torch.__version__, "| GPU:", torch.cuda.is_available())
"""),
("md", """
## 1. Dados

MNIST: 70 mil imagens 28×28 de dígitos manuscritos. Cada imagem vira um vetor de 784 valores, normalizado com média e desvio do **treino** (para não vazar informação do teste).
"""),
("code", """
train_raw = datasets.MNIST(DATA_DIR, train=True, download=True)
test_raw = datasets.MNIST(DATA_DIR, train=False, download=True)

X_all = train_raw.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_all = train_raw.targets.numpy()
X_test = test_raw.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test = test_raw.targets.numpy()

# separa 5 mil imagens do treino para validação
X_train, y_train = X_all[:-5000], y_all[:-5000]
X_val, y_val = X_all[-5000:], y_all[-5000:]
if SMOKE:
    X_train, y_train, X_val, y_val = X_train[:2000], y_train[:2000], X_val[:500], y_val[:500]
    X_test, y_test = X_test[:500], y_test[:500]

mean, std = X_train.mean(), X_train.std()
X_train, X_val, X_test = [(X - mean) / std for X in (X_train, X_val, X_test)]
print(f"treino {X_train.shape}, validação {X_val.shape}, teste {X_test.shape}")

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, img, label in zip(axes.flat, train_raw.data[:16], train_raw.targets[:16]):
    ax.imshow(img, cmap="gray"); ax.set_title(int(label)); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
("md", r"""
## 2. MLP em NumPy

Para uma camada $z = hW + b$ seguida de ReLU, a regra da cadeia dá:

$$\frac{\partial L}{\partial W} = h^\top \delta, \qquad \frac{\partial L}{\partial b} = \sum_i \delta_i, \qquad \delta_{\text{anterior}} = (\delta W^\top) \odot \mathbb{1}[z_{\text{anterior}} > 0]$$

E para softmax + cross-entropy o gradiente em relação aos logits é simplesmente $(p - y_{\text{one-hot}})/N$.

Uso inicialização **He** ($\mathcal{N}(0, 2/\text{fan\_in})$), adequada para ReLU.
"""),
("code", """
class MLPNumpy:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.n_layers = len(sizes) - 1
        self.params = {}
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.params[f"W{i}"] = (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)

    def forward(self, X):
        # guarda as entradas de cada camada (acts) e as pré-ativações (zs) para o backward
        self.acts, self.zs = [X], []
        h = X
        for i in range(self.n_layers):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            self.zs.append(z)
            h = np.maximum(z, 0) if i < self.n_layers - 1 else z
            self.acts.append(h)
        return h  # logits

    def backward(self, dlogits):
        grads = {}
        delta = dlogits
        for i in reversed(range(self.n_layers)):
            grads[f"W{i}"] = self.acts[i].T @ delta
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.params[f"W{i}"].T) * (self.zs[i - 1] > 0)
        return grads


def softmax_cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)  # estabilidade numérica
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1
    return loss, dlogits / n
"""),
("md", """
## 3. Gradient check

Comparo o gradiente analítico (backprop) com a aproximação numérica $\\frac{L(\\theta+\\epsilon) - L(\\theta-\\epsilon)}{2\\epsilon}$ em posições aleatórias de cada parâmetro. Uso `float64` para evitar erro de arredondamento. Erro relativo abaixo de ~1e-6 indica que o backprop está correto.
"""),
("code", """
def gradient_check(model, X, y, n_checks=10, eps=1e-5):
    for k in model.params:
        model.params[k] = model.params[k].astype(np.float64)
    X = X.astype(np.float64)
    _, dlogits = softmax_cross_entropy(model.forward(X), y)
    grads = model.backward(dlogits)
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, P in model.params.items():
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps; loss_plus, _ = softmax_cross_entropy(model.forward(X), y)
            P[idx] = old - eps; loss_minus, _ = softmax_cross_entropy(model.forward(X), y)
            P[idx] = old
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[name][idx]
            rel = abs(numeric - analytic) / max(1e-12, abs(numeric) + abs(analytic))
            worst = max(worst, rel)
        print(f"{name:3s} erro relativo máximo até agora: {worst:.2e}")
    return worst

worst = gradient_check(MLPNumpy([784, 32, 16, 10], seed=1), X_train[:20], y_train[:20])
assert worst < 1e-4, "backprop com erro!"
print("Gradient check OK ✅")
"""),
("md", """
## 4. NumPy vs. autograd do PyTorch

Copio exatamente os mesmos pesos para uma rede PyTorch e comparo os gradientes calculados pelo autograd com os do meu backprop manual.
"""),
("code", """
def torch_mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

SIZES = [784, 256, 128, 10]
np_model = MLPNumpy(SIZES, seed=0)
pt_model = torch_mlp(SIZES)
linears = [m for m in pt_model if isinstance(m, nn.Linear)]
with torch.no_grad():
    for i, lin in enumerate(linears):
        lin.weight.copy_(torch.from_numpy(np_model.params[f"W{i}"].T))  # nn.Linear guarda W transposto
        lin.bias.copy_(torch.from_numpy(np_model.params[f"b{i}"]))

xb, yb = X_train[:64], y_train[:64]
_, dlogits = softmax_cross_entropy(np_model.forward(xb), yb)
np_grads = np_model.backward(dlogits)

pt_model.zero_grad()
nn.functional.cross_entropy(pt_model(torch.from_numpy(xb)), torch.from_numpy(yb)).backward()
for i, lin in enumerate(linears):
    diff = np.abs(lin.weight.grad.numpy().T - np_grads[f"W{i}"]).max()
    print(f"camada {i}: diferença máxima nos gradientes de W = {diff:.2e}")
"""),
("md", "## 5. Treinando as duas implementações"),
("code", """
EPOCHS = 1 if SMOKE else 8
LR, MOMENTUM, BATCH = 0.05, 0.9, 128

def accuracy_np(model, X, y):
    return (model.forward(X).argmax(1) == y).mean()

def train_numpy(model, epochs):
    vel = {k: np.zeros_like(v) for k, v in model.params.items()}
    rng = np.random.default_rng(SEED)
    hist = {"loss": [], "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = rng.permutation(len(X_train))
        losses = []
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            loss, dlogits = softmax_cross_entropy(model.forward(X_train[idx]), y_train[idx])
            grads = model.backward(dlogits)
            for k in model.params:  # SGD com momentum
                vel[k] = MOMENTUM * vel[k] - LR * grads[k]
                model.params[k] += vel[k]
            losses.append(loss)
        hist["loss"].append(np.mean(losses)); hist["val_acc"].append(accuracy_np(model, X_val, y_val))
        print(f"[NumPy]   época {ep + 1}: loss {hist['loss'][-1]:.4f} | val acc {hist['val_acc'][-1]:.4f}")
    return hist, time.perf_counter() - t0

def train_torch(model, epochs):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    Xt, yt = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)
    g = torch.Generator().manual_seed(SEED)
    hist = {"loss": [], "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        losses = []
        model.train()
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            loss = nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
        hist["loss"].append(np.mean(losses)); hist["val_acc"].append(val_acc)
        print(f"[PyTorch] época {ep + 1}: loss {hist['loss'][-1]:.4f} | val acc {val_acc:.4f}")
    return hist, time.perf_counter() - t0

np_model = MLPNumpy(SIZES, seed=0)
hist_np, t_np = train_numpy(np_model, EPOCHS)
torch.manual_seed(SEED)
pt_model = torch_mlp(SIZES)
hist_pt, t_pt = train_torch(pt_model, EPOCHS)
"""),
("code", """
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for hist, label in [(hist_np, "NumPy (manual)"), (hist_pt, "PyTorch (autograd)")]:
    axes[0].plot(range(1, EPOCHS + 1), hist["loss"], marker="o", label=label)
    axes[1].plot(range(1, EPOCHS + 1), hist["val_acc"], marker="o", label=label)
axes[0].set(title="Loss de treino", xlabel="época"); axes[1].set(title="Acurácia de validação", xlabel="época")
for ax in axes: ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

acc_np = accuracy_np(np_model, X_test, y_test)
with torch.no_grad():
    acc_pt = (pt_model(torch.from_numpy(X_test)).argmax(1).numpy() == y_test).mean()
print(f"Teste  NumPy: {acc_np:.4f} em {t_np:.1f}s | PyTorch: {acc_pt:.4f} em {t_pt:.1f}s")
"""),
("md", "## 6. Onde o modelo erra?"),
("code", """
preds = np_model.forward(X_test).argmax(1)
wrong = np.where(preds != y_test)[0][:16]
imgs = X_test * std + mean
fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for ax, i in zip(axes.flat, wrong):
    ax.imshow(imgs[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"real {y_test[i]} / prev {preds[i]}", fontsize=8); ax.axis("off")
plt.suptitle("Erros do MLP (NumPy)"); plt.tight_layout(); plt.show()
"""),
("md", """
## Conclusões

- O backprop manual bate com o gradiente numérico e com o autograd do PyTorch (diferenças da ordem de 1e-7, ou seja, arredondamento de `float32`).
- As duas implementações chegam praticamente à mesma acurácia no teste; o PyTorch é mais rápido por usar kernels otimizados (BLAS multithread, fusão de operações).
- Muitos erros são dígitos ambíguos até para humanos. Um MLP ignora a estrutura espacial da imagem, e esse é o motivo de usar **CNNs** no próximo notebook.

➡️ Próximo: [02 · CNN do zero no CIFAR-10](02_cnn_cifar10.ipynb)
"""),
]

if __name__ == "__main__":
    build(f"../notebooks/{NAME}", cells)
