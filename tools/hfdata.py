"""Célula compartilhada pelos notebooks 02-04: datasets lidos de espelhos Parquet no Hugging Face Hub.

Os servidores originais do CIFAR-10 (Toronto) e do Oxford-IIIT Pet (Oxford) ficaram muito lentos;
os espelhos no Hugging Face têm os mesmos splits e rótulos.
"""

CELL = r'''
import io, copy, urllib.request
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

HF = "https://huggingface.co/datasets"
NUM_WORKERS = 2 if os.name != "nt" else 0  # no Windows, workers não enxergam classes definidas no notebook

class ParquetImageDataset(Dataset):
    """Imagens + rótulos lidos de um arquivo Parquet (formato dos datasets do Hugging Face Hub)."""

    def __init__(self, url, filename, root, image_col, classes, transform=None):
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            print(f"baixando {url}")
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        df = pd.read_parquet(path, columns=[image_col, "label"])
        self.images = [d["bytes"] for d in df[image_col]]  # JPEG/PNG comprimido, decodificado sob demanda
        self.targets = df["label"].tolist()
        self.classes, self.transform = classes, transform

    def with_transform(self, transform):
        """Mesma base (sem copiar as imagens) com outro transform: útil para treino com e sem augmentation."""
        new = copy.copy(self)
        new.transform = transform
        return new

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        return (self.transform(img) if self.transform else img), self.targets[i]
'''

CIFAR10 = r'''
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_cifar10(split, transform=None):
    return ParquetImageDataset(f"{HF}/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet",
                               f"cifar10_{split}.parquet", DATA_DIR, "img", CIFAR10_CLASSES, transform)
'''

PETS = r'''
PET_CLASSES = ["abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "bengal",
               "birman", "bombay", "boxer", "british_shorthair", "chihuahua", "egyptian_mau", "english_cocker_spaniel",
               "english_setter", "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin", "keeshond",
               "leonberger", "maine_coon", "miniature_pinscher", "newfoundland", "persian", "pomeranian", "pug",
               "ragdoll", "russian_blue", "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "siamese",
               "sphynx", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"]
PET_CLASSES = [" ".join(w.capitalize() for w in c.split("_")) for c in PET_CLASSES]

def load_pets(split, transform=None):
    """split: "trainval" (3.680 imagens) ou "test" (3.669), iguais aos splits oficiais."""
    hf_split = "train" if split == "trainval" else "test"
    return ParquetImageDataset(f"{HF}/timm/oxford-iiit-pet/resolve/main/data/{hf_split}-00000-of-00001.parquet",
                               f"pets_{hf_split}.parquet", DATA_DIR, "image", PET_CLASSES, transform)
'''
