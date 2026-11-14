"""Código compartilhado entre os notebooks (modelo, classes, dados, carregamento de runs)."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------- dispositivo


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- classes


def cifar100_classes(data_dir="data"):
    return datasets.CIFAR100(data_dir, train=False, download=True).classes


def resolve_classes(spec, all_classes, seed=0):
    """Converte a especificação de classes em uma lista ordenada de índices originais."""
    if spec is None:
        return list(range(len(all_classes)))
    if isinstance(spec, int):
        rng = random.Random(seed)
        return sorted(rng.sample(range(len(all_classes)), spec))
    idxs = []
    for c in spec:
        if isinstance(c, str):
            if c not in all_classes:
                raise ValueError(f"Classe desconhecida: {c!r}")
            idxs.append(all_classes.index(c))
        else:
            if not 0 <= int(c) < len(all_classes):
                raise ValueError(f"Índice fora do intervalo: {c}")
            idxs.append(int(c))
    if len(set(idxs)) != len(idxs):
        raise ValueError("Classes repetidas na seleção")
    return sorted(idxs)


def classes_dirname(names, max_len=150):
    """Nome da pasta a partir das classes (ordenadas por índice, unidas por '-').
    Se ficar longo demais para o sistema de arquivos, usa '<K>classes_<hash>'."""
    name = "-".join(names)
    if len(name) <= max_len:
        return name
    return f"{len(names)}classes_{hashlib.sha1(name.encode()).hexdigest()[:10]}"


# ----------------------------------------------------------------------------- modelo

VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG16CIFAR(nn.Module):
    """VGG16 com BatchNorm para entradas 32x32 (após 5 poolings o mapa fica 1x1x512)."""

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        layers, in_ch = [], 3
        for v in VGG16_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


def build_model(variant: str, num_classes: int, dropout: float = 0.5, pretrained: bool = True) -> nn.Module:
    if variant == "cifar":
        return VGG16CIFAR(num_classes, dropout)
    if variant == "imagenet":
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vgg16_bn(weights=weights, dropout=dropout)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"model_variant desconhecido: {variant}")


def conv_layers(model: nn.Module) -> dict:
    """Nomeia as camadas conv da VGG16 como 'conv<bloco>_<i>' (ex.: conv1_1 ... conv5_3).

    Retorna {nome: (conv, bn, relu)} — funciona para as duas variantes, já que ambas
    usam `model.features` com o padrão Conv -> BN -> ReLU (+ MaxPool entre blocos).
    """
    out, block, i = {}, 1, 0
    mods = list(model.features)
    for j, m in enumerate(mods):
        if isinstance(m, nn.MaxPool2d):
            block, i = block + 1, 0
        elif isinstance(m, nn.Conv2d):
            i += 1
            out[f"conv{block}_{i}"] = (m, mods[j + 1], mods[j + 2])
    return out


# ----------------------------------------------------------------------------- dados


def norm_stats(variant: str):
    return (IMAGENET_MEAN, IMAGENET_STD) if variant == "imagenet" else (CIFAR100_MEAN, CIFAR100_STD)


def eval_transform(cfg: dict):
    mean, std = norm_stats(cfg["model_variant"])
    size = cfg["img_size"]
    resize = [transforms.Resize(size, antialias=True)] if size != 32 else []
    return transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])


# ----------------------------------------------------------------------------- runs salvos


def find_run_dir(usuario, teste, classes=None, root=".", data_dir="data", seed=42) -> Path:
    """Localiza a pasta ./usuario/teste/<classes>. Com classes=None, usa a única run existente."""
    base = Path(root) / usuario / teste
    if not base.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {base.resolve()}")
    runs = sorted(p for p in base.iterdir() if (p / "model.pt").is_file())
    if classes is None:
        if len(runs) == 1:
            return runs[0]
        listing = "\n".join(f"  - {p.name}" for p in runs) or "  (nenhuma)"
        raise ValueError(f"Há {len(runs)} runs em {base}; defina CLASSES. Disponíveis:\n{listing}")
    all_classes = cifar100_classes(data_dir)
    names = [all_classes[i] for i in resolve_classes(classes, all_classes, seed)]
    run_dir = base / classes_dirname(names)
    if not (run_dir / "model.pt").is_file():
        listing = "\n".join(f"  - {p.name}" for p in runs) or "  (nenhuma)"
        raise FileNotFoundError(f"Sem model.pt em {run_dir}. Runs disponíveis em {base}:\n{listing}")
    return run_dir


def load_run(run_dir, device="cpu"):
    """Carrega o checkpoint de uma run. Retorna (model em eval, ckpt)."""
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg["model_variant"], len(ckpt["selected"]), cfg.get("dropout", 0.5), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
