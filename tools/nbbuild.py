"""Utilitário mínimo para montar notebooks .ipynb a partir de células em texto."""
import json
from pathlib import Path

GITHUB_USER = "luisaluna841"
REPO = "deep-learning-portfolio"


def colab_badge(nb_name):
    url = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO}/blob/main/notebooks/{nb_name}"
    return f"[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def _source(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build(path, cells):
    nb_cells = []
    for kind, text in cells:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": _source(text)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("escrito:", path)
