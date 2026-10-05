"""Avaliação do modelo e geração de gráficos."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROTULOS = ["baixo_risco", "alto_risco"]


def avaliar_modelo(nome_modelo, y_real, y_previsto):
    """Calcula métricas (média ponderada) e matriz de confusão."""
    metricas = pd.DataFrame(
        [{
            "modelo": nome_modelo,
            "accuracy": accuracy_score(y_real, y_previsto),
            "precision": precision_score(y_real, y_previsto, average="weighted", zero_division=0),
            "recall": recall_score(y_real, y_previsto, average="weighted", zero_division=0),
            "f1_score": f1_score(y_real, y_previsto, average="weighted", zero_division=0),
        }]
    )
    matriz = confusion_matrix(y_real, y_previsto, labels=ROTULOS)
    relatorio = classification_report(y_real, y_previsto, labels=ROTULOS, zero_division=0)
    return metricas, matriz, relatorio


def salvar_metricas(metricas, caminho):
    """Salva o CSV de métricas."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    metricas.round(4).to_csv(caminho, index=False, encoding="utf-8")


def _finalizar(fig, caminho):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    return fig


def plotar_distribuicao_classes(base, caminho):
    """Gráfico 1: distribuição das classes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=base, x="situacao", order=ROTULOS, hue="situacao", legend=False, ax=ax)
    ax.set_title("Distribuição das classes (dados simulados)")
    ax.set_xlabel("Situação")
    ax.set_ylabel("Quantidade de frases")
    return _finalizar(fig, caminho)


def plotar_matriz_confusao(matriz, caminho):
    """Gráfico 2: matriz de confusão."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=ROTULOS, yticklabels=ROTULOS, ax=ax)
    ax.set_title("Matriz de confusão")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    return _finalizar(fig, caminho)


def plotar_metricas(metricas, caminho):
    """Gráfico 3: comparação de accuracy, precision, recall e F1."""
    valores = metricas.iloc[0][["accuracy", "precision", "recall", "f1_score"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(valores.index, valores.values, color="#4C78A8")
    ax.bar_label(barras, fmt="%.2f")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Métricas — {metricas.iloc[0]['modelo']}")
    ax.set_ylabel("Valor")
    return _finalizar(fig, caminho)
