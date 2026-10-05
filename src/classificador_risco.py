"""Classificador de risco (simulado): TF-IDF + Regressão Logística."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import normalizar_texto

RANDOM_STATE = 42
CLASSES_VALIDAS = {"baixo_risco", "alto_risco"}


def carregar_base_risco(caminho):
    """Carrega e valida a base rotulada (colunas, ausentes, rótulos e duplicatas)."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Base de risco não encontrada: {caminho}")
    base = pd.read_csv(caminho, encoding="utf-8")
    if not {"frase", "situacao"}.issubset(base.columns):
        raise ValueError("A base precisa das colunas 'frase' e 'situacao'.")
    if base[["frase", "situacao"]].isna().any().any():
        raise ValueError("A base possui valores ausentes.")
    invalidos = set(base["situacao"]) - CLASSES_VALIDAS
    if invalidos:
        raise ValueError(f"Rótulos inválidos: {invalidos}")
    return base.drop_duplicates(subset="frase").reset_index(drop=True)


def preparar_base(base):
    """Adiciona a coluna 'frase_normalizada' usada pelo modelo."""
    base = base.copy()
    base["frase_normalizada"] = base["frase"].apply(normalizar_texto)
    return base


def dividir_treino_teste(base, test_size=0.20):
    """Divisão estratificada treino/teste com random_state fixo."""
    return train_test_split(
        base["frase_normalizada"],
        base["situacao"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=base["situacao"],
    )


def criar_pipeline():
    """Pipeline: TF-IDF (unigramas + bigramas) -> Regressão Logística."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def treinar_modelo(X_treino, y_treino):
    """Treina o pipeline e o retorna."""
    pipeline = criar_pipeline()
    pipeline.fit(X_treino, y_treino)
    return pipeline


def prever(modelo, frases):
    """Normaliza e classifica uma lista de frases novas; retorna DataFrame."""
    normalizadas = [normalizar_texto(f) for f in frases]
    return pd.DataFrame({"frase": list(frases), "classificacao_prevista": modelo.predict(normalizadas)})
