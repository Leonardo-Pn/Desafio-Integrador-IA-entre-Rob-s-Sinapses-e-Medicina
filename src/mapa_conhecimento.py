"""Carregamento e consulta do mapa de conhecimento (associações simuladas)."""

from pathlib import Path

import pandas as pd

from src.preprocessing import normalizar_texto

COLUNAS_OBRIGATORIAS = ["sintoma_1", "sintoma_2", "condicao_associada"]


def carregar_mapa(caminho):
    """Carrega o CSV do mapa e valida colunas e valores ausentes."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Mapa de conhecimento não encontrado: {caminho}")
    mapa = pd.read_csv(caminho, encoding="utf-8")
    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in mapa.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no mapa: {faltantes}")
    if mapa[COLUNAS_OBRIGATORIAS].isna().any().any():
        raise ValueError("O mapa possui valores ausentes.")
    return mapa


def normalizar_mapa(mapa):
    """Cria colunas normalizadas das expressões de sintomas."""
    mapa = mapa.copy()
    mapa["sintoma_1_norm"] = mapa["sintoma_1"].apply(normalizar_texto)
    mapa["sintoma_2_norm"] = mapa["sintoma_2"].apply(normalizar_texto)
    return mapa


def buscar_associacoes(mapa_normalizado, expressao):
    """Retorna as linhas do mapa em que a expressão aparece (sintoma_1 ou sintoma_2)."""
    expr = normalizar_texto(expressao)
    filtro = (mapa_normalizado["sintoma_1_norm"] == expr) | (
        mapa_normalizado["sintoma_2_norm"] == expr
    )
    return mapa_normalizado[filtro]


def listar_condicoes(mapa_normalizado, expressao):
    """Retorna a lista de condições simuladas associadas a uma expressão."""
    encontrados = buscar_associacoes(mapa_normalizado, expressao)
    return sorted(encontrados["condicao_associada"].unique().tolist())


def expressoes_unicas(mapa_normalizado):
    """Retorna o conjunto de todas as expressões normalizadas do mapa."""
    return sorted(
        set(mapa_normalizado["sintoma_1_norm"]) | set(mapa_normalizado["sintoma_2_norm"])
    )
