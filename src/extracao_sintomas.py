"""Extração de sintomas e associação simulada a condições, com pontuação."""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.mapa_conhecimento import expressoes_unicas, listar_condicoes
from src.preprocessing import normalizar_texto


def carregar_frases(caminho):
    """Lê o arquivo de texto e retorna uma frase por linha (linhas vazias ignoradas)."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de sintomas não encontrado: {caminho}")
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    return [linha.strip() for linha in linhas if linha.strip()]


def encontrar_sintomas(texto_normalizado, expressoes):
    """Retorna as expressões do mapa presentes no texto (busca por palavra inteira)."""
    return [
        expr
        for expr in expressoes
        if re.search(rf"\b{re.escape(expr)}\b", texto_normalizado)
    ]


def calcular_pontuacao(sintomas, mapa_normalizado):
    """Cada sintoma encontrado soma +1 para cada condição associada a ele."""
    pontos = Counter()
    for sintoma in sintomas:
        for condicao in listar_condicoes(mapa_normalizado, sintoma):
            pontos[condicao] += 1
    return pontos


def definir_principal(pontos):
    """Retorna (condições_principais, pontuação). Em empate, retorna todas as empatadas."""
    if not pontos:
        return [], 0
    maximo = max(pontos.values())
    principais = sorted(c for c, p in pontos.items() if p == maximo)
    return principais, maximo


def extrair_sintomas(frase, mapa_normalizado):
    """Executa o fluxo completo para uma frase e retorna um dicionário de resultado."""
    normalizada = normalizar_texto(frase)
    sintomas = encontrar_sintomas(normalizada, expressoes_unicas(mapa_normalizado))
    pontos = calcular_pontuacao(sintomas, mapa_normalizado)
    principais, pontuacao = definir_principal(pontos)
    return {
        "frase_original": frase,
        "frase_normalizada": normalizada,
        "sintomas_identificados": "; ".join(sintomas),
        "condicoes_encontradas": "; ".join(f"{c} ({p})" for c, p in pontos.most_common()),
        "condicao_principal": " | ".join(principais) if principais else "Nenhuma associação",
        "pontuacao": pontuacao,
    }


def processar_frases(frases, mapa_normalizado):
    """Processa uma lista de frases e retorna um DataFrame com id e resultados."""
    registros = [extrair_sintomas(f, mapa_normalizado) for f in frases]
    df = pd.DataFrame(registros)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def salvar_resultados(df, caminho_processado, caminho_resultados):
    """Salva a versão completa (processed) e a versão resumida (results)."""
    Path(caminho_processado).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho_resultados).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_processado, index=False, encoding="utf-8")
    resumo = df.rename(columns={"frase_original": "frase"})[
        ["frase", "sintomas_identificados", "condicoes_encontradas",
         "condicao_principal", "pontuacao"]
    ]
    resumo.to_csv(caminho_resultados, index=False, encoding="utf-8")
    return resumo
