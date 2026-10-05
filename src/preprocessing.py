"""Funções de pré-processamento de texto, compartilhadas pela Parte 1 e pela Parte 2."""

import re
import unicodedata

import pandas as pd


def validar_texto(texto):
    """Garante que o valor recebido é uma string não vazia.

    Levanta TypeError se não for string e ValueError se estiver vazia.
    """
    if not isinstance(texto, str):
        raise TypeError(f"Esperado str, recebido {type(texto).__name__}.")
    if not texto.strip():
        raise ValueError("O texto está vazio.")
    return texto


def remover_acentos(texto):
    """Remove acentos e diacríticos (ex.: 'cansaço' -> 'cansaco')."""
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def limpar_texto(texto):
    """Troca pontuação e símbolos por espaço e normaliza espaços repetidos."""
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_texto(texto):
    """Pipeline completo: valida, converte para minúsculas, remove acentos e limpa.

    Exemplo: 'Dificuldade Para Respirar!' -> 'dificuldade para respirar'
    """
    validar_texto(texto)
    texto = remover_acentos(texto.lower())
    return limpar_texto(texto)


def normalizar_serie(serie):
    """Aplica normalizar_texto a uma Series do pandas."""
    if not isinstance(serie, pd.Series):
        raise TypeError("Esperado pandas.Series.")
    return serie.apply(normalizar_texto)
