# Sistema Inteligente de Análise de Sintomas e Classificação de Risco

> **Aviso:** este projeto possui finalidade **exclusivamente acadêmica e demonstrativa**. Os dados, associações, classificações e resultados são **simulados** e **não devem ser utilizados para diagnóstico, triagem ou tomada de decisão médica real**. O sistema não é uma ferramenta médica, não substitui profissionais de saúde e não identifica doenças reais.

## Descrição

Projeto de NLP e Machine Learning que analisa relatos textuais **simulados** de sintomas e executa duas tarefas independentes, compartilhando o mesmo módulo de pré-processamento.

## Objetivo

1. **Parte 1 — Extração e associação de sintomas:** normaliza frases, identifica sintomas com um mapa de conhecimento, associa-os a condições genéricas (fictícias), pontua cada condição (+1 por correspondência) e indica a `condicao_principal`. Em caso de **empate**, todas as condições empatadas são exibidas.
2. **Parte 2 — Classificação de risco:** classifica frases como `baixo_risco` ou `alto_risco` com **TF-IDF (unigramas + bigramas) + Regressão Logística**, avalia o desempenho e testa o modelo com frases novas.

## Arquitetura

```text
Dados
 ↓
Pré-processamento
 ↓
 ┌───────────────┬───────────────┐
 │               │               │
Extração       TF-IDF
 │               │
Mapa            Machine Learning
 │               │
Associação     Classificação
 │               │
 └───────────────┴───────────────┘
                 ↓
             Avaliação
```

## Tecnologias

Python · Pandas · NumPy · Scikit-learn · TF-IDF · Logistic Regression · Matplotlib · Seaborn · Jupyter

## Estrutura

```text
sistema_analise_sintomas/
├── data/
│   ├── raw/
│   │   ├── sintomas.txt
│   │   ├── mapa_conhecimento.csv
│   │   └── base_risco.csv
│   └── processed/
│       └── resultados_processamento.csv
├── notebooks/
│   └── sistema_analise_sintomas.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── mapa_conhecimento.py
│   ├── extracao_sintomas.py
│   ├── classificador_risco.py
│   └── avaliacao.py
├── results/
│   ├── figuras/
│   │   ├── distribuicao_classes.png
│   │   ├── matriz_confusao.png
│   │   └── metricas_modelo.png
│   ├── resultados_extracao.csv
│   └── metricas_modelo.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

Abra o notebook e execute todas as células, da primeira à última:

```text
notebooks/sistema_analise_sintomas.ipynb
```

```bash
jupyter notebook notebooks/sistema_analise_sintomas.ipynb
```

O notebook localiza a raiz do projeto automaticamente (usa apenas caminhos relativos) e importa a lógica de `src/`.

## Resultados

Gerados automaticamente pelo notebook:

- `results/resultados_extracao.csv` e `data/processed/resultados_processamento.csv`: sintomas, condições, condição principal e pontuação por frase;
- `results/metricas_modelo.csv`: accuracy, precision, recall e F1-score (média ponderada);
- `results/figuras/`: distribuição das classes, matriz de confusão e comparação de métricas.

Execução de referência (`random_state=42`, 63 frases, 13 no teste): accuracy, precision, recall e F1 ≈ **0,85**. Como o conjunto de teste é muito pequeno, esses valores são instáveis e servem apenas como demonstração.

## Limitações

- Dados **simulados e pequenos**, escritos para fins didáticos; não representam relatos reais.
- O mapa de conhecimento contém associações **fictícias**, sem validade clínica.
- A extração usa correspondência exata de expressões (sem sinônimos não cadastrados, negação ou contexto).
- O classificador aprende palavras indicativas do conjunto simulado; não há validação cruzada, ajuste de hiperparâmetros nem validação clínica.

## Aviso

Este sistema **não possui finalidade médica real**. Não use seus resultados para diagnóstico, triagem ou qualquer decisão sobre saúde. Em caso de sintomas reais, procure um profissional de saúde.
