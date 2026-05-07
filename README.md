# Machine Learning Prático com Python
## Da Exploração de Dados ao Deploy de Modelos

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Sobre

Repositório com todos os exemplos de código do livro **"Machine Learning Prático com Python"**. Cada capítulo está organizado em um diretório próprio. Alguns scripts dependem da execução sequencial dos arquivos anteriores do mesmo capítulo.

## Estrutura do Repositório

```
ml-pratico-python/
├── README.md
├── requirements.txt
├── cap01-fundamentos-python/
│   ├── 01_estruturas_de_dados.py
│   ├── 02_controle_de_fluxo.py
│   ├── 03_funcoes_e_lambda.py
│   ├── 04_list_comprehension.py
│   └── 05_classes_e_oop.py
├── cap02-numpy-pandas-visualizacao/
│   ├── 01_numpy_operacoes_essenciais.py
│   ├── 02_pandas_manipulacao_dados.py
│   ├── 03_pandas_transformacoes.py
│   └── 04_visualizacao_matplotlib_seaborn.py
├── cap03-analise-exploratoria/
│   ├── 01_importacao_e_inspecao.py
│   ├── 02_tratamento_valores_ausentes.py
│   ├── 03_deteccao_outliers.py
│   └── 04_feature_engineering.py
├── cap04-primeiro-modelo/
│   ├── 01_classificacao_vinho.py
│   ├── 02_avaliacao_matriz_confusao.py
│   ├── 03_validacao_cruzada_overfitting.py
│   └── 04_regressao_california_housing.py
├── cap05-pipelines-ml/
│   ├── 01_pipeline_feature_selection_tuning.py
│   └── 02_curvas_roc_precisao_recall.py
├── cap06-dados-desbalanceados/
│   ├── 01_tecnicas_desbalanceamento.py
│   └── 02_comparacao_visual_estrategias.py
├── cap07-explicabilidade-xai/
│   ├── 01_feature_importance_permutation.py
│   └── 02_shap_explicabilidade.py
├── cap08-deep-learning/
│   ├── 01_rede_neural_dados_tabulares.py
│   └── 02_cnn_mnist.py
├── cap09-projetos-dados-reais/
│   ├── 01_predicao_churn_pipeline.py
│   └── 02_visualizacao_resultados_churn.py
└── cap10-transfer-learning/
    └── 01_transfer_learning_mobilenet.py
```

## Capítulos

| # | Capítulo | Descrição |
|---|---|---|
| 1 | Fundamentos de Python | Estruturas de dados, funções, classes e padrões OOP para ML |
| 2 | NumPy, Pandas e Visualização | Ferramentas essenciais para ciência de dados |
| 3 | Análise Exploratória (EDA) | Inspeção, nulos, outliers e feature engineering |
| 4 | Primeiro Modelo de ML | Classificação, regressão, validação cruzada |
| 5 | Pipelines de ML | Métricas, feature selection, tuning, curvas ROC |
| 6 | Dados Desbalanceados | SMOTE, class weights, estratégias de reamostragem |
| 7 | IA Explicável (XAI) | Feature importance, permutation importance, SHAP |
| 8 | Deep Learning | Redes neurais densas e CNNs com TensorFlow/Keras |
| 9 | Projetos com Dados Reais | Pipeline end-to-end de predição de churn |
| 10 | Transfer Learning | Feature extraction e fine-tuning com MobileNetV2 |

## Como Usar

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/ml-pratico-python.git
cd ml-pratico-python

# Instalar dependências
pip install -r requirements.txt

# Executar qualquer exemplo
python cap01-fundamentos-python/01_estruturas_de_dados.py
```

## Pré-requisitos

- Python 3.9+
- Ambiente virtual recomendado (venv ou conda)

## Dependências Principais

- **NumPy** — Computação numérica e arrays
- **Pandas** — Manipulação de dados tabulares
- **Matplotlib / Seaborn** — Visualização de dados
- **Scikit-learn** — Machine Learning clássico
- **TensorFlow / Keras** — Deep Learning
- **SHAP** — Explicabilidade de modelos
- **Imbalanced-learn** — Técnicas para dados desbalanceados

## Convenções

- Termos técnicos em inglês mantidos no original (*overfitting*, *feature engineering*, *cross-validation*)
- Variáveis de negócio em português (`renda_mensal`, `tempo_emprego`)
- APIs e parâmetros em inglês (`learning_rate`, `train_test_split`)

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

**Autor:** Wiliam Rosa  
**Livro:** Machine Learning Prático com Python — Da Exploração de Dados ao Deploy de Modelos
