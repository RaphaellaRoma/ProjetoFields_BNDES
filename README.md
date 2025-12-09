# ProjetoFields_BNDES

**Descrição**
- **ProjetoFields_BNDES** é uma aplicação para classificação de normativos com modelos de ML (BERT e uma rede neural) que avaliam relevância e aplicabilidade ao BNDES. A interface gráfica está em `Arquivos finais/interface.py`.

Sistema para **classificação automatizada de normativos** utilizando modelos de NLP.  
O projeto combina modelos tradicionais de Machine Learning com um modelo BERT fine-tunado, além de oferecer uma **interface gráfica em Tkinter** para facilitar o uso final.
O objetivo do ProjetoFields_BNDES é apoiar o processo de análise de normativos, classificando-os em diferentes dimensões, como:

- **Relevância**  
- **Aplicabilidade ao BNDES** 

**Requisitos**
- Python 3.8+ (recomendado 3.10/3.11+)
- Bibliotecas principais: `customtkinter`, `Pillow` (outras dependências dos modelos podem ser necessárias, ex.: `torch`, `transformers` se usar BERT real).

**Instalação rápida (Windows - PowerShell)**

```powershell
cd C:\Users\raphy\ProjetoFields_BNDES
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

```

**Executando a interface**

Abra o PowerShell com o virtualenv ativado e rode:

```powershell
python "Arquivos finais\interface.py"
```

Observações:
- O arquivo carrega funções de `predict_bert` e `predicao_aplicavel_assunto`. Garanta que esses módulos (e os modelos salvos em `Modelos/`) estejam presentes e configurados.



**Estrutura principal do repositório**
- `Arquivos finais/` — scripts finais, incluindo `interface.py`.
- `Dados CSV/` e `Dados Excel/` — bases e dados de entrada.
- `Modelos/` — modelos salvos (p. ex. `modelo_bert_salvo`).
- `Testes Modelos ML/` — scripts de comparação e testes de modelos.



**Documentação Completa do Repositório**

Visão geral rápida
- Projeto para classificação de normativos (relevância e aplicabilidade) usando modelos ML (BERT e redes neurais). Contém scripts finais de interface, scripts de geração/limpeza de dados, modelos salvos e notebooks/scripts de testes/comparação.

Estrutura do repositório
- `Arquivos finais/` — scripts prontos para uso: `interface.py`, `predict_bert.py`, `predicao_aplicavel_assunto.py`, entre outros.
- `Dados CSV/` — dados CSV utilizados no projeto (ex.: `df_assunto_limpo.csv`).
- `Dados Excel/` — planilhas originais (se houver).
- `Gerar Banco de Dados/` — scripts para coletar e unificar dados (ex.: `df.py`, `df_assunto.py`, `unindo_bases.py`).
- `Modelos/` — modelos treinados salvos (ex.: `modelo_nn_salvo/`).
- `Testes Modelos ML/` — scripts de teste e comparação de modelos (Logistic Regression, Random Forest, NN, XGBoost, métricas e visualizações).
- `limpar_modelo.py` — utilitários para limpeza de modelos/artefatos.

## Dependências

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Observações sobre dependências adicionais
- Se você usar modelos BERT reais, precisará de uma GPU/CPU compatível e versões compatíveis de `torch` e `transformers`. Ajuste `requirements.txt` conforme sua plataforma (ex.: instalar `torch` com o comando do site oficial para suporte CUDA).

Como executar a interface GUI
- Com o virtualenv ativado:

```powershell
python "Arquivos finais\interface.py"
```

Como rodar os scripts de geração de dados
- Scripts em `Gerar Banco de Dados/` geram e unificam bases. Exemplos:

```powershell
python "Gerar Banco de Dados\df.py"
python "Gerar Banco de Dados\df_assunto.py"
```

Modelos e predições
- `Arquivos finais/predict_bert.py` — wrapper para carregar tokenizer/modelo e realizar predições com BERT.
- `Arquivos finais/predicao_aplicavel_assunto.py` — usa `joblib` para carregar transformadores (TF-IDF) e o classificador NN salvo.
- Verifique a pasta `Modelos/` para ver os arquivos de modelos salvos; se não existir, rode os scripts de treino em `Testes Modelos ML/` para gerar modelos.


Dados (bancos)
- Os dados limpos estão em `Dados CSV/df_assunto_limpo.csv`. 
- Para gerar `Dados CSV/df_assunto_limpo.csv` é necessário rodar o arquivo `Gerar Banco de Dados/df.py`.

Modelos
`Modelos\modelo_nn_salvo` modelo de rede neural para classificação binaria de aplicabilidade.
`Modelos\modelo_bert_salvo` modelo bert para classificação de relevância


## Autores

- **Beatriz Marques**
- **Gabrielle Mascarelo**
- **Raphaella Roma**
# ProjetoFields_BNDES
