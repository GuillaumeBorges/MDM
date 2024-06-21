# MDM
Mineração de Dados Massivos - Mercado Financeiro


## Descrição dos Diretórios e Arquivos

- **data/**: Diretório para armazenar dados brutos, processados e externos.
  - **data/raw/**: Dados brutos.
  - **data/processed/**: Dados processados.
  - **data/external/**: Dados externos.
  
- **notebooks/**: Jupyter notebooks para exploração e análise inicial dos dados.
  - **01_exploratory_data_analysis.ipynb**: Análise exploratória dos dados.
  - **02_data_preprocessing.ipynb**: Pré-processamento dos dados.
  - **03_model_training.ipynb**: Treinamento do modelo.
  - **04_results_analysis.ipynb**: Análise dos resultados.

- **src/**: Diretório principal do código fonte do projeto.
  - **src/data/**: Módulo para carregamento e pré-processamento de dados.
    - **data_loader.py**: Funções para carregar os dados.
    - **data_preprocessing.py**: Funções de pré-processamento dos dados.
  - **src/models/**: Módulo para definição e treinamento do modelo de rede neural.
    - **model.py**: Definição do modelo de rede neural.
    - **train.py**: Funções de treinamento do modelo.
  - **src/utils/**: Módulo para funções auxiliares e utilitárias.
    - **helpers.py**: Funções auxiliares e utilitárias.
  - **src/visualize/**: Módulo para funções de visualização de dados.
    - **visualize.py**: Funções de visualização de dados.
  - **main.py**: Script principal para execução do projeto.

- **tests/**: Diretório para testes unitários e de integração.
  - **test_data_loader.py**: Testes para `data_loader.py`.
  - **test_data_preprocessing.py**: Testes para `data_preprocessing.py`.
  - **test_model.py**: Testes para `model.py`.
  - **test_train.py**: Testes para `train.py`.

- **config/**: Arquivos de configuração do projeto.
  - **config.yaml**: Arquivo de configuração principal.

- **results/**: Diretório para armazenar resultados e saídas do modelo.
  - **results/figures/**: Figuras geradas.
  - **results/logs/**: Logs de execução.

- **requirements.txt**: Lista de dependências do projeto.

- **README.md**: Documentação do projeto.
