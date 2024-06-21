from typing import Any

import yaml
import os


# Função para carregar o arquivo de configuração
def load_config() -> Any:
    # Obter o caminho absoluto do diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Calcular o caminho absoluto do arquivo de configuração a partir do diretório raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    # Carregar o arquivo de configuração
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
