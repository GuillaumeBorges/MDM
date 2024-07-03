import os

root = 'data/raw'


def read_path_dir(path: str) -> str:
    # Obter o caminho absoluto do diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Calcular o caminho absoluto do arquivo de configuração a partir do diretório raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


def read_path(path: str, file: str) -> str:
    project_root = read_path_dir(path)
    path = os.path.join(project_root, path, file)
    return path
