import os


def read_path(path: str, file: str) -> str:
    # Obter o caminho absoluto do diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Calcular o caminho absoluto do arquivo de configuração a partir do diretório raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    path = os.path.join(project_root, path, file)
    return path
