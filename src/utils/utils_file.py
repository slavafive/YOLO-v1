import yaml
from yaml.loader import SafeLoader


def load_yaml(path: str | dict) -> dict:
    if isinstance(path, str):
        with open(path) as file:
            return yaml.load(file, Loader=SafeLoader)
    return path


def save_yaml(path: str, data: dict):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
