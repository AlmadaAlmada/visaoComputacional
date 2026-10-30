import os

import yaml


def carregar_config(caminho):
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def salvar_config(cfg, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def resolver_caminho(caminho_config, caminho):
    if os.path.isabs(caminho) or os.path.exists(caminho):
        return caminho
    candidato = os.path.join(os.path.dirname(os.path.abspath(caminho_config)), caminho)
    return candidato if os.path.exists(candidato) else caminho
