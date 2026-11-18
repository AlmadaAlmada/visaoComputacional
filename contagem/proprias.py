import glob
import os
import random
import shutil

import cv2
import pandas as pd


def recortar(img, x1, y1, x2, y2, margem=0.05):
    altura, largura = img.shape[:2]
    dx, dy = (x2 - x1) * margem, (y2 - y1) * margem
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(largura, int(x2 + dx)), min(altura, int(y2 + dy))
    return img[y1:y2, x1:x2]


def dividir_por_grupo(anotacoes, fracao_teste=0.4, semente=42):
    rng = random.Random(semente)
    teste = set()
    grupos = anotacoes.groupby("classe")["grupo"].unique()
    for classe in sorted(grupos.index):
        lista = sorted(grupos[classe])
        rng.shuffle(lista)
        teste.update(lista[:max(1, round(len(lista) * fracao_teste))])
    return anotacoes.assign(divisao=["teste" if g in teste else "treino" for g in anotacoes["grupo"]])


def gravar_recortes(anotacoes, pasta_fotos, saida, margem=0.05):
    caminhos = []
    imagens = {}
    for i, r in enumerate(anotacoes.itertuples(index=False)):
        if r.foto not in imagens:
            imagens[r.foto] = cv2.imread(os.path.join(pasta_fotos, r.foto))
        destino = os.path.join(saida, r.divisao, r.classe)
        os.makedirs(destino, exist_ok=True)
        caminho = os.path.join(destino, f"{os.path.splitext(r.foto)[0]}_{i:02d}.jpg")
        cv2.imwrite(caminho, recortar(imagens[r.foto], r.x1, r.y1, r.x2, r.y2, margem))
        caminhos.append(caminho)
    return anotacoes.assign(recorte=caminhos)


def _vincular(origem, destino):
    try:
        os.link(origem, destino)
    except OSError:
        shutil.copy2(origem, destino)


def montar_conjunto(base, recortes_treino, saida, repeticoes):
    if os.path.exists(saida):
        shutil.rmtree(saida)
    for divisao in ("train", "val"):
        for arquivo in glob.glob(os.path.join(base, divisao, "*", "*")):
            classe = os.path.basename(os.path.dirname(arquivo))
            destino = os.path.join(saida, divisao, classe)
            os.makedirs(destino, exist_ok=True)
            _vincular(arquivo, os.path.join(destino, os.path.basename(arquivo)))
    for r in recortes_treino.itertuples(index=False):
        destino = os.path.join(saida, "train", r.classe)
        os.makedirs(destino, exist_ok=True)
        nome = os.path.splitext(os.path.basename(r.recorte))[0]
        for k in range(repeticoes):
            _vincular(r.recorte, os.path.join(destino, f"proprias_{nome}_r{k:02d}.jpg"))


def preparar(arquivo_anotacoes, pasta_fotos, saida, base_mio, saida_conjunto, fracao_teste=0.4, repeticoes=20,
             semente=42, margem=0.05):
    anotacoes = dividir_por_grupo(pd.read_csv(arquivo_anotacoes), fracao_teste, semente)
    if os.path.exists(saida):
        shutil.rmtree(saida)
    anotacoes = gravar_recortes(anotacoes, pasta_fotos, saida, margem)
    anotacoes.drop(columns="recorte").to_csv(os.path.join(saida, "divisao.csv"), index=False, encoding="utf-8")
    montar_conjunto(base_mio, anotacoes[anotacoes["divisao"] == "treino"], saida_conjunto, repeticoes)
    return anotacoes
