import argparse
import glob
import os
from collections import defaultdict

import cv2
import pandas as pd


def borrar(img, x1, y1, x2, y2):
    altura, largura = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(largura, int(x2)), min(altura, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    regiao = img[y1:y2, x1:x2]
    pequena = cv2.resize(regiao, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(pequena, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST),
                                         (0, 0), 3)


def main():
    p = argparse.ArgumentParser(description="Borra placas, rostos e telefones antes de publicar as fotos próprias")
    p.add_argument("--originais", default="fotos_nossas_originais")
    p.add_argument("--tarjas", default="fotos_nossas/tarjas.csv")
    p.add_argument("--saida", default="fotos_nossas")
    p.add_argument("--qualidade", type=int, default=92)
    a = p.parse_args()

    regioes = defaultdict(list)
    for r in pd.read_csv(a.tarjas).itertuples(index=False):
        regioes[r.foto].append((r.x1, r.y1, r.x2, r.y2))
    os.makedirs(a.saida, exist_ok=True)
    fotos = sorted(glob.glob(os.path.join(a.originais, "*.jpg")))
    for caminho in fotos:
        nome = os.path.basename(caminho)
        img = cv2.imread(caminho)
        for caixa in regioes.get(nome, []):
            borrar(img, *caixa)
        cv2.imwrite(os.path.join(a.saida, nome), img, [cv2.IMWRITE_JPEG_QUALITY, a.qualidade])
    print(f"{len(fotos)} fotos gravadas em {a.saida} ({sum(map(len, regioes.values()))} regiões borradas)")


if __name__ == "__main__":
    main()
