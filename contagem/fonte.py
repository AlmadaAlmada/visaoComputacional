import glob
import os
import re

import cv2

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")


def _chave_natural(caminho):
    nome = os.path.basename(caminho)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nome)]


def numero_do_arquivo(caminho, padrao=None):
    numeros = re.findall(r"\d+", os.path.splitext(os.path.basename(caminho))[0])
    return int(numeros[-1]) if numeros else padrao


class FonteQuadros:
    def __init__(self, caminho, fps=None):
        self.caminho = caminho
        self.cap = None
        if os.path.isdir(caminho):
            arquivos = [f for f in glob.glob(os.path.join(caminho, "*")) if f.lower().endswith(EXTENSOES)]
            if not arquivos:
                raise FileNotFoundError(f"Nenhuma imagem encontrada em {caminho}")
            self.arquivos = sorted(arquivos, key=_chave_natural)
            primeiro = cv2.imread(self.arquivos[0])
            if primeiro is None:
                raise IOError(f"Falha ao ler {self.arquivos[0]}")
            self.altura, self.largura = primeiro.shape[:2]
            self.fps = float(fps or 25.0)
            self.total = len(self.arquivos)
        else:
            if not os.path.exists(caminho):
                raise FileNotFoundError(caminho)
            self.cap = cv2.VideoCapture(caminho)
            if not self.cap.isOpened():
                raise IOError(f"Falha ao abrir o vídeo {caminho}")
            self.fps = float(fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.largura = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.altura = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        if self.cap is None:
            for i, arquivo in enumerate(self.arquivos):
                quadro = cv2.imread(arquivo)
                if quadro is None:
                    continue
                yield i, numero_do_arquivo(arquivo, i), quadro
            return
        i = 0
        while True:
            ok, quadro = self.cap.read()
            if not ok:
                break
            yield i, i, quadro
            i += 1

    def quadro(self, indice=0):
        if self.cap is None:
            return cv2.imread(self.arquivos[min(indice, self.total - 1)])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, quadro = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return quadro if ok else None

    def fechar(self):
        if self.cap is not None:
            self.cap.release()
