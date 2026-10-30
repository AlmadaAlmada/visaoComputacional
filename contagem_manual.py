import argparse
import csv
from collections import Counter

import cv2

from contagem.config import carregar_config
from contagem.contador import linha_da_config
from contagem.visual import desenhar_linha, texto

TECLAS = {"c": "carro", "m": "moto", "o": "onibus", "t": "caminhao", "v": "van", "b": "bicicleta", "x": "outro"}


def salvar(caminho, registros):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["tempo_s", "quadro", "sentido", "categoria"])
        escritor.writeheader()
        escritor.writerows(registros)


def main():
    p = argparse.ArgumentParser(description="Contagem manual de referência (ground truth) com o teclado")
    p.add_argument("--video", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--saida", default="referencia_manual.csv")
    p.add_argument("--velocidade", type=float, default=0.5)
    a = p.parse_args()

    cfg = carregar_config(a.config)
    sentidos = cfg["linha"]["sentidos"]
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Falha ao abrir {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    linha = linha_da_config(cfg, largura, altura)

    legenda = " ".join(f"{k}={v}" for k, v in TECLAS.items())
    registros = []
    velocidade = a.velocidade
    pausado = False
    quadro = None
    i = -1
    janela = "contagem manual"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    while True:
        if not pausado or quadro is None:
            ok, novo = cap.read()
            if not ok:
                break
            quadro = novo
            i += 1
        tela = quadro.copy()
        desenhar_linha(tela, linha, sentidos)
        contagem = Counter(r["sentido"] for r in registros)
        texto(tela, f"t={i / fps:7.2f}s  vel={velocidade:.2f}x  {'PAUSADO' if pausado else ''}", (10, 24), escala=0.6)
        texto(tela, f"minuscula -> {sentidos[0]} ({contagem[sentidos[0]]})   "
                    f"MAIUSCULA -> {sentidos[1]} ({contagem[sentidos[1]]})", (10, 50), escala=0.55)
        texto(tela, legenda, (10, 76), escala=0.5)
        texto(tela, "espaco=pausa  z=desfaz  +/-=velocidade  q=sai e salva", (10, 100), escala=0.5)
        if registros:
            u = registros[-1]
            texto(tela, f"ultimo: {u['categoria']} {u['sentido']} em {u['tempo_s']}s", (10, 126), (0, 255, 0), 0.55)
        cv2.imshow(janela, tela)
        espera = 30 if pausado else max(1, int(1000 / (fps * velocidade)))
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla == 255:
            continue
        caractere = chr(tecla)
        if caractere == "q" or tecla == 27:
            break
        if caractere == " ":
            pausado = not pausado
        elif caractere in "+=":
            velocidade = min(4.0, velocidade * 1.25)
        elif caractere == "-":
            velocidade = max(0.1, velocidade / 1.25)
        elif caractere == "z" and registros:
            registros.pop()
            salvar(a.saida, registros)
        elif caractere.lower() in TECLAS:
            sentido = sentidos[0] if caractere.islower() else sentidos[1]
            registros.append({"tempo_s": round(i / fps, 3), "quadro": i, "sentido": sentido,
                              "categoria": TECLAS[caractere.lower()]})
            salvar(a.saida, registros)
    cap.release()
    cv2.destroyAllWindows()
    salvar(a.saida, registros)
    print(f"{len(registros)} veículos registrados em {a.saida}")
    for (s, c), n in sorted(Counter((r["sentido"], r["categoria"]) for r in registros).items()):
        print(f"  {s} | {c}: {n}")


if __name__ == "__main__":
    main()
