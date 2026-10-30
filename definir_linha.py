import argparse

import cv2

from contagem.config import carregar_config, salvar_config
from contagem.contador import LinhaContagem
from contagem.fonte import FonteQuadros
from contagem.visual import desenhar_linha, texto


def main():
    p = argparse.ArgumentParser(description="Define a linha de contagem clicando em dois pontos no quadro")
    p.add_argument("--video", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--quadro", type=int, default=0)
    p.add_argument("--gravar", action="store_true", help="grava a linha no arquivo de config")
    a = p.parse_args()

    cfg = carregar_config(a.config)
    sentidos = cfg["linha"]["sentidos"]
    fonte = FonteQuadros(a.video)
    base = fonte.quadro(a.quadro)
    fonte.fechar()
    if base is None:
        raise SystemExit("Não foi possível ler o quadro")
    altura, largura = base.shape[:2]
    pontos = []

    def clique(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            if len(pontos) == 2:
                pontos.clear()
            pontos.append((x, y))

    janela = "definir linha"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, clique)
    confirmado = False
    while True:
        tela = base.copy()
        for pt in pontos:
            cv2.circle(tela, pt, 5, (0, 255, 255), -1)
        if len(pontos) == 2 and pontos[0] != pontos[1]:
            desenhar_linha(tela, LinhaContagem(pontos[0], pontos[1]), sentidos)
        texto(tela, "clique 2 pontos atravessando a via | ENTER confirma | R limpa | ESC sai", (10, 24), escala=0.55)
        cv2.imshow(janela, tela)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (13, 10) and len(pontos) == 2 and pontos[0] != pontos[1]:
            confirmado = True
            break
        if tecla in (ord("r"), ord("R")):
            pontos.clear()
        if tecla in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    if not confirmado:
        return

    p1 = [round(pontos[0][0] / largura, 4), round(pontos[0][1] / altura, 4)]
    p2 = [round(pontos[1][0] / largura, 4), round(pontos[1][1] / altura, 4)]
    print("linha:")
    print(f"  p1: {p1}")
    print(f"  p2: {p2}")
    print("  normalizada: true")
    if a.gravar:
        cfg["linha"]["p1"] = p1
        cfg["linha"]["p2"] = p2
        cfg["linha"]["normalizada"] = True
        salvar_config(cfg, a.config)
        print(f"Linha gravada em {a.config}")


if __name__ == "__main__":
    main()
