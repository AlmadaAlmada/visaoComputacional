import cv2
import numpy as np

CORES = {
    "carro": (255, 170, 0),
    "moto": (0, 210, 255),
    "onibus": (0, 140, 255),
    "caminhao": (60, 60, 230),
    "van": (210, 100, 210),
    "bicicleta": (120, 220, 120),
    "outro": (180, 180, 180),
}
AMARELO = (0, 255, 255)
VERMELHO = (0, 0, 220)
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def fator(img):
    return max(1.0, img.shape[1] / 1280.0)


def texto(img, conteudo, origem, cor=(255, 255, 255), escala=0.5, espessura=1, fundo=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(conteudo, FONTE, escala, espessura)
    x, y = int(origem[0]), int(origem[1])
    cv2.rectangle(img, (x - 2, y - h - base), (x + w + 2, y + base), fundo, -1)
    cv2.putText(img, conteudo, (x, y), FONTE, escala, cor, espessura, cv2.LINE_AA)
    return h + base


def desenhar_linha(img, linha, sentidos):
    f = fator(img)
    p1 = (int(linha.p1[0]), int(linha.p1[1]))
    p2 = (int(linha.p2[0]), int(linha.p2[1]))
    cv2.line(img, p1, p2, AMARELO, max(2, int(2 * f)), cv2.LINE_AA)
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    nx, ny = linha.normal()
    for sinal, nome in ((1, sentidos[0]), (-1, sentidos[1])):
        ponta = (int(mx + sinal * 50 * f * nx), int(my + sinal * 50 * f * ny))
        cv2.arrowedLine(img, (int(mx), int(my)), ponta, AMARELO, max(2, int(2 * f)), cv2.LINE_AA, tipLength=0.3)
        texto(img, nome, (ponta[0] + 6 * f, ponta[1] + 6 * f), AMARELO, 0.5 * f, max(1, int(f)))


def desenhar(img, linha, deteccoes, contador, alertas, sentidos, carimbo):
    f = fator(img)
    fina = max(1, int(f))
    desenhar_linha(img, linha, sentidos)
    for d in deteccoes:
        tr = contador.trilhas.get(d.id)
        categoria = tr.categoria if tr else d.categoria
        cor = CORES.get(categoria, CORES["outro"])
        x1, y1, x2, y2 = (int(v) for v in d.caixa)
        contado = tr is not None and tr.contado
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3 * fina if contado else fina)
        texto(img, f"#{d.id} {categoria}", (x1, max(14 * f, y1 - 4 * f)), (0, 0, 0), 0.45 * f, fina, cor)
        if tr is not None and len(tr.pontos) > 1:
            pontos = np.array([(int(x), int(y)) for _, x, y in tr.pontos], np.int32)
            cv2.polylines(img, [pontos], False, cor, fina, cv2.LINE_AA)
    totais = contador.totais()
    y = 28 * f
    y += texto(img, carimbo, (10 * f, y), escala=0.6 * f, espessura=fina) + 10 * f
    for s in sentidos:
        partes = " ".join(f"{c}:{n}" for (ss, c), n in sorted(totais.items()) if ss == s)
        total = sum(n for (ss, _), n in totais.items() if ss == s)
        y += texto(img, f"{s}: {total}  {partes}", (10 * f, y), escala=0.6 * f, espessura=fina) + 10 * f
    for alerta in alertas:
        y += texto(img, f"ALERTA {alerta['tipo']}", (10 * f, y), (255, 255, 255), 0.7 * f, 2 * fina, VERMELHO) + 10 * f
    return img
