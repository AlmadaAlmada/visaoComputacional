import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

from .contador import Deteccao, contador_da_config, linha_da_config
from .fonte import numero_do_arquivo

TAGS_ID = ("ID", "id", "trackid", "track_id", "trackID", "object_id")
TAGS_CLASSE = ("class", "name", "label", "type")
TAGS_QUADRO = ("frame", "number", "num", "frame_number")
ELEMENTOS_QUADRO = ("frame", "image")


def _valor(el, tags):
    for t in tags:
        filho = el.find(t)
        if filho is not None and filho.text and filho.text.strip():
            return filho.text.strip()
        if t in el.attrib:
            return el.attrib[t]
    return None


def _caixa(el):
    alvo = el.find("bndbox")
    alvo = alvo if alvo is not None else el
    for nomes in (("xmin", "ymin", "xmax", "ymax"), ("xtl", "ytl", "xbr", "ybr")):
        valores = []
        for n in nomes:
            filho = alvo.find(n)
            v = filho.text if filho is not None else alvo.attrib.get(n)
            if v is None:
                break
            valores.append(float(v))
        if len(valores) == 4:
            return tuple(valores)
    return None


def _inteiro(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _percorrer(el, quadro, objetos):
    if el.tag in ELEMENTOS_QUADRO:
        q = _inteiro(_valor(el, TAGS_QUADRO + ("id",)))
        quadro = q if q is not None else quadro
    if el.tag != "bndbox":
        caixa = _caixa(el)
        if caixa is not None and el.tag not in ELEMENTOS_QUADRO:
            q = _inteiro(_valor(el, TAGS_QUADRO))
            objetos.append({"quadro": q if q is not None else quadro, "id": _valor(el, TAGS_ID),
                            "classe": (_valor(el, TAGS_CLASSE) or "").lower(), "caixa": caixa,
                            "fora": el.attrib.get("outside") == "1"})
            return
    for filho in el:
        _percorrer(filho, quadro, objetos)


def _ler_cvat(raiz):
    objetos = []
    for trilha in raiz.iter("track"):
        for caixa in trilha.iter("box"):
            if caixa.attrib.get("outside") == "1":
                continue
            objetos.append({"quadro": int(caixa.attrib["frame"]), "id": trilha.attrib.get("id"),
                            "classe": trilha.attrib.get("label", "").lower(),
                            "caixa": tuple(float(caixa.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr"))})
    return objetos


def ler_anotacoes(caminho):
    arquivos = sorted(glob.glob(os.path.join(caminho, "*.xml"))) if os.path.isdir(caminho) else [caminho]
    if not arquivos:
        raise FileNotFoundError(f"Nenhum XML em {caminho}")
    objetos = []
    for arquivo in arquivos:
        raiz = ET.parse(arquivo).getroot()
        if raiz.find(".//track/box") is not None:
            objetos.extend(_ler_cvat(raiz))
            continue
        encontrados = []
        _percorrer(raiz, numero_do_arquivo(arquivo), encontrados)
        objetos.extend(o for o in encontrados if not o.pop("fora"))
    validos = [o for o in objetos if o["quadro"] is not None]
    for o in validos:
        o["id"] = None if o["id"] is None else str(o["id"])
    return validos


def resumo_anotacoes(objetos):
    classes = defaultdict(int)
    for o in objetos:
        classes[o["classe"]] += 1
    return {"objetos": len(objetos), "quadros": len({o["quadro"] for o in objetos}),
            "trilhas": len({o["id"] for o in objetos if o["id"] is not None}),
            "sem_id": sum(1 for o in objetos if o["id"] is None), "classes": dict(classes)}


def eventos_referencia(objetos, cfg, largura, altura, fps):
    linha = linha_da_config(cfg, largura, altura)
    contador = contador_da_config(cfg, linha)
    contador.min_observacoes = 1
    contador.dedup_iou = 2.0
    mapa = cfg["categorias"]["mapa"]
    por_quadro = defaultdict(list)
    for o in objetos:
        if o["id"] is not None:
            por_quadro[o["quadro"]].append(o)
    if not por_quadro:
        return []
    quadros = sorted(por_quadro)
    primeiro = quadros[0]
    for q in range(primeiro, quadros[-1] + 1):
        deteccoes = [Deteccao(o["id"], o["caixa"], mapa.get(o["classe"], "outro"), 1.0) for o in por_quadro.get(q, [])]
        contador.atualizar(q - primeiro, q, (q - primeiro) / fps, deteccoes)
    return contador.tabela_eventos()
