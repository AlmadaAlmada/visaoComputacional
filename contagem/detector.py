from ultralytics import YOLO

from .contador import Deteccao


class ClassificadorCategoria:
    def __init__(self, pesos, mapa, min_px=24, imgsz=128, conf_min=0.5, dispositivo=None, compativeis=None):
        self.modelo = YOLO(pesos)
        self.mapa = mapa
        self.compativeis = compativeis or {}
        self.min_px = min_px
        self.parametros = dict(imgsz=imgsz, verbose=False)
        if dispositivo is not None:
            self.parametros["device"] = dispositivo
        self.conf_min = conf_min

    def _melhor(self, resultado, categoria_detector):
        permitidas = set(self.compativeis.get(categoria_detector, [categoria_detector]))
        probabilidades = {}
        for indice, p in enumerate(resultado.probs.data.tolist()):
            categoria = self.mapa.get(resultado.names[indice])
            if categoria in permitidas:
                probabilidades[categoria] = probabilidades.get(categoria, 0.0) + p
        if not probabilidades:
            return None
        categoria, conf = max(probabilidades.items(), key=lambda kv: kv[1])
        return (categoria, conf) if conf >= self.conf_min else None

    def classificar(self, quadro, caixas, categorias):
        altura, largura = quadro.shape[:2]
        recortes, indices = [], []
        for k, (x1, y1, x2, y2) in enumerate(caixas):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(largura, int(x2)), min(altura, int(y2))
            if min(x2 - x1, y2 - y1) < self.min_px:
                continue
            recortes.append(quadro[y1:y2, x1:x2])
            indices.append(k)
        saida = [None] * len(caixas)
        if not recortes:
            return saida
        for k, r in zip(indices, self.modelo.predict(recortes, **self.parametros)):
            saida[k] = self._melhor(r, categorias[k])
        return saida


class DetectorRastreador:
    def __init__(self, cfg):
        m = cfg["modelo"]
        self.modelo = YOLO(m["pesos"])
        mapa = cfg["categorias"]["mapa"]
        self.categorias = {i: mapa[n] for i, n in self.modelo.names.items() if n in mapa}
        if not self.categorias:
            raise ValueError(f"Nenhuma classe do modelo {m['pesos']} está mapeada em categorias.mapa")
        comum = dict(iou=m["iou_nms"], agnostic_nms=m["agnostic_nms"], classes=sorted(self.categorias),
                     imgsz=m["imgsz"], verbose=False)
        if m.get("dispositivo") is not None:
            comum["device"] = m["dispositivo"]
        self.parametros_rastreio = dict(comum, persist=True, tracker=m["rastreador"], conf=m["conf"])
        self.parametros_deteccao = comum
        self.conf_operacao = m.get("conf_operacao", 0.3)
        self.classificador = None
        c = m.get("classificador") or {}
        if c.get("pesos"):
            self.classificador = ClassificadorCategoria(c["pesos"], mapa, c.get("min_px", 24), c.get("imgsz", 128),
                                                        c.get("conf_min", 0.5), m.get("dispositivo"),
                                                        c.get("compativeis"))

    def processar(self, quadro):
        r = self.modelo.track(quadro, **self.parametros_rastreio)[0]
        caixas = r.boxes
        if caixas is None or caixas.id is None:
            return []
        xyxy = caixas.xyxy.cpu().numpy().tolist()
        ids = caixas.id.int().cpu().tolist()
        classes = caixas.cls.int().cpu().tolist()
        confs = caixas.conf.cpu().tolist()
        deteccoes = [Deteccao(i, tuple(c), self.categorias[k], s)
                     for i, c, k, s in zip(ids, xyxy, classes, confs) if k in self.categorias]
        if self.classificador is None or not deteccoes:
            return deteccoes
        refinadas = self.classificador.classificar(quadro, [d.caixa for d in deteccoes],
                                                   [d.categoria for d in deteccoes])
        return [d if ref is None else d._replace(categoria=ref[0], conf=ref[1]) for d, ref in zip(deteccoes, refinadas)]

    def detectar(self, quadro, conf=0.05):
        r = self.modelo.predict(quadro, conf=conf, **self.parametros_deteccao)[0]
        caixas = r.boxes
        if caixas is None or len(caixas) == 0:
            return []
        xyxy = caixas.xyxy.cpu().numpy().tolist()
        classes = caixas.cls.int().cpu().tolist()
        confs = caixas.conf.cpu().tolist()
        return [(self.categorias[k], s, tuple(c)) for c, k, s in zip(xyxy, classes, confs) if k in self.categorias]
