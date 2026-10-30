import math
from collections import Counter, defaultdict, deque, namedtuple

Deteccao = namedtuple("Deteccao", ["id", "caixa", "categoria", "conf"])


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def categoria_dos_votos(votos, padrao="outro"):
    if not votos:
        return padrao
    return max(votos.items(), key=lambda kv: kv[1])[0]


class LinhaContagem:
    def __init__(self, p1, p2, margem=8.0, tolerancia_extremos=0.0):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        self.comprimento = math.hypot(dx, dy)
        if self.comprimento == 0:
            raise ValueError("Os dois pontos da linha de contagem são iguais")
        self.ux = dx / self.comprimento
        self.uy = dy / self.comprimento
        self.margem = float(margem)
        self.tolerancia = float(tolerancia_extremos)

    def distancia(self, p):
        return self.ux * (p[1] - self.p1[1]) - self.uy * (p[0] - self.p1[0])

    def projecao(self, p):
        return ((p[0] - self.p1[0]) * self.ux + (p[1] - self.p1[1]) * self.uy) / self.comprimento

    def lado(self, p):
        d = self.distancia(p)
        if d > self.margem:
            return 1
        if d < -self.margem:
            return -1
        return 0

    def cruza_segmento(self, a, b):
        da = self.distancia(a)
        db = self.distancia(b)
        if da == db:
            return False
        t = da / (da - db)
        ponto = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        s = self.projecao(ponto)
        return -self.tolerancia <= s <= 1.0 + self.tolerancia

    def normal(self):
        return (-self.uy, self.ux)


class Trilha:
    def __init__(self, id_trilha, tempo, maximo_pontos):
        self.id = id_trilha
        self.lado = 0
        self.ponto_lado = None
        self.votos = defaultdict(float)
        self.pontos = deque(maxlen=maximo_pontos)
        self.inicio_t = tempo
        self.ultimo_t = tempo
        self.observacoes = 0
        self.contado = False
        self.evento = None
        self.caixa = None
        self.parado_desde = None

    @property
    def categoria(self):
        return categoria_dos_votos(self.votos)

    @property
    def ponto(self):
        if not self.pontos:
            return None
        return (self.pontos[-1][1], self.pontos[-1][2])


class Contador:
    def __init__(self, linha, sentidos=("A->B", "B->A"), min_observacoes=2, ponto_referencia="base",
                 dedup_iou=0.3, dedup_janela_s=1.0, retencao_s=10.0, janela_velocidade_s=1.0,
                 limiar_parado=0.15, distancia_fila_px=200.0, maximo_pontos=90):
        self.linha = linha
        self.sentidos = tuple(sentidos)
        self.min_observacoes = int(min_observacoes)
        self.ponto_referencia = ponto_referencia
        self.dedup_iou = float(dedup_iou)
        self.dedup_janela_s = float(dedup_janela_s)
        self.retencao_s = float(retencao_s)
        self.janela_velocidade_s = float(janela_velocidade_s)
        self.limiar_parado = float(limiar_parado)
        self.distancia_fila_px = float(distancia_fila_px)
        self.maximo_pontos = int(maximo_pontos)
        self.trilhas = {}
        self.eventos = []
        self.duplicatas = []

    def ponto(self, caixa):
        x1, y1, x2, y2 = caixa
        if self.ponto_referencia == "centro":
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return ((x1 + x2) / 2.0, float(y2))

    def atualizar(self, quadro, quadro_id, tempo, deteccoes, carimbo=None):
        ativos = {d.id for d in deteccoes}
        novos = []
        for d in deteccoes:
            tr = self.trilhas.get(d.id)
            if tr is None:
                tr = Trilha(d.id, tempo, self.maximo_pontos)
                self.trilhas[d.id] = tr
            tr.observacoes += 1
            tr.votos[d.categoria] += float(d.conf)
            tr.ultimo_t = tempo
            tr.caixa = tuple(float(v) for v in d.caixa)
            p = self.ponto(tr.caixa)
            tr.pontos.append((tempo, p[0], p[1]))
            self._atualizar_parada(tr, tempo)
            evento = self._verificar_cruzamento(tr, p, quadro, quadro_id, tempo, carimbo, ativos)
            if evento is not None:
                novos.append(evento)
        self._podar(tempo)
        return novos

    def _verificar_cruzamento(self, tr, p, quadro, quadro_id, tempo, carimbo, ativos):
        lado = self.linha.lado(p)
        if lado == 0:
            return None
        if tr.lado == 0 or lado == tr.lado or tr.contado:
            tr.lado = lado
            tr.ponto_lado = p
            return None
        if tr.observacoes < self.min_observacoes:
            return None
        anterior = tr.ponto_lado
        tr.lado = lado
        tr.ponto_lado = p
        if not self.linha.cruza_segmento(anterior, p):
            return None
        sentido = self.sentidos[0] if lado > 0 else self.sentidos[1]
        tr.contado = True
        original = self._duplicata(tr, tempo, sentido, ativos)
        if original is not None:
            for categoria, peso in tr.votos.items():
                original.votos[categoria] += peso
            tr.votos = original.votos
            self.duplicatas.append({"id": tr.id, "id_original": original.id, "quadro": quadro,
                                    "quadro_id": quadro_id, "tempo_s": round(tempo, 3), "sentido": sentido})
            return None
        evento = {"id": tr.id, "quadro": quadro, "quadro_id": quadro_id, "tempo_s": round(tempo, 3),
                  "carimbo": carimbo, "sentido": sentido, "categoria": tr.categoria,
                  "x": round(p[0], 1), "y": round(p[1], 1), "_votos": tr.votos}
        tr.evento = evento
        self.eventos.append(evento)
        return evento

    def _duplicata(self, tr, tempo, sentido, ativos):
        for outra in self.trilhas.values():
            if outra is tr or outra.id in ativos or outra.evento is None:
                continue
            if outra.evento["sentido"] != sentido or tempo - outra.ultimo_t > self.dedup_janela_s:
                continue
            if iou(outra.caixa, tr.caixa) >= self.dedup_iou:
                return outra
        return None

    def _atualizar_parada(self, tr, tempo):
        limite = tempo - self.janela_velocidade_s
        antigo = next((pt for pt in tr.pontos if pt[0] >= limite), None)
        if antigo is None or tempo - antigo[0] < 0.5 * self.janela_velocidade_s:
            return
        altura = max(1.0, tr.caixa[3] - tr.caixa[1])
        ultimo = tr.pontos[-1]
        velocidade = math.hypot(ultimo[1] - antigo[1], ultimo[2] - antigo[2]) / (tempo - antigo[0]) / altura
        if velocidade < self.limiar_parado:
            if tr.parado_desde is None:
                tr.parado_desde = antigo[0]
        else:
            tr.parado_desde = None

    def parados(self, tempo, minimo_s):
        return [tr for tr in self.trilhas.values()
                if tr.ultimo_t == tempo and tr.parado_desde is not None
                and tempo - tr.parado_desde >= minimo_s
                and abs(self.linha.distancia(tr.ponto)) <= self.distancia_fila_px]

    def _podar(self, tempo):
        vencidas = [i for i, tr in self.trilhas.items() if tempo - tr.ultimo_t > self.retencao_s]
        for i in vencidas:
            del self.trilhas[i]

    def totais(self):
        return Counter((ev["sentido"], categoria_dos_votos(ev["_votos"])) for ev in self.eventos)

    def tabela_eventos(self):
        linhas = []
        for ev in self.eventos:
            linha = {k: v for k, v in ev.items() if not k.startswith("_")}
            linha["categoria_inicial"] = ev["categoria"]
            linha["categoria"] = categoria_dos_votos(ev["_votos"])
            linhas.append(linha)
        return linhas


def linha_da_config(cfg, largura, altura):
    c = cfg["linha"]

    def converter(p):
        if c.get("normalizada", True):
            return (float(p[0]) * largura, float(p[1]) * altura)
        return (float(p[0]), float(p[1]))

    return LinhaContagem(converter(c["p1"]), converter(c["p2"]), c.get("margem_px", 8.0),
                         c.get("tolerancia_extremos", 0.0))


def contador_da_config(cfg, linha):
    return Contador(linha, sentidos=tuple(cfg["linha"]["sentidos"]), **cfg.get("contador", {}))
