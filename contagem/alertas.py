import math
import statistics
from collections import Counter
from datetime import timedelta

from .contador import categoria_dos_votos


def _categoria(evento):
    if evento.get("_votos"):
        return categoria_dos_votos(evento["_votos"])
    return evento.get("categoria", "outro")


class MonitorFluxo:
    def __init__(self, inicio=None, janela_s=60.0, min_janelas_base=3, limiar_z=2.5, fator_alto=1.5,
                 fator_baixo=0.4, janelas_sem_fluxo=2, linha_base_veic_h=None, alfa=0.3, parado_s=20.0,
                 parados_min=3, intervalo_congestionamento_s=60.0, frac_pesados=0.35,
                 min_veiculos_pesados=8, categorias_pesadas=("caminhao", "onibus")):
        self.inicio = inicio
        self.janela_s = float(janela_s)
        self.min_janelas_base = int(min_janelas_base)
        self.limiar_z = float(limiar_z)
        self.fator_alto = float(fator_alto)
        self.fator_baixo = float(fator_baixo)
        self.janelas_sem_fluxo = int(janelas_sem_fluxo)
        self.linha_base_veic_h = None if linha_base_veic_h is None else float(linha_base_veic_h)
        self.alfa = float(alfa)
        self.parado_s = float(parado_s)
        self.parados_min = int(parados_min)
        self.intervalo_congestionamento_s = float(intervalo_congestionamento_s)
        self.frac_pesados = float(frac_pesados)
        self.min_veiculos_pesados = int(min_veiculos_pesados)
        self.categorias_pesadas = set(categorias_pesadas)
        self.inicio_janela = 0.0
        self.eventos_janela = []
        self.janelas = []
        self.base = []
        self.media = None
        self.variancia = 0.0
        self.sem_fluxo = 0
        self.ultimo_congestionamento = None
        self.alertas = []

    def _carimbo(self, t):
        if self.inicio is None:
            return None
        return (self.inicio + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")

    def esperado(self, fracao=1.0):
        if self.linha_base_veic_h is not None:
            return self.linha_base_veic_h * self.janela_s / 3600.0 * fracao
        if len(self.base) < self.min_janelas_base:
            return None
        return self.media * fracao

    def registrar(self, evento):
        self.eventos_janela.append(evento)

    def atualizar(self, tempo, n_parados=0):
        novos = []
        while tempo >= self.inicio_janela + self.janela_s:
            novos.extend(self._fechar_janela(self.inicio_janela + self.janela_s, 1.0))
        if n_parados >= self.parados_min and (self.ultimo_congestionamento is None or
                                              tempo - self.ultimo_congestionamento >= self.intervalo_congestionamento_s):
            self.ultimo_congestionamento = tempo
            novos.append(self._alerta(
                "CONGESTIONAMENTO", "alta", tempo, tempo, n_parados, self.parados_min,
                f"{n_parados} veículos parados junto à linha há mais de {self.parado_s:.0f} s.",
                "Possível fila ou retenção: verificar o semáforo e o cruzamento a jusante, ampliar o verde da "
                "aproximação congestionada e acionar a operação de trânsito se a fila não dissipar no próximo ciclo."))
        return novos

    def encerrar(self, tempo):
        duracao = tempo - self.inicio_janela
        if duracao >= 0.5 * self.janela_s:
            return self._fechar_janela(tempo, duracao / self.janela_s)
        return []

    def _fechar_janela(self, fim, fracao):
        inicio = self.inicio_janela
        eventos = self.eventos_janela
        self.eventos_janela = []
        self.inicio_janela = fim
        n = len(eventos)
        esperado = self.esperado(fracao)
        self.janelas.append({"inicio_s": round(inicio, 2), "fim_s": round(fim, 2), "carimbo": self._carimbo(inicio),
                             "veiculos": n, "esperado": None if esperado is None else round(esperado, 2)})
        novos = []
        anomala = False
        if esperado is not None and esperado > 0:
            desvio = max(math.sqrt(self.variancia * fracao), math.sqrt(esperado))
            z = (n - esperado) / desvio
            if n >= self.fator_alto * esperado and z >= self.limiar_z:
                anomala = True
                dominante = Counter(e["sentido"] for e in eventos).most_common(1)[0][0]
                novos.append(self._alerta(
                    "FLUXO_ALTO", "media", inicio, fim, n, esperado,
                    f"{n} veículos na janela, {n / esperado - 1:.0%} acima do esperado ({esperado:.1f}); "
                    f"sentido dominante: {dominante}.",
                    f"Estender o tempo de verde no sentido {dominante}, coordenar os semáforos adjacentes "
                    "(onda verde) e posicionar agente de trânsito se o pico persistir."))
            elif n <= self.fator_baixo * esperado and z <= -self.limiar_z:
                anomala = True
                novos.append(self._alerta(
                    "FLUXO_BAIXO", "alta", inicio, fim, n, esperado,
                    f"{n} veículos na janela contra {esperado:.1f} esperados ({1 - n / esperado:.0%} abaixo).",
                    "Queda abrupta sugere bloqueio, acidente ou interdição a montante: conferir as imagens ao vivo, "
                    "despachar equipe de campo e, se confirmado, divulgar rota alternativa."))
        self.sem_fluxo = self.sem_fluxo + 1 if n == 0 else 0
        if self.sem_fluxo == self.janelas_sem_fluxo and (esperado is None or esperado >= 1):
            novos.append(self._alerta(
                "SEM_FLUXO", "alta", inicio, fim, 0, esperado,
                f"Nenhum veículo contado em {self.sem_fluxo} janelas seguidas "
                f"({self.sem_fluxo * self.janela_s / 60:.1f} min).",
                "Verificar se a câmera está obstruída ou desalinhada e se há interdição da via; "
                "não usar estes dados para planejamento até a confirmação."))
        if n >= self.min_veiculos_pesados:
            pesados = sum(1 for e in eventos if _categoria(e) in self.categorias_pesadas)
            if pesados / n >= self.frac_pesados:
                novos.append(self._alerta(
                    "PESADOS", "baixa", inicio, fim, pesados, round(self.frac_pesados * n, 2),
                    f"{pesados / n:.0%} dos veículos da janela são pesados ({pesados}/{n}).",
                    "Avaliar restrição de circulação de carga neste horário, fiscalização de peso e tempo de verde "
                    "compatível com a partida lenta de veículos pesados."))
        if not anomala and fracao >= 1.0:
            self._atualizar_base(n)
        return novos

    def _atualizar_base(self, n):
        self.base.append(n)
        if len(self.base) <= self.min_janelas_base:
            self.media = statistics.mean(self.base)
            self.variancia = statistics.pvariance(self.base)
            return
        d = n - self.media
        self.media += self.alfa * d
        self.variancia = (1 - self.alfa) * (self.variancia + self.alfa * d * d)

    def _alerta(self, tipo, severidade, inicio, fim, observado, esperado, mensagem, recomendacao):
        alerta = {"tipo": tipo, "severidade": severidade, "inicio_s": round(inicio, 2), "fim_s": round(fim, 2),
                  "carimbo": self._carimbo(inicio), "observado": observado,
                  "esperado": None if esperado is None else round(esperado, 2),
                  "mensagem": mensagem, "recomendacao": recomendacao}
        self.alertas.append(alerta)
        return alerta


def monitor_da_config(cfg, inicio=None, linha_base=None):
    parametros = dict(cfg.get("alerta", {}))
    if linha_base is not None:
        parametros["linha_base_veic_h"] = linha_base
    return MonitorFluxo(inicio=inicio, categorias_pesadas=tuple(cfg["categorias"]["pesadas"]), **parametros)
