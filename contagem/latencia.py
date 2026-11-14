from collections import defaultdict

import numpy as np
import pandas as pd

ETAPAS_ONLINE = ("deteccao_rastreio", "classificacao", "contagem_decisao")


def dispositivo():
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


class MedidorLatencia:
    def __init__(self, aquecimento=10):
        self.aquecimento = int(aquecimento)
        self.vistos = 0
        self.amostras = defaultdict(list)

    def registrar(self, tempos_ms):
        self.vistos += 1
        if self.vistos <= self.aquecimento:
            return
        for etapa, ms in tempos_ms.items():
            self.amostras[etapa].append(float(ms))
        self.amostras["processamento"].append(sum(tempos_ms.get(e, 0.0) for e in ETAPAS_ONLINE))
        self.amostras["total"].append(sum(tempos_ms.values()))

    def tabela(self):
        linhas = []
        for etapa, valores in self.amostras.items():
            v = np.asarray(valores)
            linhas.append({"etapa": etapa, "media_ms": round(float(v.mean()), 2),
                           "p50_ms": round(float(np.percentile(v, 50)), 2),
                           "p95_ms": round(float(np.percentile(v, 95)), 2),
                           "p99_ms": round(float(np.percentile(v, 99)), 2),
                           "max_ms": round(float(v.max()), 2)})
        return pd.DataFrame(linhas, columns=["etapa", "media_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"])

    def resumo(self, fps_video):
        if not self.amostras["processamento"]:
            return {}
        orcamento = 1000.0 / fps_video
        processamento = np.asarray(self.amostras["processamento"])
        total = np.asarray(self.amostras["total"])
        p95 = float(np.percentile(processamento, 95))
        return {"quadros_medidos": len(processamento), "aquecimento_descartado": self.aquecimento,
                "orcamento_ms": round(orcamento, 2),
                "processamento_media_ms": round(float(processamento.mean()), 2),
                "processamento_p95_ms": round(p95, 2),
                "total_media_ms": round(float(total.mean()), 2),
                "fps_sustentavel": round(1000.0 / float(processamento.mean()), 2),
                "tempo_real": p95 <= orcamento}
