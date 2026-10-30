import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contagem.alertas import MonitorFluxo
from contagem.avaliacao import avaliar_contagem, avaliar_deteccoes
from contagem.contador import Contador, Deteccao, LinhaContagem

FPS = 10.0


def caixa(x, y_base, w=40, h=30):
    return (x - w / 2, y_base - h, x + w / 2, y_base)


def rodar(contador, trajetorias):
    quadros = max(len(t["ys"]) + t.get("inicio", 0) for t in trajetorias)
    for q in range(quadros):
        deteccoes = []
        for t in trajetorias:
            k = q - t.get("inicio", 0)
            if 0 <= k < len(t["ys"]) and t["ys"][k] is not None:
                xs = t.get("xs") or [t.get("x", 320)] * len(t["ys"])
                deteccoes.append(Deteccao(t["id"], caixa(xs[k], t["ys"][k], *t.get("tam", (40, 30))),
                                          t["cats"][k] if "cats" in t else t.get("cat", "carro"), t.get("conf", 0.9)))
        contador.atualizar(q, q, q / FPS, deteccoes)
    return contador


def novo_contador(**kw):
    linha = LinhaContagem((0, 200), (640, 200), margem=8)
    return Contador(linha, sentidos=("descendo", "subindo"), **kw)


class TestContador(unittest.TestCase):
    def test_cruzamento_nos_dois_sentidos(self):
        c = rodar(novo_contador(), [
            {"id": 1, "ys": list(range(100, 320, 10))},
            {"id": 2, "x": 500, "ys": list(range(320, 100, -10))},
        ])
        self.assertEqual(sorted(e["sentido"] for e in c.eventos), ["descendo", "subindo"])

    def test_veiculo_parado_oscilando_sobre_a_linha_conta_uma_vez(self):
        oscilacao = [188, 214, 186, 215, 187, 213, 189, 216] * 10
        c = rodar(novo_contador(), [{"id": 1, "ys": list(range(120, 188, 8)) + oscilacao + list(range(220, 330, 8))}])
        self.assertEqual(len(c.eventos), 1)

    def test_jitter_dentro_da_margem_nao_conta(self):
        c = rodar(novo_contador(), [{"id": 1, "ys": [150, 160] + [195, 205, 196, 204] * 20}])
        self.assertEqual(len(c.eventos), 0)

    def test_moto_ao_lado_do_carro_conta_os_dois(self):
        ys = list(range(100, 320, 10))
        c = rodar(novo_contador(), [
            {"id": 1, "x": 300, "ys": ys, "tam": (80, 60), "cat": "carro"},
            {"id": 2, "x": 360, "ys": ys, "tam": (20, 30), "cat": "moto"},
        ])
        self.assertEqual(sorted(c.totais().items()), [(("descendo", "carro"), 1), (("descendo", "moto"), 1)])

    def test_troca_de_id_apos_cruzar_nao_conta_de_novo(self):
        c = rodar(novo_contador(), [
            {"id": 1, "ys": list(range(150, 222, 6))},
            {"id": 7, "inicio": 12, "ys": [190, 196, 202, 210, 218, 226, 234, 242]},
        ])
        self.assertEqual(len(c.eventos), 1)
        self.assertEqual(len(c.duplicatas), 1)

    def test_cruzamento_fora_do_segmento_e_ignorado(self):
        linha = LinhaContagem((0, 200), (300, 200), margem=8)
        c = rodar(Contador(linha, sentidos=("descendo", "subindo")), [{"id": 1, "x": 500, "ys": list(range(100, 320, 10))}])
        self.assertEqual(len(c.eventos), 0)

    def test_categoria_por_votacao_na_trilha(self):
        ys = list(range(100, 400, 10))
        cats = ["onibus", "onibus"] + ["caminhao"] * (len(ys) - 2)
        c = rodar(novo_contador(), [{"id": 1, "ys": ys, "cats": cats}])
        self.assertEqual(c.tabela_eventos()[0]["categoria"], "caminhao")

    def test_trilha_curta_espera_min_observacoes(self):
        c = rodar(novo_contador(min_observacoes=3), [{"id": 1, "ys": [195, 215, 225, 235]}])
        self.assertEqual(len(c.eventos), 0)
        c = rodar(novo_contador(min_observacoes=3), [{"id": 1, "ys": [180, 215, 225, 235]}])
        self.assertEqual(len(c.eventos), 1)

    def test_veiculos_parados_na_fila(self):
        c = rodar(novo_contador(), [{"id": i, "x": 100 + 60 * i, "ys": [180] * 300} for i in range(4)])
        self.assertEqual(len(c.parados(299 / FPS, 20)), 4)


class TestMonitor(unittest.TestCase):
    def _simular(self, contagens, **kw):
        m = MonitorFluxo(janela_s=60, min_janelas_base=3, **kw)
        t = 0.0
        for n in contagens:
            for k in range(n):
                m.atualizar(t + k * 60.0 / max(n, 1))
                m.registrar({"sentido": "descendo", "categoria": "carro"})
            t += 60.0
        m.atualizar(t)
        return m

    def test_fluxo_alto(self):
        m = self._simular([10, 11, 9, 10, 30])
        self.assertEqual([a["tipo"] for a in m.alertas], ["FLUXO_ALTO"])

    def test_fluxo_baixo_e_sem_fluxo(self):
        m = self._simular([12, 10, 11, 1, 0, 0])
        tipos = [a["tipo"] for a in m.alertas]
        self.assertIn("FLUXO_BAIXO", tipos)
        self.assertIn("SEM_FLUXO", tipos)

    def test_fluxo_normal_sem_alerta(self):
        self.assertEqual(self._simular([10, 12, 9, 11, 10, 13]).alertas, [])

    def test_linha_base_externa(self):
        m = self._simular([30], linha_base_veic_h=600)
        self.assertEqual([a["tipo"] for a in m.alertas], ["FLUXO_ALTO"])


class TestAvaliacao(unittest.TestCase):
    def test_pareamento_e_causas(self):
        ref = pd.DataFrame({"tempo_s": [1.0, 5.0, 5.2, 9.0], "sentido": ["a", "a", "a", "b"],
                            "categoria": ["carro", "carro", "moto", "onibus"]})
        sis = pd.DataFrame({"tempo_s": [1.5, 5.3, 9.4, 9.9], "sentido": ["a", "a", "b", "b"],
                            "categoria": ["carro", "carro", "caminhao", "caminhao"], "id": [1, 2, 3, 4]})
        r = avaliar_contagem(sis, ref, "tempo_s", 2.0)
        self.assertEqual((r["pares"], r["extras"], r["perdidos"]), (3, 1, 1))
        causas = set(r["erros"]["causa_provavel"])
        self.assertTrue(any(c.startswith("contagem_dupla") for c in causas))
        self.assertTrue(any(c.startswith("veiculos_simultaneos") for c in causas))
        total = r["totais"].iloc[-1]
        self.assertEqual((total["referencia"], total["sistema"], total["erro_pct"]), (4, 4, 0.0))
        self.assertAlmostEqual(r["acuracia_categoria"], 2 / 3, places=3)

    def test_referencia_agregada(self):
        ref = pd.DataFrame({"sentido": ["a", "a"], "categoria": ["carro", "moto"], "quantidade": [10, 5]})
        sis = pd.DataFrame({"sentido": ["a"] * 12, "categoria": ["carro"] * 9 + ["moto"] * 3})
        r = avaliar_contagem(sis, ref)
        self.assertEqual(r["totais"].iloc[-1]["erro_pct"], -20.0)

    def test_ap_detector(self):
        gts = [(0, "carro", (0, 0, 10, 10)), (0, "moto", (20, 20, 30, 30))]
        preds = [(0, "carro", 0.9, (0, 0, 10, 10)), (0, "carro", 0.8, (50, 50, 60, 60))]
        tabela, map50 = avaliar_deteccoes(preds, gts)
        carro = tabela[tabela["categoria"] == "carro"].iloc[0]
        self.assertEqual((carro["tp"], carro["ap50"]), (1, 1.0))
        self.assertEqual(map50, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
