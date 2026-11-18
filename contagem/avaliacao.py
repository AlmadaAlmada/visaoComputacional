from collections import defaultdict

import numpy as np
import pandas as pd

from .contador import iou

PENALIDADE_CATEGORIA = 0.25


def casar_eventos(sis, ref, coluna, tolerancia):
    candidatos = []
    for sentido in set(sis["sentido"]) | set(ref["sentido"]):
        a = sis[sis["sentido"] == sentido]
        b = ref[ref["sentido"] == sentido]
        if a.empty or b.empty:
            continue
        ta = a[coluna].to_numpy(dtype=float)[:, None]
        tb = b[coluna].to_numpy(dtype=float)[None, :]
        diferenca = np.abs(ta - tb)
        mesma = a["categoria"].to_numpy()[:, None] == b["categoria"].to_numpy()[None, :]
        custo = diferenca + np.where(mesma, 0.0, PENALIDADE_CATEGORIA * tolerancia)
        for i, j in zip(*np.nonzero(diferenca <= tolerancia)):
            candidatos.append((custo[i, j], a.index[i], b.index[j]))
    candidatos.sort()
    usados_s, usados_r, pares = set(), set(), []
    for _, i, j in candidatos:
        if i in usados_s or j in usados_r:
            continue
        usados_s.add(i)
        usados_r.add(j)
        pares.append((i, j))
    return pares


def _tabela_totais(sis_agregado, ref_agregado):
    tabela = pd.merge(ref_agregado, sis_agregado, on=["sentido", "categoria"], how="outer",
                      suffixes=("_referencia", "_sistema")).fillna(0)
    tabela = tabela.rename(columns={"quantidade_referencia": "referencia", "quantidade_sistema": "sistema"})
    tabela[["referencia", "sistema"]] = tabela[["referencia", "sistema"]].astype(int)
    for s in sorted(tabela["sentido"].unique()):
        parcial = tabela[tabela["sentido"] == s]
        tabela = pd.concat([tabela, pd.DataFrame([{"sentido": s, "categoria": "TOTAL",
                                                   "referencia": parcial["referencia"].sum(),
                                                   "sistema": parcial["sistema"].sum()}])], ignore_index=True)
    geral = tabela[tabela["categoria"] != "TOTAL"]
    tabela = pd.concat([tabela, pd.DataFrame([{"sentido": "TODOS", "categoria": "TOTAL",
                                               "referencia": geral["referencia"].sum(),
                                               "sistema": geral["sistema"].sum()}])], ignore_index=True)
    tabela["diferenca"] = tabela["sistema"] - tabela["referencia"]
    tabela["erro_pct"] = np.where(tabela["referencia"] > 0,
                                  (100.0 * tabela["diferenca"] / tabela["referencia"].replace(0, np.nan)).round(1),
                                  np.nan)
    return tabela


def _agregar(df):
    if "quantidade" in df.columns:
        return df.groupby(["sentido", "categoria"], as_index=False)["quantidade"].sum()
    return df.groupby(["sentido", "categoria"]).size().reset_index(name="quantidade")


def _causa_extra(e, sis, ref_livres, coluna, tolerancia):
    oposto = ref_livres[(ref_livres["sentido"] != e["sentido"]) & ((ref_livres[coluna] - e[coluna]).abs() <= tolerancia)]
    if not oposto.empty:
        return "sentido_invertido"
    vizinhos = sis[(sis["sentido"] == e["sentido"]) & (sis.index != e.name) &
                   ((sis[coluna] - e[coluna]).abs() <= tolerancia)]
    if not vizinhos.empty:
        return "contagem_dupla (mesmo veículo contado de novo: troca de ID, veículo parado ou oscilando na linha)"
    return "falso_positivo (detecção espúria, veículo fora da via ou pedestre/objeto confundido)"


def _causa_perdida(r, ref, sis_livres, coluna, tolerancia, simultaneo):
    oposto = sis_livres[(sis_livres["sentido"] != r["sentido"]) & ((sis_livres[coluna] - r[coluna]).abs() <= tolerancia)]
    if not oposto.empty:
        return "sentido_invertido"
    vizinhos = ref[(ref.index != r.name) & ((ref[coluna] - r[coluna]).abs() <= simultaneo)]
    if not vizinhos.empty:
        return "veiculos_simultaneos (supressão pelo NMS lado a lado ou oclusão por veículo maior)"
    if r["categoria"] in ("moto", "bicicleta"):
        return "objeto_pequeno (moto/bicicleta não detectada ou trilha curta demais)"
    return "nao_detectado_ou_trilha_perdida (oclusão, iluminação, distância)"


def avaliar_contagem(sis, ref, coluna=None, tolerancia=2.0, simultaneo=None):
    resultado = {"totais": _tabela_totais(_agregar(sis), _agregar(ref))}
    if coluna is None or coluna not in ref.columns or coluna not in sis.columns:
        return resultado
    simultaneo = tolerancia / 2.0 if simultaneo is None else simultaneo
    sis = sis.reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    pares = casar_eventos(sis, ref, coluna, tolerancia)
    casados_s = {i for i, _ in pares}
    casados_r = {j for _, j in pares}
    extras = sis[~sis.index.isin(casados_s)]
    perdidos = ref[~ref.index.isin(casados_r)]

    erros = []
    for _, e in extras.iterrows():
        erros.append({"tipo": "contagem_extra", coluna: e[coluna], "sentido": e["sentido"],
                      "categoria": e["categoria"], "id_trilha": e.get("id"),
                      "causa_provavel": _causa_extra(e, sis, perdidos, coluna, tolerancia)})
    for _, r in perdidos.iterrows():
        erros.append({"tipo": "contagem_perdida", coluna: r[coluna], "sentido": r["sentido"],
                      "categoria": r["categoria"], "id_trilha": None,
                      "causa_provavel": _causa_perdida(r, ref, extras, coluna, tolerancia, simultaneo)})
    for i, j in pares:
        if sis.loc[i, "categoria"] != ref.loc[j, "categoria"]:
            erros.append({"tipo": "categoria_errada", coluna: sis.loc[i, coluna], "sentido": sis.loc[i, "sentido"],
                          "categoria": f"{ref.loc[j, 'categoria']} -> {sis.loc[i, 'categoria']}",
                          "id_trilha": sis.loc[i].get("id"), "causa_provavel": "classificação"})
    erros = pd.DataFrame(erros, columns=["tipo", coluna, "sentido", "categoria", "id_trilha", "causa_provavel"])
    erros = erros.sort_values(coluna).reset_index(drop=True)

    cat_ref = [ref.loc[j, "categoria"] for _, j in pares]
    cat_sis = [sis.loc[i, "categoria"] for i, _ in pares]
    confusao = pd.crosstab(pd.Series(cat_ref, name="referencia"), pd.Series(cat_sis, name="sistema"))
    acertos = pd.Series([a == b for a, b in zip(cat_ref, cat_sis)], dtype=bool)
    por_categoria = pd.DataFrame({"referencia": cat_ref, "acerto": acertos}).groupby("referencia")["acerto"] \
        .agg(["count", "mean"]).rename(columns={"count": "pareados", "mean": "acuracia"}).reset_index()
    por_categoria["acuracia"] = por_categoria["acuracia"].round(3)

    tp = len(pares)
    resultado.update({
        "pares": tp,
        "extras": len(extras),
        "perdidos": len(perdidos),
        "precisao_contagem": round(tp / len(sis), 4) if len(sis) else 0.0,
        "revocacao_contagem": round(tp / len(ref), 4) if len(ref) else 0.0,
        "acuracia_categoria": round(float(acertos.mean()), 4) if tp else 0.0,
        "acuracia_por_categoria": por_categoria,
        "confusao": confusao,
        "erros": erros,
        "causas": erros.groupby(["tipo", "causa_provavel"]).size().reset_index(name="ocorrencias"),
    })
    return resultado


def avaliar_classificacao(verdadeiro, previsto):
    verdadeiro = pd.Series(list(verdadeiro), name="referencia")
    previsto = pd.Series(list(previsto), name="sistema")
    linhas = []
    for c in sorted(verdadeiro.unique()):
        tp = int(((verdadeiro == c) & (previsto == c)).sum())
        n_prev = int((previsto == c).sum())
        n_ref = int((verdadeiro == c).sum())
        precisao = tp / n_prev if n_prev else 0.0
        revocacao = tp / n_ref
        f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
        linhas.append({"classe": c, "amostras": n_ref, "precisao": round(precisao, 3),
                       "revocacao": round(revocacao, 3), "f1": round(f1, 3)})
    por_classe = pd.DataFrame(linhas)
    return {"acuracia": round(float((verdadeiro == previsto).mean()), 4),
            "f1_macro": round(float(por_classe["f1"].mean()), 4),
            "por_classe": por_classe,
            "confusao": pd.crosstab(verdadeiro, previsto)}


def _ap_voc(revocacao, precisao):
    r = np.concatenate(([0.0], revocacao, [1.0]))
    p = np.concatenate(([0.0], precisao, [0.0]))
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def _avaliar_classe(predicoes, gts, limiar_iou, conf_operacao):
    gts_por_quadro = defaultdict(list)
    for q, caixa in gts:
        gts_por_quadro[q].append(caixa)
    usados = {q: [False] * len(c) for q, c in gts_por_quadro.items()}
    predicoes = sorted(predicoes, key=lambda p: -p[1])
    tp = np.zeros(len(predicoes))
    for k, (q, conf, caixa) in enumerate(predicoes):
        melhor, idx = 0.0, -1
        for j, g in enumerate(gts_por_quadro.get(q, [])):
            v = iou(caixa, g)
            if v > melhor:
                melhor, idx = v, j
        if melhor >= limiar_iou and not usados[q][idx]:
            usados[q][idx] = True
            tp[k] = 1
    n_gt = len(gts)
    acumulado_tp = np.cumsum(tp)
    acumulado_fp = np.cumsum(1 - tp)
    revocacao = acumulado_tp / max(n_gt, 1)
    precisao = acumulado_tp / np.maximum(acumulado_tp + acumulado_fp, 1e-9)
    ap = _ap_voc(revocacao, precisao) if len(predicoes) and n_gt else 0.0
    operacao = np.array([p[1] >= conf_operacao for p in predicoes], dtype=bool)
    tp_op = int(tp[operacao].sum()) if len(predicoes) else 0
    n_op = int(operacao.sum()) if len(predicoes) else 0
    return {"gt": n_gt, "predicoes": n_op, "tp": tp_op,
            "precisao": round(tp_op / n_op, 4) if n_op else 0.0,
            "revocacao": round(tp_op / n_gt, 4) if n_gt else 0.0,
            "ap50": round(ap, 4)}


def avaliar_deteccoes(predicoes, gts, limiar_iou=0.5, conf_operacao=0.3):
    linhas = []
    categorias = sorted({p[1] for p in predicoes} | {g[1] for g in gts})
    for c in categorias:
        p = [(q, s, caixa) for q, cat, s, caixa in predicoes if cat == c]
        g = [(q, caixa) for q, cat, caixa in gts if cat == c]
        linhas.append(dict(categoria=c, **_avaliar_classe(p, g, limiar_iou, conf_operacao)))
    p = [(q, s, caixa) for q, _, s, caixa in predicoes]
    g = [(q, caixa) for q, _, caixa in gts]
    linhas.append(dict(categoria="TODOS (sem classe)", **_avaliar_classe(p, g, limiar_iou, conf_operacao)))
    tabela = pd.DataFrame(linhas)
    com_gt = tabela[(tabela["categoria"] != "TODOS (sem classe)") & (tabela["gt"] > 0)]
    return tabela, round(float(com_gt["ap50"].mean()), 4) if not com_gt.empty else 0.0
