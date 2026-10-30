import json
import os

import pandas as pd

COLUNAS_EVENTOS = ["id", "quadro", "quadro_id", "tempo_s", "carimbo", "sentido", "categoria",
                   "categoria_inicial", "x", "y"]


def eventos_para_df(linhas):
    df = pd.DataFrame(linhas, columns=COLUNAS_EVENTOS)
    df["carimbo"] = pd.to_datetime(df["carimbo"])
    return df


def _categorias(df, ucp):
    ordem = list(ucp.keys())
    return ordem + sorted(c for c in df["categoria"].unique() if c not in ordem)


def tabela_faixas(df, inicio, duracao_s, faixa_min, sentidos, ucp):
    inicio = pd.Timestamp(inicio)
    fim = inicio + pd.Timedelta(seconds=duracao_s)
    passo = pd.Timedelta(minutes=faixa_min)
    categorias = _categorias(df, ucp)
    linhas = []
    atual = inicio.floor(f"{faixa_min}min")
    while atual < fim:
        proxima = atual + passo
        observado = (min(proxima, fim) - max(atual, inicio)).total_seconds() / 60.0
        selecao = df[(df["carimbo"] >= atual) & (df["carimbo"] < proxima)]
        for s in sentidos:
            parcial = selecao[selecao["sentido"] == s]
            linha = {"faixa_inicio": atual.strftime("%Y-%m-%d %H:%M"), "faixa_fim": proxima.strftime("%H:%M"),
                     "observado_min": round(observado, 2), "sentido": s}
            for c in categorias:
                linha[c] = int((parcial["categoria"] == c).sum())
            linha["total"] = int(len(parcial))
            linha["ucp"] = round(sum(ucp.get(c, 1.0) * linha[c] for c in categorias), 2)
            fator = 60.0 / observado if observado > 0 else 0.0
            linha["veic_h"] = round(linha["total"] * fator, 1)
            linha["ucp_h"] = round(linha["ucp"] * fator, 1)
            linhas.append(linha)
        atual = proxima
    return pd.DataFrame(linhas)


def recomendacoes(df, faixas, alertas, sentidos, pesadas, duracao_s):
    if df.empty or duracao_s <= 0:
        return ["Nenhum veículo foi contado: verificar se a linha virtual atravessa a via inteira e se a câmera "
                "enquadra o fluxo antes de usar o ponto para planejamento."]
    horas = duracao_s / 3600.0
    recs = [f"Volume observado: {len(df)} veículos em {duracao_s / 60:.1f} min "
            f"(taxa equivalente de {len(df) / horas:.0f} veíc/h)."]
    ucp_sentido = faixas.groupby("sentido")["ucp"].sum()
    if len(sentidos) == 2 and ucp_sentido.sum() > 0:
        a, b = (float(ucp_sentido.get(s, 0.0)) for s in sentidos)
        fracao = a / (a + b)
        recs.append(f"Fluxo equivalente: {sentidos[0]} {fracao:.0%} ({a / horas:.0f} UCP/h) x {sentidos[1]} "
                    f"{1 - fracao:.0%} ({b / horas:.0f} UCP/h). Sugestão: repartir o tempo de verde útil na "
                    f"proporção {fracao * 100:.0f}/{(1 - fracao) * 100:.0f} entre as aproximações.")
        if max(fracao, 1 - fracao) >= 0.65:
            dominante = sentidos[0] if fracao >= 0.5 else sentidos[1]
            recs.append(f"Fluxo desbalanceado no sentido {dominante}: avaliar onda verde nesse sentido e, "
                        "se o padrão se repetir em outros dias, faixa reversível no horário.")
    frac_pesados = df["categoria"].isin(pesadas).mean()
    if frac_pesados >= 0.15:
        recs.append(f"Veículos pesados são {frac_pesados:.0%} do tráfego: considerar janela de carga e descarga "
                    "fora do pico, fiscalização de peso e manutenção preventiva do pavimento.")
    frac_motos = (df["categoria"] == "moto").mean()
    if frac_motos >= 0.25:
        recs.append(f"Motos são {frac_motos:.0%} do tráfego: avaliar área de espera para motos (motobox) "
                    "no semáforo e campanha de segurança.")
    frac_bicicletas = (df["categoria"] == "bicicleta").mean()
    if frac_bicicletas >= 0.05:
        recs.append(f"Bicicletas são {frac_bicicletas:.0%} do tráfego: o volume justifica estudo de ciclofaixa.")
    if faixas["faixa_inicio"].nunique() > 1:
        por_faixa = faixas.groupby("faixa_inicio")["veic_h"].sum()
        recs.append(f"Faixa de pico: {por_faixa.idxmax()} com {por_faixa.max():.0f} veíc/h; concentrar "
                    "fiscalização e operação de trânsito nesse horário.")
    if alertas:
        vistos = {}
        for a in alertas:
            vistos.setdefault(a["tipo"], []).append(a)
        for tipo, lista in vistos.items():
            recs.append(f"{len(lista)} alerta(s) de {tipo}: {lista[0]['recomendacao']}")
    else:
        recs.append("Nenhum fluxo anormal detectado: manter a programação semafórica atual e seguir monitorando.")
    return recs


def graficos(df, pasta, sentidos):
    if df.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arquivos = []
    tabela = df.groupby(["categoria", "sentido"]).size().unstack(fill_value=0)
    ax = tabela.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_title("Contagem por categoria e sentido")
    ax.set_xlabel("")
    ax.set_ylabel("veículos")
    plt.tight_layout()
    caminho = os.path.join(pasta, "contagem_categorias.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    arquivos.append(caminho)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s in sentidos:
        selecao = df[df["sentido"] == s].sort_values("carimbo")
        if not selecao.empty:
            ax.step(selecao["carimbo"], range(1, len(selecao) + 1), where="post", label=s)
    ax.set_title("Contagem acumulada por sentido")
    ax.set_ylabel("veículos")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    caminho = os.path.join(pasta, "contagem_acumulada.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    arquivos.append(caminho)

    por_minuto = (df.assign(minuto=df["carimbo"].dt.floor("1min"))
                  .groupby(["minuto", "sentido"]).size().unstack(fill_value=0))
    por_minuto.index = por_minuto.index.strftime("%H:%M")
    ax = por_minuto.plot(kind="bar", stacked=True, figsize=(9, 4.5), rot=45)
    ax.set_title("Fluxo por minuto")
    ax.set_xlabel("")
    ax.set_ylabel("veículos/min")
    plt.tight_layout()
    caminho = os.path.join(pasta, "fluxo_por_minuto.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    arquivos.append(caminho)
    return arquivos


def tabela_md(df):
    if df is None or df.empty:
        return "_sem dados_"
    cabecalho = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separador = "|" + "|".join("---" for _ in df.columns) + "|"
    linhas = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cabecalho, separador] + linhas)


def gerar(pasta, contador, monitor, cfg, inicio, duracao_s, descricao_fonte, extras=None):
    os.makedirs(pasta, exist_ok=True)
    sentidos = list(cfg["linha"]["sentidos"])
    ucp = cfg["categorias"]["ucp"]
    pesadas = cfg["categorias"]["pesadas"]

    df = eventos_para_df(contador.tabela_eventos())
    df.to_csv(os.path.join(pasta, "eventos.csv"), index=False, encoding="utf-8")
    pd.DataFrame(contador.duplicatas, columns=["id", "id_original", "quadro", "quadro_id", "tempo_s", "sentido"]
                 ).to_csv(os.path.join(pasta, "duplicatas_suprimidas.csv"), index=False, encoding="utf-8")
    alertas = monitor.alertas
    pd.DataFrame(alertas, columns=["tipo", "severidade", "inicio_s", "fim_s", "carimbo", "observado", "esperado",
                                   "mensagem", "recomendacao"]
                 ).to_csv(os.path.join(pasta, "alertas.csv"), index=False, encoding="utf-8")
    pd.DataFrame(monitor.janelas).to_csv(os.path.join(pasta, "janelas_fluxo.csv"), index=False, encoding="utf-8")

    faixas = tabela_faixas(df, inicio, duracao_s, cfg["relatorio"]["faixa_min"], sentidos, ucp)
    faixas.to_csv(os.path.join(pasta, "contagem_por_faixa.csv"), index=False, encoding="utf-8")

    recs = recomendacoes(df, faixas, alertas, sentidos, pesadas, duracao_s)
    figuras = graficos(df, pasta, sentidos)

    cruzada = pd.crosstab(df["categoria"], df["sentido"], margins=True, margins_name="total") if not df.empty \
        else pd.DataFrame()
    resumo = {
        "fonte": descricao_fonte,
        "inicio": pd.Timestamp(inicio).strftime("%Y-%m-%d %H:%M:%S"),
        "duracao_s": round(duracao_s, 2),
        "total_veiculos": int(len(df)),
        "taxa_veic_h": round(len(df) * 3600.0 / duracao_s, 1) if duracao_s > 0 else 0.0,
        "por_sentido": {k: int(v) for k, v in df["sentido"].value_counts().items()},
        "por_categoria": {k: int(v) for k, v in df["categoria"].value_counts().items()},
        "por_sentido_categoria": {f"{s}|{c}": int(n) for (s, c), n in
                                  df.groupby(["sentido", "categoria"]).size().items()},
        "duplicatas_suprimidas": len(contador.duplicatas),
        "categoria_corrigida_por_votacao": int((df["categoria"] != df["categoria_inicial"]).sum()),
        "alertas": {t: int(n) for t, n in pd.Series([a["tipo"] for a in alertas], dtype=object).value_counts().items()},
        "recomendacoes": recs,
    }
    if extras:
        resumo.update(extras)
    with open(os.path.join(pasta, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    tabela_alertas = pd.DataFrame(alertas, columns=["carimbo", "tipo", "severidade", "observado", "esperado",
                                                    "mensagem"])
    partes = [
        "# Relatório de contagem veicular",
        "",
        f"- Fonte: `{descricao_fonte}`",
        f"- Início: {resumo['inicio']}  |  Duração: {duracao_s / 60:.1f} min",
        f"- Total contado: **{resumo['total_veiculos']}** veículos ({resumo['taxa_veic_h']} veíc/h)",
        f"- Duplicatas suprimidas (troca de identidade): {resumo['duplicatas_suprimidas']}",
        f"- Categorias corrigidas pela votação ao longo da trilha: {resumo['categoria_corrigida_por_votacao']}",
        "",
        "## Recomendações de ação",
        "",
    ]
    partes += [f"{i}. {r}" for i, r in enumerate(recs, 1)]
    partes += ["", "## Contagem por categoria e sentido", "",
               tabela_md(cruzada.reset_index() if not cruzada.empty else cruzada),
               "", f"## Contagem por faixa horária ({cfg['relatorio']['faixa_min']} min)", "", tabela_md(faixas),
               "", "## Alertas de fluxo anormal", "", tabela_md(tabela_alertas), ""]
    if figuras:
        partes += ["## Gráficos", ""] + [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figuras]
    with open(os.path.join(pasta, "relatorio.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(partes) + "\n")
    return resumo
