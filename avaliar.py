import argparse
import json
import os

import pandas as pd

from contagem.avaliacao import avaliar_contagem, avaliar_deteccoes
from contagem.config import carregar_config
from contagem.relatorio import tabela_md


def _escrever(resultado, pasta, titulo, extras=None):
    os.makedirs(pasta, exist_ok=True)
    resultado["totais"].to_csv(os.path.join(pasta, "totais.csv"), index=False, encoding="utf-8")
    total = resultado["totais"].iloc[-1]
    resumo = {"referencia": int(total["referencia"]), "sistema": int(total["sistema"]),
              "erro_pct_total": None if pd.isna(total["erro_pct"]) else float(total["erro_pct"]),
              "erro_abs_pct_por_classe_medio": round(float(
                  resultado["totais"].loc[resultado["totais"]["categoria"] != "TOTAL", "erro_pct"].abs().mean()), 2)}
    partes = [f"# {titulo}", "", f"- Contagem de referência: **{resumo['referencia']}**",
              f"- Contagem do sistema: **{resumo['sistema']}**",
              f"- Erro percentual total: **{resumo['erro_pct_total']}%**", ""]
    if "pares" in resultado:
        for chave in ("pares", "extras", "perdidos", "precisao_contagem", "revocacao_contagem", "acuracia_categoria"):
            resumo[chave] = resultado[chave]
        partes += [f"- Eventos pareados (mesmo veículo, mesmo sentido): {resultado['pares']}",
                   f"- Contagens extras: {resultado['extras']}  |  Contagens perdidas: {resultado['perdidos']}",
                   f"- Precisão da contagem: {resultado['precisao_contagem']:.3f}  |  "
                   f"Revocação da contagem: {resultado['revocacao_contagem']:.3f}",
                   f"- Acurácia de categoria (eventos pareados): **{resultado['acuracia_categoria']:.3f}**", ""]
        resultado["erros"].to_csv(os.path.join(pasta, "erros.csv"), index=False, encoding="utf-8")
        resultado["causas"].to_csv(os.path.join(pasta, "causas.csv"), index=False, encoding="utf-8")
        resultado["confusao"].to_csv(os.path.join(pasta, "matriz_confusao.csv"), encoding="utf-8")
        resultado["acuracia_por_categoria"].to_csv(os.path.join(pasta, "acuracia_por_categoria.csv"), index=False,
                                                   encoding="utf-8")
    if extras:
        resumo.update(extras)
    partes += ["## Erro por sentido e categoria", "", tabela_md(resultado["totais"]), ""]
    if "pares" in resultado:
        partes += ["## Acurácia por categoria", "", tabela_md(resultado["acuracia_por_categoria"]), "",
                   "## Matriz de confusão (linhas = referência)", "",
                   tabela_md(resultado["confusao"].reset_index()), "",
                   "## Causas de contagem dupla ou perdida", "", tabela_md(resultado["causas"]), "",
                   "## Lista de erros para inspeção no vídeo", "", tabela_md(resultado["erros"]), ""]
    with open(os.path.join(pasta, "avaliacao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(partes) + "\n")
    with open(os.path.join(pasta, "avaliacao.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print("\n".join(partes[:12]))
    print(f"Detalhes em: {os.path.abspath(pasta)}")


def cmd_contagem(a):
    sis = pd.read_csv(a.sistema)
    ref = pd.read_csv(a.referencia)
    coluna = "tempo_s" if "tempo_s" in ref.columns else None
    if coluna and a.deslocamento:
        ref[coluna] = ref[coluna] + a.deslocamento
    resultado = avaliar_contagem(sis, ref, coluna, a.tolerancia)
    _escrever(resultado, a.saida, "Avaliação da contagem contra a referência manual",
              {"tolerancia_s": a.tolerancia})


def _tamanho_quadros(pasta):
    from contagem.fonte import FonteQuadros
    fonte = FonteQuadros(pasta)
    return fonte.largura, fonte.altura


def cmd_gram_contagem(a):
    from contagem.gram import eventos_referencia, ler_anotacoes, resumo_anotacoes
    cfg = carregar_config(a.config)
    objetos = ler_anotacoes(a.anotacoes)
    print(f"Anotações: {resumo_anotacoes(objetos)}")
    largura, altura = _tamanho_quadros(a.quadros)
    ref = pd.DataFrame(eventos_referencia(objetos, cfg, largura, altura, a.fps))
    if ref.empty:
        raise SystemExit("Nenhum cruzamento nas trilhas anotadas: confira a linha no config e o caminho das anotações")
    os.makedirs(a.saida, exist_ok=True)
    ref.to_csv(os.path.join(a.saida, "eventos_referencia.csv"), index=False, encoding="utf-8")
    sis = pd.read_csv(a.sistema)
    resultado = avaliar_contagem(sis, ref, "quadro_id", a.tolerancia * a.fps)
    _escrever(resultado, a.saida, "Avaliação da contagem no GRAM-RTM (somente teste)",
              {"tolerancia_quadros": a.tolerancia * a.fps})


def cmd_detector(a):
    from contagem.detector import DetectorRastreador
    from contagem.fonte import FonteQuadros
    from contagem.gram import ler_anotacoes
    cfg = carregar_config(a.config)
    if a.pesos:
        cfg["modelo"]["pesos"] = a.pesos
    mapa = cfg["categorias"]["mapa"]
    objetos = ler_anotacoes(a.anotacoes)
    gts_por_quadro = {}
    for o in objetos:
        gts_por_quadro.setdefault(o["quadro"], []).append((mapa.get(o["classe"], "outro"), o["caixa"]))
    detector = DetectorRastreador(cfg)
    fonte = FonteQuadros(a.quadros)
    predicoes, gts, avaliados = [], [], 0
    for i, quadro_id, quadro in fonte:
        if i % a.passo or quadro_id not in gts_por_quadro:
            continue
        for categoria, conf, caixa in detector.detectar(quadro, a.conf_min):
            predicoes.append((quadro_id, categoria, conf, caixa))
        gts.extend((quadro_id, c, caixa) for c, caixa in gts_por_quadro[quadro_id])
        avaliados += 1
    tabela, map50 = avaliar_deteccoes(predicoes, gts, a.iou, detector.conf_operacao)
    os.makedirs(a.saida, exist_ok=True)
    tabela.to_csv(os.path.join(a.saida, "detector.csv"), index=False, encoding="utf-8")
    texto = "\n".join([f"# Avaliação do detector ({cfg['modelo']['pesos']})", "",
                       f"- Quadros avaliados: {avaliados} (1 a cada {a.passo})",
                       f"- IoU mínimo: {a.iou}  |  confiança de operação: {detector.conf_operacao}",
                       f"- mAP@0.5 (média das categorias com anotação): **{map50}**", "",
                       tabela_md(tabela), ""])
    with open(os.path.join(a.saida, "detector.md"), "w", encoding="utf-8") as f:
        f.write(texto)
    print(texto)


def main():
    p = argparse.ArgumentParser(description="Avaliação da contagem e do detector")
    sub = p.add_subparsers(dest="comando", required=True)

    c = sub.add_parser("contagem", help="compara eventos.csv do sistema com a contagem manual da equipe")
    c.add_argument("--sistema", required=True)
    c.add_argument("--referencia", required=True)
    c.add_argument("--tolerancia", type=float, default=2.0, help="segundos para parear evento do sistema e manual")
    c.add_argument("--deslocamento", type=float, default=0.0, help="soma ao tempo da referência (atraso de reação)")
    c.add_argument("--saida", default="avaliacao")
    c.set_defaults(func=cmd_contagem)

    g = sub.add_parser("gram-contagem", help="gera a referência a partir das trilhas do GRAM-RTM e compara")
    g.add_argument("--sistema", required=True)
    g.add_argument("--anotacoes", required=True)
    g.add_argument("--quadros", required=True)
    g.add_argument("--config", default="config.yaml")
    g.add_argument("--fps", type=float, default=25.0)
    g.add_argument("--tolerancia", type=float, default=1.0)
    g.add_argument("--saida", default="avaliacao_gram")
    g.set_defaults(func=cmd_gram_contagem)

    d = sub.add_parser("detector", help="precisão, revocação e AP50 do detector quadro a quadro")
    d.add_argument("--quadros", required=True)
    d.add_argument("--anotacoes", required=True)
    d.add_argument("--config", default="config.yaml")
    d.add_argument("--pesos")
    d.add_argument("--passo", type=int, default=10)
    d.add_argument("--iou", type=float, default=0.5)
    d.add_argument("--conf-min", type=float, default=0.05)
    d.add_argument("--saida", default="avaliacao_detector")
    d.set_defaults(func=cmd_detector)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
