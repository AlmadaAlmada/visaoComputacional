import argparse
import os
import time
from datetime import datetime, timedelta

import cv2

from contagem import relatorio
from contagem.alertas import monitor_da_config
from contagem.config import carregar_config, resolver_caminho
from contagem.contador import contador_da_config, linha_da_config
from contagem.detector import DetectorRastreador
from contagem.fonte import FonteQuadros
from contagem.visual import desenhar


def argumentos():
    p = argparse.ArgumentParser(description="Contagem de veículos por categoria e sentido com linha virtual")
    p.add_argument("--video", required=True, help="arquivo de vídeo ou pasta de quadros (ex.: sequência GRAM-RTM)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--saida", default="resultados")
    p.add_argument("--inicio", help="data e hora do primeiro quadro, AAAA-MM-DD HH:MM:SS")
    p.add_argument("--fps", type=float, help="fps da fonte (obrigatório para pasta de quadros se diferente de 25)")
    p.add_argument("--pesos", help="substitui modelo.pesos do config")
    p.add_argument("--classificador", help="pesos do classificador de categoria (MIO-TCD); 'nenhum' desliga")
    p.add_argument("--linha-base", type=float, help="volume esperado em veíc/h para o alerta de fluxo anormal")
    p.add_argument("--mostrar", action="store_true", help="exibe o vídeo anotado durante o processamento")
    p.add_argument("--sem-video", action="store_true", help="não grava o vídeo anotado")
    p.add_argument("--max-quadros", type=int)
    return p.parse_args()


def main():
    a = argumentos()
    cfg = carregar_config(a.config)
    if a.pesos:
        cfg["modelo"]["pesos"] = a.pesos
    if a.classificador:
        cfg["modelo"].setdefault("classificador", {})["pesos"] = None if a.classificador == "nenhum" else a.classificador
    cfg["modelo"]["rastreador"] = resolver_caminho(a.config, cfg["modelo"]["rastreador"])
    os.makedirs(a.saida, exist_ok=True)

    fonte = FonteQuadros(a.video, a.fps or cfg["video"].get("fps"))
    inicio = datetime.strptime(a.inicio or str(cfg["video"]["inicio"]), "%Y-%m-%d %H:%M:%S")
    sentidos = tuple(cfg["linha"]["sentidos"])
    linha = linha_da_config(cfg, fonte.largura, fonte.altura)
    contador = contador_da_config(cfg, linha)
    monitor = monitor_da_config(cfg, inicio, a.linha_base)
    detector = DetectorRastreador(cfg)

    escritor = None
    if not a.sem_video:
        escritor = cv2.VideoWriter(os.path.join(a.saida, "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                   fonte.fps, (fonte.largura, fonte.altura))
    print(f"Fonte: {a.video} | {fonte.largura}x{fonte.altura} @ {fonte.fps:.2f} fps | {fonte.total} quadros")
    print(f"Modelo: {cfg['modelo']['pesos']} | categorias: {sorted(set(detector.categorias.values()))}")
    if detector.classificador is not None:
        print(f"Classificador de categoria: {cfg['modelo']['classificador']['pesos']}")

    exibidos = []
    processados = 0
    t0 = time.time()
    try:
        for i, quadro_id, quadro in fonte:
            if a.max_quadros and i >= a.max_quadros:
                break
            tempo = i / fonte.fps
            carimbo = (inicio + timedelta(seconds=tempo)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            deteccoes = detector.processar(quadro)
            novos = contador.atualizar(i, quadro_id, tempo, deteccoes, carimbo)
            parados = contador.parados(tempo, monitor.parado_s)
            for alerta in monitor.atualizar(tempo, len(parados)):
                exibidos.append((tempo, alerta))
                print(f"[{alerta['carimbo']}] ALERTA {alerta['tipo']}: {alerta['mensagem']}")
            for evento in novos:
                monitor.registrar(evento)
            processados = i + 1
            if escritor is not None or a.mostrar:
                ativos = [al for t, al in exibidos if tempo - t <= 5.0]
                desenhar(quadro, linha, deteccoes, contador, ativos, sentidos, carimbo)
                if escritor is not None:
                    escritor.write(quadro)
                if a.mostrar:
                    cv2.imshow("contagem", quadro)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if processados % 100 == 0:
                velocidade = processados / (time.time() - t0)
                print(f"quadro {processados}/{fonte.total} | {velocidade:.1f} q/s | contados: {len(contador.eventos)}")
    finally:
        fonte.fechar()
        if escritor is not None:
            escritor.release()
        if a.mostrar:
            cv2.destroyAllWindows()

    duracao = processados / fonte.fps
    for alerta in monitor.encerrar(duracao):
        print(f"[{alerta['carimbo']}] ALERTA {alerta['tipo']}: {alerta['mensagem']}")
    decorrido = time.time() - t0
    resumo = relatorio.gerar(a.saida, contador, monitor, cfg, inicio, duracao, a.video, {
        "quadros": processados,
        "fps_video": round(fonte.fps, 3),
        "fps_processamento": round(processados / decorrido, 2) if decorrido > 0 else None,
        "modelo": cfg["modelo"]["pesos"],
        "classificador": (cfg["modelo"].get("classificador") or {}).get("pesos"),
        "linha_px": [list(map(round, linha.p1)), list(map(round, linha.p2))],
    })

    print()
    print(f"Total contado: {resumo['total_veiculos']} veículos em {duracao / 60:.1f} min")
    for chave, n in sorted(resumo["por_sentido_categoria"].items()):
        print(f"  {chave}: {n}")
    print("Recomendações:")
    for i, r in enumerate(resumo["recomendacoes"], 1):
        print(f"  {i}. {r}")
    print(f"Resultados em: {os.path.abspath(a.saida)}")


if __name__ == "__main__":
    main()
