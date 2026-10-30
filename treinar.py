import argparse
import csv
import io
import os
import random
import shutil
from collections import Counter, defaultdict

CLASSES_MIO = ["articulated_truck", "bicycle", "bus", "car", "motorcycle", "motorized_vehicle",
               "non-motorized_vehicle", "pedestrian", "pickup_truck", "single_unit_truck", "work_van"]


def _ler_gt(linhas):
    caixas = defaultdict(list)
    for linha in csv.reader(linhas):
        if len(linha) < 6 or linha[1].strip() not in CLASSES_MIO:
            continue
        caixas[linha[0].strip()].append((CLASSES_MIO.index(linha[1].strip()), *map(float, linha[2:6])))
    return caixas


def _selecionar(caixas, amostra, limiar_raras, rng):
    ids = sorted(caixas)
    rng.shuffle(ids)
    if not amostra or amostra >= len(ids):
        return ids
    contagem = Counter(c for lista in caixas.values() for c, *_ in lista)
    raras = {c for c, n in contagem.items() if n < limiar_raras}
    com_raras = [i for i in ids if any(c in raras for c, *_ in caixas[i])]
    vistos = set(com_raras)
    escolhidos = (com_raras + [i for i in ids if i not in vistos])[:amostra]
    rng.shuffle(escolhidos)
    return escolhidos


def _rotulos(caixas, w, h):
    linhas = []
    for c, x1, y1, x2, y2 in caixas:
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        linhas.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return "\n".join(linhas) + ("\n" if linhas else "")


def _gravar(a, divisao, ident, nome_arquivo, conteudo, caixas):
    from PIL import Image

    with Image.open(io.BytesIO(conteudo)) as im:
        w, h = im.size
    with open(os.path.join(a.saida, "images", divisao, nome_arquivo), "wb") as f:
        f.write(conteudo)
    nome = os.path.splitext(nome_arquivo)[0] + ".txt"
    with open(os.path.join(a.saida, "labels", divisao, nome), "w", encoding="utf-8") as f:
        f.write(_rotulos(caixas[ident], w, h))


def preparar(a):
    import tarfile

    eh_tar = os.path.isfile(a.mio_tcd)
    if eh_tar:
        with tarfile.open(a.mio_tcd, "r") as tar:
            membro = next(m for m in tar if m.name.replace("\\", "/").endswith("gt_train.csv"))
            caixas = _ler_gt(io.TextIOWrapper(tar.extractfile(membro), encoding="utf-8"))
    else:
        with open(os.path.join(a.mio_tcd, "gt_train.csv"), newline="", encoding="utf-8") as f:
            caixas = _ler_gt(f)
    rng = random.Random(a.semente)
    ids = _selecionar(caixas, a.amostra, a.limiar_raras, rng)
    n_val = max(1, int(len(ids) * a.val))
    divisao_de = {i: ("val" if k < n_val else "train") for k, i in enumerate(ids)}
    print(f"{len(ids)} imagens selecionadas | treino {len(ids) - n_val} | validação {n_val}")
    contagem = Counter(CLASSES_MIO[c] for i in ids for c, *_ in caixas[i])
    for classe in CLASSES_MIO:
        print(f"  {classe}: {contagem[classe]} caixas")
    for divisao in ("train", "val"):
        os.makedirs(os.path.join(a.saida, "images", divisao), exist_ok=True)
        os.makedirs(os.path.join(a.saida, "labels", divisao), exist_ok=True)

    gravados = 0
    if eh_tar:
        with tarfile.open(a.mio_tcd, "r|*") as tar:
            for membro in tar:
                partes = membro.name.replace("\\", "/").split("/")
                if not membro.isfile() or len(partes) < 2 or partes[-2] != "train":
                    continue
                ident = os.path.splitext(partes[-1])[0]
                if ident not in divisao_de:
                    continue
                _gravar(a, divisao_de[ident], ident, partes[-1], tar.extractfile(membro).read(), caixas)
                gravados += 1
                if gravados % 5000 == 0:
                    print(f"  {gravados}/{len(ids)} imagens")
    else:
        for ident, divisao in divisao_de.items():
            caminho = os.path.join(a.mio_tcd, "train", ident + ".jpg")
            if os.path.exists(caminho):
                with open(caminho, "rb") as f:
                    _gravar(a, divisao, ident, ident + ".jpg", f.read(), caixas)
                gravados += 1
    if gravados < len(ids):
        print(f"  {len(ids) - gravados} imagens do gt_train.csv não foram encontradas")

    arquivo_dados = os.path.join(a.saida, "dados.yaml")
    with open(arquivo_dados, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(a.saida).replace(os.sep, '/')}\n")
        f.write("train: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(CLASSES_MIO):
            f.write(f"  {i}: {n}\n")
    print(f"Dataset YOLO pronto: {arquivo_dados} ({gravados} imagens)")


def preparar_cls(a):
    import tarfile

    limite = defaultdict(int)
    rng = random.Random(a.semente)
    extraidos = 0
    with tarfile.open(a.tar, "r|*") as tar:
        for membro in tar:
            partes = membro.name.replace("\\", "/").split("/")
            if not membro.isfile() or len(partes) < 3 or partes[-3] != "train" or not partes[-1].endswith(".jpg"):
                continue
            classe = partes[-2]
            if a.classes and classe not in a.classes:
                continue
            if limite[classe] >= a.por_classe:
                continue
            limite[classe] += 1
            divisao = "val" if rng.random() < a.val else "train"
            destino = os.path.join(a.saida, divisao, classe)
            os.makedirs(destino, exist_ok=True)
            origem = tar.extractfile(membro)
            with open(os.path.join(destino, partes[-1]), "wb") as f:
                shutil.copyfileobj(origem, f)
            extraidos += 1
            if extraidos % 5000 == 0:
                print(f"  {extraidos} imagens extraídas")
    for classe, n in sorted(limite.items()):
        print(f"  {classe}: {n}")
    print(f"Dataset de classificação pronto: {os.path.abspath(a.saida)} ({extraidos} imagens)")


def treinar_cls(a):
    from ultralytics import YOLO

    modelo = YOLO(a.base)
    modelo.train(data=a.dados, epochs=a.epocas, imgsz=a.imgsz, batch=a.lote, device=a.dispositivo,
                 project=os.path.abspath(a.projeto), name=a.nome, patience=a.paciencia, workers=a.workers,
                 plots=True, exist_ok=True)
    melhor = os.path.join(str(modelo.trainer.save_dir), "weights", "best.pt")
    metricas = YOLO(melhor).val(data=a.dados, imgsz=a.imgsz, batch=a.lote, device=a.dispositivo, plots=True)
    print(f"Acurácia top-1: {metricas.top1:.4f} | top-5: {metricas.top5:.4f}")
    print(f"Pesos treinados: {melhor}")
    print(f"Configure modelo.classificador: {melhor.replace(os.sep, '/')}")


def treinar(a):
    from ultralytics import YOLO

    modelo = YOLO(a.base)
    modelo.train(data=a.dados, epochs=a.epocas, imgsz=a.imgsz, batch=a.lote, device=a.dispositivo,
                 project=os.path.abspath(a.projeto), name=a.nome, patience=a.paciencia, workers=a.workers, cos_lr=True, exist_ok=True,
                 close_mosaic=5, plots=True)
    melhor = os.path.join(str(modelo.trainer.save_dir), "weights", "best.pt")
    metricas = YOLO(melhor).val(data=a.dados, imgsz=a.imgsz, batch=a.lote, device=a.dispositivo, plots=True)
    print(f"mAP50: {metricas.box.map50:.4f} | mAP50-95: {metricas.box.map:.4f}")
    print(f"Pesos treinados: {melhor}")
    print(f"Use: python contar.py --video ... --pesos {melhor}")


def main():
    p = argparse.ArgumentParser(description="Treino do detector no MIO-TCD (localização)")
    sub = p.add_subparsers(dest="comando", required=True)

    pr = sub.add_parser("preparar", help="converte o MIO-TCD-Localization para o formato YOLO")
    pr.add_argument("--mio-tcd", default="MIO-TCD-Localization.tar", help="o .tar ou a pasta extraída")
    pr.add_argument("--saida", default="dados/mio_tcd_yolo")
    pr.add_argument("--amostra", type=int, help="usa só N imagens (o conjunto completo é grande)")
    pr.add_argument("--val", type=float, default=0.1)
    pr.add_argument("--limiar-raras", type=int, default=15000,
                    help="classes com menos caixas que isso têm suas imagens priorizadas na amostra")
    pr.add_argument("--semente", type=int, default=42)
    pr.set_defaults(func=preparar)

    t = sub.add_parser("treinar", help="ajuste fino de um YOLO pré-treinado no COCO")
    t.add_argument("--dados", default="dados/mio_tcd_yolo/dados.yaml")
    t.add_argument("--base", default="yolo11s.pt")
    t.add_argument("--epocas", type=int, default=30)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--lote", type=int, default=16)
    t.add_argument("--dispositivo", default=None)
    t.add_argument("--paciencia", type=int, default=10)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--projeto", default="treinos")
    t.add_argument("--nome", default="mio_tcd")
    t.set_defaults(func=treinar)

    pc = sub.add_parser("preparar-cls", help="extrai um subconjunto balanceado do MIO-TCD-Classification.tar")
    pc.add_argument("--tar", default="MIO-TCD-Classification.tar")
    pc.add_argument("--saida", default="dados/mio_tcd_cls")
    pc.add_argument("--por-classe", type=int, default=3000)
    pc.add_argument("--classes", nargs="*")
    pc.add_argument("--val", type=float, default=0.15)
    pc.add_argument("--semente", type=int, default=42)
    pc.set_defaults(func=preparar_cls)

    tc = sub.add_parser("treinar-cls", help="treina o classificador de categoria de veículo")
    tc.add_argument("--dados", default="dados/mio_tcd_cls")
    tc.add_argument("--base", default="yolo11s-cls.pt")
    tc.add_argument("--epocas", type=int, default=20)
    tc.add_argument("--imgsz", type=int, default=128)
    tc.add_argument("--lote", type=int, default=128)
    tc.add_argument("--dispositivo", default=None)
    tc.add_argument("--paciencia", type=int, default=6)
    tc.add_argument("--workers", type=int, default=4)
    tc.add_argument("--projeto", default="treinos")
    tc.add_argument("--nome", default="mio_tcd_cls")
    tc.set_defaults(func=treinar_cls)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
