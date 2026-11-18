# Contagem e classificação de veículos por câmera fixa

Trabalho final de Visão Computacional (UTFPR), Tema 26. O sistema detecta, rastreia e classifica veículos em vídeo
de câmera fixa, conta os cruzamentos de uma linha virtual por categoria, sentido e faixa horária e emite alertas de
fluxo anormal. O artigo está em [artigo/](artigo/) e as fotos tiradas pela equipe em [fotos_nossas/](fotos_nossas/).

## Requisitos

- Python 3.10 a 3.13 de 64 bits
- GPU NVIDIA com CUDA (recomendado; em CPU funciona, mas não em tempo real)
- Cerca de 3 GB livres para o ambiente e mais 6,5 GB para retreinar com o MIO-TCD

## Instalação

```bash
git clone https://github.com/AlmadaAlmada/visaoComputacional.git
cd visaoComputacional
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m unittest discover -s testes -v
```

Sem GPU NVIDIA, instale o PyTorch com `pip install torch torchvision`. Os pesos `yolo11s.pt` e `yolo11s-cls.pt` são
baixados na primeira execução. Os classificadores treinados já estão em `treinos/`.

## Contagem em vídeo

```bash
python definir_linha.py --video <video.mp4> --gravar
python contar.py --video <video.mp4> --inicio "AAAA-MM-DD HH:MM:SS" --saida resultados
```

No `definir_linha.py`, clique dois pontos atravessando as faixas e tecle ENTER; a linha vai para o `config.yaml`.
O `contar.py` grava em `resultados/` o `relatorio.md`, a contagem por faixa horária (`contagem_por_faixa.csv`), os
alertas (`alertas.csv`), os eventos, a latência por etapa e o vídeo anotado.

| Opção do `contar.py` | Efeito |
|---|---|
| `--mostrar` | exibe o processamento ao vivo |
| `--sem-video` | não grava o vídeo anotado |
| `--linha-base 900` | volume esperado (veíc/h) para os alertas |
| `--classificador <pesos.pt>` | troca o classificador de categoria (`nenhum` desliga) |
| `--video <pasta> --fps 30` | processa uma pasta de quadros |

Para medir a contagem contra uma contagem manual:

```bash
python contagem_manual.py --video <video.mp4> --saida referencia_manual.csv
python avaliar.py contagem --sistema resultados/eventos.csv --referencia referencia_manual.csv
```

Na contagem manual, cada tecla registra um veículo: `c` carro, `m` moto, `o` ônibus, `t` caminhão, `v` van,
`b` bicicleta, `x` outro (minúscula para um sentido, maiúscula para o outro; `espaço` pausa, `z` desfaz).

## Imagens próprias (opção A)

As 49 fotos em `fotos_nossas/` já estão com placas, rostos e telefones borrados. `anotacoes.csv` traz uma caixa
por veículo, a classe no padrão do MIO-TCD e o grupo (fotos do mesmo veículo têm o mesmo grupo).
`tarjas.csv` guarda as regiões borradas; `python tarjar.py` refaz as fotos a partir dos originais, que não são
publicados.

Os passos abaixo exigem o `MIO-TCD-Classification.tar` (https://tcd.miovision.com/challenge/dataset.html) na raiz.

```bash
python treinar.py preparar-cls --por-classe 3000
python treinar.py preparar-proprias
```

O primeiro comando monta o recorte balanceado do MIO-TCD em `dados/mio_tcd_cls`. O segundo recorta os veículos das
fotos, separa 40% dos veículos para teste (sem repetir o mesmo veículo nos dois lados) e cria
`dados/mio_tcd_cls_com_proprias`, com o MIO-TCD mais os recortes de treino repetidos 20 vezes.

Treino sem e com as imagens próprias, em três sementes:

```bash
python treinar.py treinar-cls --dados dados/mio_tcd_cls --nome cls_sem_proprias_s0 --semente 0
python treinar.py treinar-cls --dados dados/mio_tcd_cls_com_proprias --nome cls_com_proprias_s0 --semente 0
```

(repita com `--semente 1` e `--semente 2`, trocando o sufixo do nome)

Comparação no teste das imagens próprias e na validação do MIO-TCD:

```bash
python avaliar.py classificador --saida artigo/resultados ^
  --modelos sem=treinos/cls_sem_proprias_s0/weights/best.pt,treinos/cls_sem_proprias_s1/weights/best.pt,treinos/cls_sem_proprias_s2/weights/best.pt ^
            com=treinos/cls_com_proprias_s0/weights/best.pt,treinos/cls_com_proprias_s1/weights/best.pt,treinos/cls_com_proprias_s2/weights/best.pt ^
  --conjuntos proprias=dados/proprias/teste mio=dados/mio_tcd_cls/val
```

No Linux/macOS, troque `^` por `\`. A saída traz acurácia e F1 macro por semente, a média, o F1 por classe, as
matrizes de confusão e a figura de acertos e erros.

## GRAM-RTM (somente teste)

Baixe as sequências em https://gram.web.uah.es/data/datasets/rtm/index.html e, para cada uma:

```bash
python definir_linha.py --video GRAM-RTM/M-30 --config config_m30.yaml --gravar
python contar.py --video GRAM-RTM/M-30 --fps 30 --config config_m30.yaml --saida resultados_m30 --sem-video
python avaliar.py gram-contagem --sistema resultados_m30/eventos.csv --anotacoes GRAM-RTM/Annotations/M-30 --quadros GRAM-RTM/M-30 --fps 30 --config config_m30.yaml
python avaliar.py detector --quadros GRAM-RTM/M-30 --anotacoes GRAM-RTM/Annotations/M-30 --config config_m30.yaml
```

Crie o `config_m30.yaml` copiando o `config.yaml`.

## Estrutura

```
contar.py, definir_linha.py, contagem_manual.py   contagem em vídeo
treinar.py                                        preparação dos dados e treino
avaliar.py                                        avaliação da contagem, do detector e do classificador
tarjar.py                                         borra placas e rostos das fotos próprias
contagem/                                         detector, contador, alertas, relatório, métricas, latência
fotos_nossas/                                     fotos da equipe, anotações e regiões borradas
artigo/                                           artigo (LaTeX, modelo SBC), figuras e resultados
testes/                                           testes automatizados
```
