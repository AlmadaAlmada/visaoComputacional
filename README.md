# Contagem e classificação de veículos por câmera fixa

Detecta, rastreia e classifica veículos em vídeo de câmera fixa, conta os cruzamentos de uma linha virtual por
categoria, sentido e faixa horária e emite alertas de fluxo anormal com recomendação de ação.
A fundamentação, os resultados e a discussão estão no [relatório final](RELATORIO.md).

## Requisitos

- Windows, Linux ou macOS com **Python 3.10 a 3.13 de 64 bits**
- GPU NVIDIA com CUDA (recomendado; em CPU o sistema funciona, mas não em tempo real)
- Cerca de 3 GB livres para o ambiente; mais 6,5 GB se for retreinar os modelos com o MIO-TCD

## Instalação

```bash
git clone <url-do-repositorio>
cd visaoComputacional
python -m venv .venv
```

Ative o ambiente:

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
```

Instale o PyTorch com CUDA e as dependências:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Sem GPU NVIDIA, troque a primeira linha por `pip install torch torchvision`.

Confira a instalação rodando os testes:

```bash
python -m unittest discover -s testes -v
```

Os pesos do detector (`yolo11s.pt`) são baixados automaticamente na primeira execução. O classificador de
categoria já treinado acompanha o repositório em `treinos/mio_tcd_cls/weights/best.pt`.

## Uso

### 1. Definir a linha de contagem

```bash
python definir_linha.py --video <video.mp4> --gravar
```

Clique em dois pontos formando uma linha que atravesse todas as faixas a contar e pressione ENTER (`R` limpa,
`ESC` sai). A linha é gravada no `config.yaml` em coordenadas normalizadas. As setas mostram qual sentido
recebe cada nome de `linha.sentidos`; renomeie no `config.yaml` para os nomes reais da via.

### 2. Contar

```bash
python contar.py --video <video.mp4> --inicio "AAAA-MM-DD HH:MM:SS" --saida resultados
```

| Opção | Efeito |
|---|---|
| `--inicio` | data e hora do primeiro quadro (usadas nas faixas horárias) |
| `--mostrar` | exibe o processamento ao vivo (`q` encerra) |
| `--sem-video` | não grava o vídeo anotado |
| `--linha-base 900` | volume esperado em veíc/h para os alertas; sem ela, a linha de base é aprendida |
| `--classificador nenhum` | desliga o classificador de categoria (usa só o detector) |
| `--pesos <arquivo.pt>` | troca o detector, por exemplo `treinos/mio_tcd_det/weights/best.pt` |
| `--video <pasta>` `--fps 30` | processa uma pasta de quadros em vez de um vídeo |

Arquivos gerados em `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | resumo, recomendações, contagem por categoria e sentido, faixas horárias, alertas, latência e gráficos |
| `contagem_por_faixa.csv` | contagem por categoria e sentido em cada faixa horária, com veíc/h e UCP/h |
| `alertas.csv` | alertas de fluxo anormal com mensagem e recomendação |
| `eventos.csv` | um registro por veículo contado |
| `latencia.csv` | tempo por etapa (média, p50, p95, p99, máximo) |
| `resumo.json` | totais, alertas, recomendações e latência para integração |
| `janelas_fluxo.csv`, `duplicatas_suprimidas.csv` | dados de apoio |
| `video_anotado.mp4`, `*.png` | visualização |

### 3. Fazer a contagem manual de referência

```bash
python contagem_manual.py --video <video.mp4> --saida referencia_manual.csv
```

O vídeo roda a 0,5× com a linha desenhada. A cada veículo que cruza, aperte a tecla da categoria:
`c` carro, `m` moto, `o` ônibus, `t` caminhão, `v` van, `b` bicicleta, `x` outro.
Minúscula registra o primeiro sentido e MAIÚSCULA o segundo. `espaço` pausa, `z` desfaz, `+`/`-` mudam a
velocidade e `q` salva e sai.

### 4. Avaliar a contagem contra a referência manual

```bash
python avaliar.py contagem --sistema resultados/eventos.csv --referencia referencia_manual.csv --saida avaliacao
```

Gera `avaliacao/avaliacao.md` com erro percentual de contagem, acurácia por categoria, matriz de confusão e
causas de cada contagem dupla ou perdida (`erros.csv` traz o instante de cada erro). Se a contagem manual
estiver sistematicamente atrasada, use `--deslocamento -0.5`.

### 5. Avaliar no GRAM-RTM

Baixe as sequências em https://gram.web.uah.es/data/datasets/rtm/index.html e, para cada sequência:

```bash
copy config.yaml config_m30.yaml
python definir_linha.py --video GRAM-RTM\M-30 --config config_m30.yaml --gravar
python contar.py --video GRAM-RTM\M-30 --fps 30 --config config_m30.yaml --saida resultados_m30 --sem-video
python avaliar.py gram-contagem --sistema resultados_m30/eventos.csv --anotacoes GRAM-RTM\Annotations\M-30 --quadros GRAM-RTM\M-30 --fps 30 --config config_m30.yaml --saida avaliacao_m30
python avaliar.py detector --quadros GRAM-RTM\M-30 --anotacoes GRAM-RTM\Annotations\M-30 --config config_m30.yaml --saida avaliacao_detector_m30
```

O primeiro `avaliar.py` mede a contagem; o segundo mede o detector quadro a quadro (AP50, precisão,
revocação). Use `--fps 25` na sequência Urban1. No Linux/macOS, troque `copy` por `cp` e `\` por `/`.

### 6. Retreinar os modelos (opcional)

Baixe `MIO-TCD-Classification.tar` e `MIO-TCD-Localization.tar` em
https://tcd.miovision.com/challenge/dataset.html e coloque-os na raiz do projeto.

```bash
python treinar.py preparar-cls --por-classe 3000
python treinar.py treinar-cls
python treinar.py preparar --amostra 20000
python treinar.py treinar --epocas 25 --nome mio_tcd_det
```

Os dois primeiros comandos treinam o classificador de categoria usado por padrão; os dois últimos treinam o
detector alternativo.

## Estrutura

```
contar.py               contagem em vídeo
definir_linha.py        definição da linha virtual
contagem_manual.py      contagem manual de referência
avaliar.py              avaliação da contagem e do detector
treinar.py              preparação dos dados e treino no MIO-TCD
config.yaml             parâmetros do sistema
rastreador.yaml         parâmetros do ByteTrack
contagem/
  detector.py           detector YOLO, rastreamento e classificador de categoria
  contador.py           linha virtual e lógica de cruzamento
  alertas.py            monitor de fluxo e alertas
  relatorio.py          relatório, tabelas, gráficos e recomendações
  avaliacao.py          métricas de contagem e de detecção
  gram.py               leitura das anotações do GRAM-RTM
  latencia.py           medição de latência por etapa
  fonte.py, visual.py, config.py
testes/                 testes automatizados
```
