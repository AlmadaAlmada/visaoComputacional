# Contagem veicular por câmera fixa

Sistema de visão computacional que transforma o vídeo de uma câmera de poste em **contagem de veículos por
categoria e sentido**, **relatório por faixa horária** e **alertas de fluxo anormal com recomendação de ação**
(ajuste de tempo de verde, despacho de equipe, verificação de câmera, restrição de carga etc.).

Aulas relacionadas: 7 (detecção e NMS) e 9 (rastreamento, identidade, implantação e monitoramento).

## Pipeline

```
vídeo ──► detector YOLO (NMS por classe) ──► ByteTrack (ID estável) ──► linha virtual com histerese
                                                                            │
           ┌────────────────────────────────────────────────────────────────┤
           ▼                                ▼                               ▼
  votação de categoria na trilha   anti-duplicata (troca de ID)   detecção de veículos parados
           │                                │                               │
           └──────────────► eventos de cruzamento (id, hora, sentido, categoria) ◄──┘
                                            │
                    ┌───────────────────────┼─────────────────────────┐
                    ▼                       ▼                         ▼
        relatório por faixa horária   monitor de fluxo (janelas)   avaliação vs. referência
        (veíc/h, UCP/h, gráficos)     FLUXO_ALTO / FLUXO_BAIXO /    (erro %, acurácia por
                    │                 SEM_FLUXO / CONGESTIONAMENTO / categoria, causas de
                    ▼                 PESADOS                        dupla/perdida)
           RECOMENDAÇÕES DE AÇÃO ◄────────────┘
```

| Etapa | Arquivo | O que faz |
|---|---|---|
| Detecção + rastreamento | [contagem/detector.py](contagem/detector.py) | YOLO (COCO ou ajustado no MIO-TCD) + ByteTrack ([rastreador.yaml](rastreador.yaml)), classes mapeadas para categorias PT-BR |
| Linha e contagem | [contagem/contador.py](contagem/contador.py) | lado da linha por distância assinada, zona morta, cruzamento só dentro do segmento, 1 contagem por ID, anti-duplicata, votação de categoria |
| Alertas | [contagem/alertas.py](contagem/alertas.py) | linha de base adaptativa (EWMA) ou informada, z-score com piso de Poisson, fila parada, participação de pesados |
| Relatório | [contagem/relatorio.py](contagem/relatorio.py) | tabela por faixa horária e sentido, UCP, gráficos, recomendações, `relatorio.md` |
| Avaliação | [contagem/avaliacao.py](contagem/avaliacao.py), [contagem/gram.py](contagem/gram.py) | pareamento de eventos, erro %, acurácia de categoria, matriz de confusão, causas; AP50 do detector |
| CLIs | [contar.py](contar.py), [avaliar.py](avaliar.py), [treinar.py](treinar.py), [definir_linha.py](definir_linha.py), [contagem_manual.py](contagem_manual.py) | |

## Instalação

Requer Python 64 bits (o PyTorch não roda em Python 32 bits) e, de preferência, GPU NVIDIA.

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Os pesos `yolo11s.pt` (COCO) são baixados automaticamente na primeira execução.

## Uso passo a passo

### 1. Definir a linha de contagem

```bash
python definir_linha.py --video meu_video.mp4 --gravar
```

Clique dois pontos atravessando **toda a largura da via** (as faixas que devem ser contadas) e pressione ENTER.
As coordenadas ficam normalizadas (0–1) no [config.yaml](config.yaml), então valem para qualquer resolução.
As setas mostram qual sentido recebe cada nome de `linha.sentidos` (padrão: `aproximando`, `afastando`); renomeie
para o nome real das vias, por exemplo `centro` e `bairro`.

Dica: posicione a linha onde os veículos aparecem inteiros e separados (não no fundo da imagem, onde ficam
pequenos e sobrepostos) e longe de onde eles param no semáforo.

### 2. Rodar a contagem

```bash
python contar.py --video meu_video.mp4 --inicio "2026-10-30 07:15:00" --saida resultados
```

Opções úteis: `--mostrar` (janela ao vivo), `--pesos treinos/mio_tcd/weights/best.pt` (detector ajustado),
`--linha-base 900` (volume esperado em veíc/h para os alertas), `--sem-video`, `--fps 25` (pasta de quadros).

Saídas em `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | relatório completo: recomendações, tabela categoria × sentido, faixas horárias, alertas, gráficos |
| `eventos.csv` | um registro por veículo contado: ID, quadro, hora, sentido, categoria final e inicial |
| `contagem_por_faixa.csv` | **contagem por categoria e sentido por faixa horária**, com veíc/h e UCP/h |
| `alertas.csv` | **alertas de fluxo anormal** com mensagem e recomendação |
| `janelas_fluxo.csv` | contagem por janela de monitoramento e valor esperado |
| `duplicatas_suprimidas.csv` | cruzamentos descartados por troca de identidade |
| `resumo.json` | totais, taxa, alertas, recomendações (para integrar com outro sistema) |
| `video_anotado.mp4` | caixas, IDs, trilhas, linha, placar e alertas |
| `*.png` | contagem por categoria, acumulada e fluxo por minuto |

### 3. Contagem manual de referência (ground truth da equipe)

```bash
python contagem_manual.py --video meu_video.mp4 --saida referencia_manual.csv
```

Reproduz o vídeo a 0,5× com a mesma linha desenhada. A cada veículo que cruza, aperte a tecla da categoria:
`c` carro, `m` moto, `o` ônibus, `t` caminhão, `v` van, `b` bicicleta, `x` outro. **Minúscula** = primeiro
sentido, **MAIÚSCULA** = segundo sentido. `espaço` pausa, `z` desfaz, `+`/`-` muda a velocidade.
Cada tecla grava o instante do vídeo, o que permite parear veículo a veículo com o sistema.

Recomendação: duas pessoas contam de forma independente e as divergências são revisadas no vídeo; o arquivo final
é a referência. Também é aceita uma referência agregada (`sentido,categoria,quantidade`) feita em papel.

### 4. Avaliar contra a referência manual

```bash
python avaliar.py contagem --sistema resultados/eventos.csv --referencia referencia_manual.csv --saida avaliacao
```

Gera `avaliacao/avaliacao.md` com:

- **erro percentual de contagem** total, por sentido e por categoria;
- **acurácia por categoria** e matriz de confusão (sobre os veículos pareados);
- precisão e revocação **da contagem** (evento do sistema ↔ evento manual, mesmo sentido, dentro de `--tolerancia` s);
- **causas** de cada contagem extra ou perdida (`erros.csv` traz o instante para conferir no vídeo):
  `contagem_dupla`, `falso_positivo`, `veiculos_simultaneos` (NMS/oclusão), `objeto_pequeno`,
  `nao_detectado_ou_trilha_perdida`, `sentido_invertido`, `categoria_errada`.

Se a equipe aperta as teclas sistematicamente atrasada, use `--deslocamento -0.5`.

## Dados

| Dataset | Uso | Observação |
|---|---|---|
| MIO-TCD Classification | **treino** do classificador de categoria | recortes em 11 classes + fundo, CC BY-NC-SA 4.0 |
| MIO-TCD Localization | treino do detector (opcional) | mesmas classes com caixas; quadros isolados, sem identidade |
| GRAM-RTM | **somente teste** da contagem | vídeo contínuo com caixas e ID de trajetória; os autores proíbem uso em treino |
| Vídeo local (3 min) | teste principal | gravado de ponto elevado; a contagem manual da equipe é o ground truth que vale |
| VisDrone2019-DET | não usado | ponto de vista de drone, objetos pequenos e geometria diferente da câmera de poste |
| UA-DETRAC | não usado | site oficial desativado, só espelhos sem licença clara |

### Classificador de categoria com o MIO-TCD Classification (usado no projeto)

A versão Classification do MIO-TCD traz ~520 mil recortes de veículos em 11 classes + `background`, sem caixas.
Ela não treina um detector, então o sistema usa **dois estágios**: o YOLO (COCO) localiza e rastreia, e um
classificador YOLO11-cls treinado no MIO-TCD reclassifica o recorte de cada veículo rastreado em todo quadro.
A categoria final continua sendo a votação ao longo da trilha, agora com os votos do classificador. Isso dá as
categorias que o COCO não separa (van de trabalho, pickup, caminhão articulado, caminhão simples).

```bash
python treinar.py preparar-cls --tar MIO-TCD-Classification.tar --saida dados/mio_tcd_cls --por-classe 3000
python treinar.py treinar-cls --dados dados/mio_tcd_cls --epocas 20 --imgsz 128
python contar.py --video meu_video.mp4 --classificador treinos/mio_tcd_cls/weights/best.pt
```

`preparar-cls` lê o `.tar` em fluxo, sem extrair tudo, e limita cada classe a `--por-classe` imagens: o conjunto
original é muito desbalanceado (260 mil carros contra 2 mil motos) e, sem esse corte, o classificador chamaria
tudo de carro. Recortes menores que `classificador.min_px` ou com confiança abaixo de `classificador.conf_min`
mantêm a classe do detector; `background` e `pedestrian` não geram voto.

Resultado do treino (30 mil recortes balanceados, YOLO11s-cls, 128 px, 20 épocas): **acurácia top-1 de 90,5%**
na validação do MIO-TCD. No vídeo de rodovia em 4K, porém, o classificador livre errou 3 de 10 veículos (dois
carros viraram ônibus e um hatch virou van): os recortes do MIO-TCD têm baixa resolução e outro ângulo de câmera,
então há diferença de domínio. Por isso o refino é **hierárquico** (`classificador.compativeis`): o classificador
só escolhe entre as categorias compatíveis com a classe do detector (carro → carro/van, caminhão →
caminhão/van/ônibus…), somando as probabilidades das classes MIO-TCD de cada categoria. Com essa restrição,
as 10 categorias da amostra ficaram corretas. Compare as duas configurações no vídeo da equipe com
`--classificador nenhum` e `avaliar.py contagem`.

### Treinar o detector no MIO-TCD Localization (alternativa)

```bash
python treinar.py preparar --mio-tcd C:\dados\MIO-TCD-Localization --saida dados/mio_tcd_yolo --amostra 40000
python treinar.py treinar --dados dados/mio_tcd_yolo/dados.yaml --base yolo11s.pt --epocas 30 --imgsz 640 --lote 16
```

`preparar` lê `gt_train.csv`, converte as caixas para o formato YOLO e separa 10% para validação (as imagens são
ligadas por hard link, sem duplicar espaço). `--amostra` limita o volume para caber no tempo do trabalho. O treino
parte do YOLO pré-treinado no COCO e as 11 classes do MIO-TCD são mapeadas em `categorias.mapa` do config
(`pickup_truck`→carro, `articulated_truck`/`single_unit_truck`→caminhão, `work_van`→van; `pedestrian` é ignorado).

### Avaliar no GRAM-RTM (somente teste)

1. Defina a linha da sequência num config próprio: `copy config.yaml config_m30.yaml` e
   `python definir_linha.py --video GRAM-RTM\M-30 --config config_m30.yaml --gravar`.
2. Rode o sistema na pasta de quadros:
   `python contar.py --video GRAM-RTM\M-30 --fps 30 --config config_m30.yaml --saida resultados_m30 --sem-video`
3. Gere a referência a partir das trajetórias anotadas (mesma linha, mesma lógica) e compare:
   `python avaliar.py gram-contagem --sistema resultados_m30/eventos.csv --anotacoes GRAM-RTM\Annotations\M-30 --quadros GRAM-RTM\M-30 --fps 30 --config config_m30.yaml --saida aval_m30`
4. Precisão do **detector** (separada da contagem):
   `python avaliar.py detector --quadros GRAM-RTM\M-30 --anotacoes GRAM-RTM\Annotations\M-30 --config config_m30.yaml --saida aval_det_m30`

O leitor de XML aceita um arquivo por quadro (estilo PASCAL VOC com `ID` no objeto), um arquivo único com
`<frame>` e o formato de trilhas do CVAT; o comando imprime quantos objetos, trilhas e classes encontrou para
conferência. Ajuste o `--fps` à sequência usada.

## Armadilhas conhecidas e como o sistema trata

| Armadilha | Tratamento |
|---|---|
| Veículo parado sobre a linha contado várias vezes | zona morta de `margem_px` em torno da linha (o lado só muda fora dela) e **uma única contagem por ID** |
| Troca de ID logo após cruzar (oclusão) | novo cruzamento no mesmo sentido, com IoU ≥ `dedup_iou` com uma trilha já contada que sumiu há ≤ `dedup_janela_s`, é descartado e os votos de categoria são fundidos |
| Moto ao lado de carro suprimida pelo NMS | NMS **por classe** (`agnostic_nms: false`) e IoU de NMS 0,6; a moto só compete com outras motos |
| Oclusão por caminhão | ByteTrack associa também detecções de baixa confiança e mantém trilhas perdidas por 60 quadros (`track_buffer`); o cruzamento compara o último lado estável com o atual, então uma lacuna na trilha não perde a contagem |
| Categoria oscilando quadro a quadro | categoria final = votação ponderada pela confiança ao longo de toda a trilha |
| Veículo em outra via cruzando o prolongamento da linha | o ponto de cruzamento precisa cair dentro do segmento desenhado |
| Ponto de referência | base central da caixa (contato com o solo), menos afetado pela perspectiva que o centro |
| Precisão do detector ≠ precisão da contagem | avaliadas separadamente: `avaliar.py detector` (AP50, P, R por quadro) e `avaliar.py contagem`/`gram-contagem` (erro %, eventos pareados). **A segunda é a que importa** — um detector com AP médio pode contar bem se a linha estiver num ponto favorável, e um detector ótimo conta mal se o rastreamento troca IDs na linha |

## Alertas de fluxo anormal

O fluxo é agregado em janelas de `alerta.janela_s` segundos. O valor esperado é a linha de base informada
(`--linha-base` em veíc/h, por exemplo a taxa do mesmo horário em outro dia, disponível em `resumo.json`) ou,
na falta dela, a média adaptativa das janelas normais anteriores. O desvio usa o maior entre o desvio observado
e o de Poisson (√esperado), o que evita alarmes em volumes baixos.

| Alerta | Condição | Recomendação gerada |
|---|---|---|
| `FLUXO_ALTO` | ≥ 1,5× o esperado e z ≥ 2,5 | estender verde no sentido dominante, onda verde, agente de trânsito |
| `FLUXO_BAIXO` | ≤ 0,4× o esperado e z ≤ −2,5 | possível bloqueio/acidente a montante: conferir imagens e despachar equipe |
| `SEM_FLUXO` | 2 janelas seguidas sem veículo | verificar câmera obstruída ou interdição; não usar os dados |
| `CONGESTIONAMENTO` | ≥ 3 veículos parados há ≥ 20 s perto da linha | fila/retenção: verificar semáforo a jusante, ampliar verde |
| `PESADOS` | ≥ 35% de caminhões/ônibus numa janela com ≥ 8 veículos | restrição de carga no horário, fiscalização de peso |

O relatório também gera recomendações de planejamento a partir do período todo: divisão de tempo de verde
proporcional ao fluxo equivalente (UCP) de cada sentido, faixa de pico, participação de pesados, motos e bicicletas.

## Testes

```bash
python -m unittest discover -s testes -v
```

Cobrem: cruzamento nos dois sentidos, veículo oscilando sobre a linha (conta 1), jitter dentro da margem (conta 0),
moto ao lado de carro (conta 2), troca de ID (conta 1), cruzamento fora do segmento, votação de categoria,
fila parada, os cinco tipos de alerta, pareamento e causas na avaliação e o AP50.

## Resultados (preencher com os números da equipe)

| Fonte | Referência | Sistema | Erro % total | Acurácia de categoria | Principais causas de erro |
|---|---|---|---|---|---|
| Vídeo local (3 min) | | | | | |
| GRAM-RTM M-30 | | | | | |
| GRAM-RTM Urban1 | | | | | |

| Detector | mAP50 GRAM-RTM | Erro % de contagem GRAM-RTM | Erro % de contagem vídeo local |
|---|---|---|---|
| YOLO11s COCO | | | |
| YOLO11s COCO + classificador MIO-TCD | | | |

## Limitações

- A câmera precisa ser fixa; se ela se mover, a linha deixa de corresponder à via.
- Veículos estacionados perto da linha podem disparar `CONGESTIONAMENTO`; ajuste `distancia_fila_px` ou a posição da linha.
- O COCO não distingue van de carro nem os tipos de caminhão; o ajuste no MIO-TCD resolve parte disso.
- Os fatores de UCP do config são indicativos; para projeto semafórico use os do manual adotado pelo órgão de trânsito.
- À noite, com chuva ou contra o sol a revocação cai; a análise de causas aponta esses trechos.
