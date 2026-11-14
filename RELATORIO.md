# Contagem e classificação de veículos por câmera fixa — Relatório final

**Tema 26 · Visão Computacional** · Aulas relacionadas: 7 (detecção e NMS) e 9 (rastreamento, identidade,
implantação e monitoramento)

## 1. Problema e decisão-alvo

O planejamento viário depende de saber quantos veículos passam por um ponto, de que tipo e em que horário.
Os contadores por laço indutivo exigem obra no pavimento, custam caro por faixa e não distinguem a categoria do
veículo. Uma câmera fixa em poste, que muitas vezes já existe para monitoramento, resolve os dois problemas
se o vídeo for convertido automaticamente em contagens confiáveis.

O sistema recebe o vídeo de uma câmera fixa sobre a via e produz duas saídas:

1. **Contagem por categoria e sentido**, agregada em faixas horárias, com volume em veículos/hora e em unidades
   de carro de passeio (UCP/h).
2. **Alerta de fluxo anormal** em tempo de operação, com recomendação de ação.

**Quem decide e o quê.** O usuário é a engenharia de tráfego do órgão municipal. O sistema apoia dois tipos
de decisão:

| Horizonte | Decisão | Saída que a sustenta |
|---|---|---|
| Planejamento (dias a meses) | divisão do tempo de verde entre aproximações, horário de pico, restrição de circulação de carga, estudo de motobox ou ciclofaixa | relatório por faixa horária, UCP por sentido, participação de pesados, motos e bicicletas |
| Operação (segundos a minutos) | estender verde, despachar equipe, verificar câmera, divulgar rota alternativa | alertas `FLUXO_ALTO`, `FLUXO_BAIXO`, `SEM_FLUXO`, `CONGESTIONAMENTO`, `PESADOS` |

**Critério de aceite.** O erro que importa é o **da contagem**, não o do detector. A equipe adotou como meta
erro absoluto de contagem total de até 5% contra a referência manual e acurácia de categoria de pelo menos 85%
nos veículos pareados, com a análise das causas de cada contagem dupla ou perdida.

## 2. Dados

| Base | Papel | Anotação | Licença / condição de uso |
|---|---|---|---|
| MIO-TCD Classification | **treino** do classificador de categoria | rótulo por recorte | CC BY-NC-SA 4.0 |
| MIO-TCD Localization | treino do detector alternativo | caixas por quadro, sem identidade | CC BY-NC-SA 4.0 |
| GRAM-RTM | **somente teste** da contagem e do detector | caixas por quadro com ID de veículo | uso científico; proibido treinar ou ajustar parâmetros |
| Conjunto local | **teste principal** | contagem manual da equipe, evento a evento | dados próprios |
| COCO | pré-treino do detector (pesos `yolo11s.pt`) | — | pesos AGPL-3.0 (Ultralytics) |

Nenhuma imagem de teste foi vista em treino. O MIO-TCD treina; o GRAM-RTM e o conjunto local avaliam. Como
as bases vêm de países, câmeras e anos diferentes, a avaliação mede generalização para outro domínio.

### 2.1 MIO-TCD

Fonte: https://tcd.miovision.com/challenge/dataset.html. São 786.702 imagens de milhares de câmeras de tráfego
no Canadá e nos EUA, em várias condições de luz e clima, com câmeras em poste e vista oblíqua, o mesmo cenário
deste projeto. Tem 11 classes, mapeadas para as categorias do sistema em `categorias.mapa` do `config.yaml`:

| Classe MIO-TCD | Categoria | Classe MIO-TCD | Categoria |
|---|---|---|---|
| `car`, `pickup_truck` | carro | `motorcycle` | moto |
| `work_van` | van | `bicycle` | bicicleta |
| `bus` | ônibus | `non-motorized_vehicle`, `motorized_vehicle` | outro |
| `articulated_truck`, `single_unit_truck` | caminhão | `pedestrian`, `background` | ignorados |

**Classification.** A divisão de treino rotulada tem 519.164 recortes, extremamente desbalanceados: 50,2% são
carros, 30,8% fundo e apenas 0,4% motos (131 carros por moto). Um classificador treinado nessa distribuição
acerta muito chamando tudo de carro, justamente nas classes que mais interessam ao planejamento. Por isso o
subconjunto usado limita cada classe a 3.000 recortes. As três classes menores entram inteiras. A divisão é
85/15 com semente 42, somando 25.485 imagens de treino e 4.532 de validação. O menor lado dos recortes tem
mediana de 84 px.

**Localization.** São 110.000 quadros de treino com 351.549 caixas, a maioria em 720×480. O subconjunto tem
20.000 quadros e prioriza as imagens que contêm classes com menos de 15.000 caixas. A divisão é 90/10 com
semente 42. Com isso, a fração de caixas que não são carro sobe de 34% para 46%, e a de motos de 0,5% para
1,2%. A divisão é feita por imagem: quadros da mesma câmera podem cair em treino e validação, o que torna a
validação otimista.

**Limitações.** A frota é norte-americana, com picapes grandes e poucas motos, ao contrário do trânsito
brasileiro. Os recortes têm baixa resolução. A classe `motorized_vehicle` é ambígua. Os quadros são isolados e
não têm identidade, então não servem para avaliar contagem.

### 2.2 GRAM-RTM

Fonte: https://gram.web.uah.es/data/datasets/rtm/index.html (Universidad de Alcalá). São três sequências de
vídeo contínuo, num total de 40.345 quadros, anotadas manualmente em XML compatível com PASCAL VOC e com
**identificador único por veículo**:

| Sequência | Quadros | Resolução | FPS | Veículos | Cenário |
|---|---:|---|---:|---:|---|
| M-30 | 7.520 | 800×480 | 30 | 256 | rodovia urbana, dia ensolarado |
| M-30-HD | 9.390 | 1200×720 | 30 | 235 | mesmo local, dia nublado |
| Urban1 | 23.435 | 600×360 | 25 | 237 | cruzamento movimentado |

As classes são `car`, `truck`, `van` e `big-truck`; não há motos nem ônibus anotados. Os autores determinam
que a base seja usada **somente para teste** e que o resultado de teste **não seja usado para ajustar
parâmetros**. O projeto cumpre as duas regras: nenhum quadro entra em treino, e a única configuração por
sequência é a posição da linha, escolhida pela geometria da via antes de rodar a avaliação.

### 2.3 Bases descartadas

O **VisDrone2019-DET** tem vista de drone, objetos pequenos e geometria diferente da câmera em poste. O
**UA-DETRAC** teve o site oficial desativado e só existe em espelhos sem licença clara.

### 2.4 Conjunto local

Uma via próxima gravada de ponto elevado por 3 minutos, com contagem manual feita pela equipe usando
`contagem_manual.py`. Cada tecla registra o instante, o sentido e a categoria do veículo. Isso permite parear
evento a evento com o sistema, em vez de comparar só totais. Duas pessoas contam de forma independente e as
divergências são revisadas; o arquivo final é a referência.

## 3. Sistema

```
quadro ─► YOLO11s (NMS por classe) ─► ByteTrack (ID estável) ─► classificador YOLO11s-cls (MIO-TCD)
                                                                         │
                         votação de categoria ao longo da trilha ◄───────┘
                                         │
                    linha virtual: lado com zona morta, cruzamento dentro do segmento,
                    1 contagem por ID, anti-duplicata por troca de ID, detecção de parados
                                         │
                       eventos de cruzamento (id, instante, sentido, categoria)
                                         │
               ┌─────────────────────────┼──────────────────────────┐
               ▼                         ▼                          ▼
     relatório por faixa horária   monitor de fluxo (janelas)   avaliação contra referência
     veíc/h, UCP/h, gráficos       alertas + recomendação        erro %, acurácia, causas
```

### 3.1 Modelo profundo

**Detector.** O detector é o YOLO11s (9,4 M parâmetros) pré-treinado no COCO. É uma rede convolucional de um
estágio com backbone de blocos C3k2, pooling piramidal (SPPF), atenção espacial (C2PSA) e cabeça
desacoplada sem âncoras. Usa entrada de 960 px, porque a câmera de poste gera veículos pequenos no fundo da
cena, e confiança mínima de 0,1 para alimentar o rastreador. O **NMS é por classe** (`agnostic_nms: false`)
com IoU de 0,6: uma moto ao lado de um carro só compete com outras motos e não é suprimida pela caixa maior.

**Rastreador.** O ByteTrack associa detecções entre quadros com filtro de Kalman e associação húngara em duas
etapas. Primeiro associa as detecções de alta confiança e depois tenta as de baixa confiança contra as
trilhas que sobraram. É isso que mantém a identidade de um veículo parcialmente oculto por um caminhão. As
trilhas perdidas ficam guardadas por 60 quadros (`track_buffer`).

**Classificador de categoria.** O COCO não separa van de carro nem os tipos de caminhão. Um YOLO11s-cls
(5,5 M parâmetros) ajustado no MIO-TCD Classification reclassifica, a cada quadro, o recorte de cada veículo
rastreado. Foi treinado por 20 épocas em 128 px com lote 128, a partir dos pesos do ImageNet. A categoria
final de cada veículo é a **votação ponderada pela confiança ao longo da trilha inteira**, o que elimina a
oscilação quadro a quadro.

O refino é **hierárquico** (`classificador.compativeis`). O classificador só escolhe entre as categorias
compatíveis com a classe do detector: carro → {carro, van}, caminhão → {caminhão, van, ônibus}, ônibus →
{ônibus, caminhão}. As probabilidades das classes MIO-TCD de cada categoria são somadas. Recortes com menos de
24 px de lado ou com confiança abaixo de 0,6 mantêm a classe do detector. A restrição existe por causa da
diferença de domínio: numa verificação qualitativa com 10 veículos em resolução 4K, o classificador sem
restrição errou 3 (dois carros viraram ônibus e um hatch virou van) e a versão hierárquica acertou os 10.

**Detector alternativo.** O YOLO11s foi ajustado no MIO-TCD Localization, em 640 px com lote 16. Fica
disponível com `--pesos treinos/mio_tcd_det/weights/best.pt` para comparação.

### 3.2 Lógica de contagem

| Regra | Motivo |
|---|---|
| Ponto de referência na base central da caixa | é o contato com o solo, menos distorcido pela perspectiva do que o centro |
| Lado da linha pela distância assinada, com **zona morta** de ±8 px | um veículo parado ou oscilando sobre a linha não muda de lado |
| Cruzamento conta só se o ponto de interseção cair **dentro do segmento** desenhado | veículos de outra via não cruzam o prolongamento da linha |
| Cruzamento compara o **último lado estável** com o atual | uma lacuna na trilha (oclusão) não perde a contagem |
| **Uma contagem por ID** e mínimo de 2 observações | elimina recontagem e trilhas espúrias de um quadro |
| **Anti-duplicata**: um novo cruzamento no mesmo sentido com IoU ≥ 0,3 em relação a uma trilha já contada que sumiu há até 1 s é descartado, e os votos de categoria são fundidos | troca de ID logo após a linha não gera contagem dupla |
| Veículo **parado**: velocidade normalizada pela altura da caixa abaixo de 0,15 por segundo | alimenta o alerta de congestionamento |

### 3.3 Camada de decisão

**Alertas em operação.** O fluxo é agregado em janelas de 30 s. O valor esperado é a linha de base informada
(`--linha-base`, por exemplo a taxa do mesmo horário em outro dia) ou, na falta dela, uma média móvel
exponencial (α = 0,3) das janelas normais anteriores. Janelas anômalas não atualizam a base, para que um pico
não se torne o novo normal. O desvio usado no z-score é o maior entre o desvio observado e o de Poisson
(√esperado), o que evita alarmes em volumes baixos.

| Alerta | Condição | Severidade | Recomendação |
|---|---|---|---|
| `FLUXO_ALTO` | ≥ 1,5× o esperado e z ≥ 2,5 | média | estender o verde no sentido dominante, onda verde, agente de trânsito |
| `FLUXO_BAIXO` | ≤ 0,4× o esperado e z ≤ −2,5 | alta | possível bloqueio a montante: conferir imagens, despachar equipe, rota alternativa |
| `SEM_FLUXO` | 2 janelas seguidas sem veículo | alta | verificar câmera obstruída ou interdição; não usar os dados até confirmar |
| `CONGESTIONAMENTO` | ≥ 3 veículos parados há ≥ 20 s perto da linha | alta | verificar o semáforo a jusante, ampliar o verde da aproximação |
| `PESADOS` | ≥ 35% de caminhões e ônibus numa janela com ≥ 8 veículos | baixa | restrição de carga no horário, fiscalização de peso |

**Recomendações de planejamento.** Ao fim do período, o relatório gera recomendações:

- divisão do tempo de verde proporcional ao fluxo em UCP de cada sentido;
- sugestão de onda verde quando um sentido passa de 65% do fluxo;
- faixa horária de pico;
- recomendações por participação de pesados (≥ 15%), motos (≥ 25%) e bicicletas (≥ 5%).

Os fatores de UCP (moto 0,33; van 1,5; ônibus 2,0; caminhão 2,5) são indicativos. Para projeto semafórico,
devem ser substituídos pelos do manual adotado pelo órgão.

**Papel humano.** O sistema **recomenda**; ele não aciona semáforos. Cada alerta traz o observado, o esperado
e a regra disparada, para que o operador possa conferir antes de agir.

### 3.4 Armadilhas conhecidas e tratamento

| Armadilha | Tratamento |
|---|---|
| Veículo parado sobre a linha contado várias vezes | zona morta e uma contagem por ID |
| Moto ao lado de carro suprimida pelo NMS | NMS por classe com IoU 0,6 |
| Oclusão por caminhão | associação de baixa confiança do ByteTrack, buffer de 60 quadros, comparação com o último lado estável, anti-duplicata |
| Troca de ID após cruzar | anti-duplicata por IoU e janela de tempo, com fusão de votos |
| Precisão do detector ≠ precisão da contagem | avaliadas separadamente (seção 4); a da contagem é a que decide |

## 4. Protocolo de avaliação

A avaliação tem dois níveis, porque um detector com AP médio pode contar bem se a linha estiver num ponto
favorável, e um detector ótimo conta mal se o rastreador trocar IDs justamente na linha.

**Nível 1 — contagem (o que importa).** Os eventos do sistema e da referência são pareados de forma gulosa
pelo menor custo. O par precisa estar no mesmo sentido e com diferença de tempo dentro da tolerância (2 s no
conjunto local, 1 s no GRAM-RTM). Uma penalidade de categoria favorece parear veículos da mesma classe. As
métricas são:

- **erro percentual de contagem** = (sistema − referência) / referência, no total, por sentido e por categoria;
- **precisão e revocação da contagem**: eventos pareados sobre o total do sistema e sobre o total da referência;
- **acurácia de categoria** e matriz de confusão sobre os eventos pareados;
- **causa provável** de cada evento não pareado:
  - `contagem_dupla`: outro evento do sistema próximo, no mesmo sentido;
  - `falso_positivo`;
  - `sentido_invertido`;
  - `veiculos_simultaneos`: outro veículo da referência quase no mesmo instante, o que indica NMS ou oclusão;
  - `objeto_pequeno`: moto ou bicicleta;
  - `nao_detectado_ou_trilha_perdida`;
  - `categoria_errada`.

O `erros.csv` traz o instante de cada erro para conferência visual.

No GRAM-RTM, a referência de contagem é gerada aplicando **a mesma linha e a mesma lógica de cruzamento** às
trilhas anotadas, com o ID verdadeiro, sem filtro de observações mínimas e sem anti-duplicata. Assim, a
diferença medida vem só da detecção e do rastreamento.

**Nível 2 — detector.** Mede AP50 no padrão VOC, precisão e revocação na confiança de operação (0,3), por
categoria, a cada 10 quadros do GRAM-RTM.

**Latência.** A latência é medida por etapa em cada quadro, descartando os 10 primeiros quadros (aquecimento
da GPU). O tempo de **processamento** soma detecção e rastreamento, classificação e contagem com decisão. Ele
é comparado ao orçamento de 1/fps do vídeo. O sistema é considerado de tempo real se o p95 do processamento
couber no orçamento.

## 5. Resultados

### 5.1 Modelos no domínio de treino (validação MIO-TCD)

| Modelo | Validação | Resultado |
|---|---|---|
| Classificador YOLO11s-cls, 128 px | 4.532 recortes balanceados | top-1 **90,6%**, top-5 99,6% |
| Detector YOLO11s ajustado, 640 px (época 16 de 25) | 2.000 quadros | mAP50 **0,83**, mAP50-95 0,61, P 0,80, R 0,77 |

Esses números medem o modelo **dentro** do domínio MIO-TCD e servem para comparar configurações. Não são a
métrica de aceite.

### 5.2 Contagem

| Fonte | Referência | Sistema | Erro % total | Precisão / revocação da contagem | Acurácia de categoria | Principais causas de erro |
|---|---:|---:|---:|---|---:|---|
| Conjunto local | — | — | — | — | — | — |
| GRAM-RTM M-30 | — | — | — | — | — | — |
| GRAM-RTM Urban1 | — | — | — | — | — | — |

| Configuração | AP50 GRAM-RTM | Erro % contagem GRAM-RTM | Erro % contagem local |
|---|---:|---:|---:|
| YOLO11s COCO | — | — | — |
| YOLO11s COCO + classificador MIO-TCD | — | — | — |
| YOLO11s ajustado no MIO-TCD | — | — | — |

### 5.3 Latência

Medida em uma NVIDIA GeForce RTX 3060 Ti, com entrada de 3840×2160 a 25 fps (orçamento de 40 ms por quadro),
detector em 960 px e classificador ativo:

| Etapa | Média (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| Decodificação do quadro 4K | 16,6 | 20,3 | 23,0 |
| Detecção + rastreamento | 17,1 | 21,3 | 24,3 |
| Classificação de categoria | 8,9 | 11,9 | 27,7 |
| Contagem e decisão | 0,05 | 0,07 | 0,08 |
| **Processamento** (sem decodificação) | **26,0** | **32,6** | 46,9 |
| Total por quadro | 42,5 | 51,4 | 64,7 |

O processamento cabe no orçamento com folga no p95 (32,6 ms contra 40 ms). A contagem e a camada de decisão
custam menos de 0,1 ms; o custo está todo na rede neural. Sem o classificador, o processamento cai para
16,3 ms (p95 de 18,9 ms). O classificador custa cerca de 9 ms por quadro, em troca das categorias que o COCO
não separa.

Com a decodificação de vídeo 4K, o total médio (42,5 ms) passa do orçamento, e o sistema sustenta cerca de
21 quadros/s. Em implantação, a câmera deve entregar o fluxo em 1080p, resolução em que a decodificação cai a
uma fração disso. Outra opção é decodificar em hardware (NVDEC).

A **latência de decisão** é diferente da latência por quadro. Os alertas de fluxo são emitidos no fechamento
de cada janela, então um evento anormal é sinalizado em até 30 s mais o tempo de processamento. O
congestionamento exige veículos parados por 20 s. Esses tempos são compatíveis com uma resposta operacional,
que leva minutos. Encurtar a janela antecipa o alerta, mas aumenta a variância e os falsos alarmes.

## 6. Ética, privacidade e uso responsável

- **Finalidade e minimização (LGPD, Lei 13.709/2018).** A saída do sistema são eventos anônimos: ID de trilha
  local, instante, sentido e categoria. Não há reconhecimento de placa nem de rosto, e o ID não sobrevive
  entre sessões nem entre câmeras, o que impede rastrear uma pessoa pela cidade. O vídeo anotado é opcional
  (`--sem-video`). Em operação, deve ficar desligado e o vídeo bruto deve ter retenção curta, definida pelo
  órgão.
- **Pedestres** ficam fora do sistema: a classe `person` é excluída da detecção e a classe `pedestrian` do
  classificador não gera voto, então pessoas não são contadas nem registradas.
- **Transparência.** A câmera deve ser sinalizada no local, com a finalidade de contagem de tráfego.
- **Viés e equidade.** O treino vem de frota norte-americana, com poucas motos e bicicletas. Se o sistema
  subcontar motos ou bicicletas, as decisões de planejamento tendem a desfavorecer exatamente os usuários mais
  vulneráveis da via. Por isso a acurácia **por categoria** é métrica obrigatória, e não só a acurácia global.
  Os fatores de UCP também escolhem quem ganha tempo de verde e precisam vir do manual técnico do órgão.
- **Humano no circuito.** Os alertas recomendam e não atuam. O alerta `SEM_FLUXO` existe para impedir que uma
  câmera obstruída gere dados falsos de "via vazia" que acabem usados no planejamento.
- **Degradação conhecida.** À noite, com chuva ou contra o sol, a revocação cai. Dados desses períodos devem
  ser sinalizados antes de entrar em decisões.
- **Licenças.** O MIO-TCD é CC BY-NC-SA 4.0: os pesos treinados herdam a cláusula não comercial e só podem ser
  usados neste trabalho acadêmico. O GRAM-RTM foi usado apenas como teste, conforme exigido pelos autores.

## 7. Limitações e trabalhos futuros

- A câmera precisa ser fixa; se ela se mover, a linha deixa de corresponder à via.
- Veículos estacionados perto da linha podem disparar `CONGESTIONAMENTO`. Ajuste `distancia_fila_px` ou a
  posição da linha.
- A diferença de domínio do MIO-TCD é a principal fonte de erro de categoria. O próximo passo é anotar
  recortes locais, sobretudo motos e ônibus urbanos, e ajustar o classificador com eles.
- O GRAM-RTM não tem motos nem ônibus, então a acurácia dessas categorias só é medida no conjunto local.
- Os limiares dos alertas foram definidos por critério de engenharia. Calibrá-los exige histórico de vários
  dias no mesmo ponto.

## 8. Referências

1. Z. Luo, F. B. Charron, C. Lemaire, J. Konrad, S. Li, A. Mishra, A. Achkar, J. Eichel, P.-M. Jodoin.
   "MIO-TCD: A new benchmark dataset for vehicle classification and localization." *IEEE Transactions on Image
   Processing*, 27(10):5129–5141, 2018. DOI: 10.1109/TIP.2018.2848705.
2. R. Guerrero-Gómez-Olmedo, R. J. López-Sastre, S. Maldonado-Bascón, A. Fernández-Caballero. "Vehicle
   Tracking by Simultaneous Detection and Viewpoint Estimation." *IWINAC 2013*, Part II, LNCS 7931,
   p. 306–316, 2013.
3. Y. Zhang et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." *ECCV*, 2022.
4. G. Jocher, J. Qiu. *Ultralytics YOLO11*, 2024. https://github.com/ultralytics/ultralytics
5. T.-Y. Lin et al. "Microsoft COCO: Common Objects in Context." *ECCV*, 2014.
6. Brasil. Lei nº 13.709, de 14 de agosto de 2018 — Lei Geral de Proteção de Dados Pessoais (LGPD).

## Apêndice — Atendimento aos requisitos

| Requisito do tema | Onde está atendido |
|---|---|
| Detectar e rastrear veículos com identidade estável | `contagem/detector.py` (YOLO11s + ByteTrack), `rastreador.yaml` |
| Classificar por categoria | classificador MIO-TCD e votação na trilha: `contagem/detector.py`, `contagem/contador.py` |
| Contar cruzamentos de linha virtual por sentido e período | `contagem/contador.py`, `definir_linha.py` |
| Evitar contar o mesmo veículo duas vezes | zona morta, uma contagem por ID, anti-duplicata: `contagem/contador.py` |
| Saída: contagem por categoria e sentido com relatório por faixa horária | `contagem/relatorio.py` → `contagem_por_faixa.csv`, `relatorio.md` |
| Saída: alerta de fluxo anormal | `contagem/alertas.py` → `alertas.csv` |
| Treinar no MIO-TCD | `treinar.py` (classificador e detector) |
| Avaliar a contagem no GRAM-RTM, só como teste | `avaliar.py gram-contagem`, `avaliar.py detector`, `contagem/gram.py` |
| Contagem manual de referência local | `contagem_manual.py` |
| Erro percentual contra a referência, acurácia por categoria, causas de dupla ou perdida | `avaliar.py contagem`, `contagem/avaliacao.py` |
| Moto ao lado de carro suprimida pelo NMS | `agnostic_nms: false` e `iou_nms: 0.6` em `config.yaml`; teste automatizado |
| Precisão do detector separada da precisão da contagem | `avaliar.py detector` × `avaliar.py contagem` / `gram-contagem` |
| Modelo profundo | YOLO11s (detector) e YOLO11s-cls ajustado no MIO-TCD (classificador) |
| Camada de decisão | alertas com recomendação e recomendações de planejamento: `contagem/alertas.py`, `contagem/relatorio.py` |
| Latência | `contagem/latencia.py` → `latencia.csv` e seção no `relatorio.md` |
| Ética | seção 6 |
| Testes automatizados das armadilhas | `testes/test_contagem.py` (17 casos) |
