# Guia de coleta e anotação

O modelo só fica bom se as imagens de treino forem parecidas com o que a câmera da loja vai ver.
Este guia cobre a parte manual do trabalho: gravar, extrair as imagens, anotar e exportar.

## 1. O que gravar

Meta: **400 a 600 imagens anotadas**, somando todas as classes. Isso costuma sair de 15 a 25 vídeos curtos.

| Classe | O que é | Onde gravar |
|---|---|---|
| `geladeira_aberta` | Compartimento de refrigeração (geladeira ou expositor vertical) com a porta aberta ou entreaberta | Em casa e no comércio |
| `geladeira_fechada` | O mesmo compartimento com a porta totalmente fechada | Em casa e no comércio |
| `freezer_aberto` | Compartimento de congelados aberto: gaveta ou porta do freezer de uma geladeira, ou freezer horizontal com a tampa aberta ou sem tampa | Em casa e no comércio |
| `freezer_fechado` | O mesmo compartimento fechado | Em casa e no comércio |

Uma geladeira com freezer (em cima ou embaixo) conta como **dois compartimentos**: a parte da geladeira
e a parte do freezer. Cada um tem a sua caixa e o seu estado.

Como gravar cada vídeo (20 a 40 segundos):

1. Apoie o celular ou segure-o parado, de preferência no alto, como uma câmera de segurança.
2. Comece com a porta fechada por 5 segundos.
3. Abra a porta devagar, deixe aberta por 10 segundos, deixe entreaberta por 5 segundos e feche.
4. Repita de outro ângulo e de outra distância.

Varie o máximo possível:

- **Equipamentos:** pelo menos 5 geladeiras ou expositores diferentes e 2 ou 3 freezers (de geladeira
  ou horizontais).
- **Ângulo:** de frente, de lado, do alto.
- **Distância:** de 1,5 a 5 metros.
- **Luz:** dia, noite com luz acesa, luz apagada com a luz interna da geladeira.
- **Cena:** com e sem pessoas passando, com produtos na frente.

No comércio:

- Peça autorização ao dono antes de gravar e explique que é um trabalho da faculdade.
- Evite filmar rostos de clientes e funcionários.
- Dois ou três comércios diferentes já ajudam bastante.

**Reserve dois ou três vídeos que nunca entram no treino.** Eles servem para o vídeo de demonstração
e para mostrar que o modelo funciona em cenas que ele nunca viu. Guarde-os em `data/videos/demo/` e
não extraia imagens deles.

## 2. Extrair as imagens

Coloque os vídeos de treino em `data/videos/` e rode:

```bash
python src/extrair_frames.py --entrada data/videos --saida data/frames --borrar-rostos
```

- Salva uma imagem a cada 0,5 segundo e pula as quase iguais à anterior.
- `--borrar-rostos` encontra as pessoas na imagem e borra o rosto, de frente ou de perfil.
  Não é perfeito: confira as imagens e apague as que ainda mostrarem alguém identificável. Na primeira vez,
  ele baixa sozinho o modelo `yolo11n-pose.pt` (cerca de 6 MB).
- Os vídeos da pasta `data/videos/demo/` são ignorados automaticamente.
- Se saírem imagens demais de um mesmo vídeo, aumente o intervalo: `--intervalo 1`.

## 3. Anotar no Roboflow

1. Crie uma conta gratuita em https://roboflow.com e um projeto do tipo **Object Detection**.
2. Crie as quatro classes com **exatamente** estes nomes: `geladeira_aberta`, `geladeira_fechada`,
   `freezer_aberto`, `freezer_fechado`. Os nomes precisam ser iguais aos do `config.yaml`.
3. Envie as imagens de `data/frames/`.
4. Anote cada imagem seguindo as regras abaixo.

### Regras de anotação

- **A caixa cobre o compartimento inteiro**, não só a porta. Com a porta aberta, a caixa cobre o corpo do
  compartimento e a porta (ou a gaveta puxada). A aplicação reconhece que é o mesmo equipamento pela
  posição da caixa, então a caixa "aberta" e a caixa "fechada" de um mesmo compartimento precisam se sobrepor.
- **Geladeira com freezer: duas caixas por imagem.** Uma caixa `geladeira_...` na parte da geladeira e
  uma caixa `freezer_...` na parte do freezer, cada uma com o seu estado. Exemplo: portas de cima abertas
  e gaveta de baixo fechada dá `geladeira_aberta` em cima e `freezer_fechado` embaixo.
- **As duas caixas não se misturam.** A caixa da geladeira não desce para a área do freezer, e a do
  freezer não sobe para a área da geladeira, mesmo com as portas abertas.
- **Porta entreaberta conta como aberta.** Qualquer fresta visível é `aberta`.
- **Duas portas no mesmo compartimento** (duas folhas lado a lado): uma caixa só. Se qualquer uma estiver
  aberta, é `aberta`.
- **Imagens tremidas ou com a câmera sendo movida:** apague. Elas aparecem no começo e no fim dos vídeos.
- **Equipamento parcialmente tapado:** anote se der para ver pelo menos metade.
- **Não dá para saber se está aberta ou fechada:** não anote, ou apague a imagem.
- **Imagens sem nenhum equipamento:** deixe algumas (cerca de 5% do total) sem anotação. Elas ensinam o
  modelo a não ver geladeira onde não tem.
- Mantenha a caixa justa, sem sobrar muito fundo.

### Gerar a versão do conjunto

Em **Generate**:

- **Divisão:** 70% treino, 20% validação, 10% teste.
- **Pré-processamento:** Auto-Orient e Resize para 640×640 (Fit within).
- **Aumento de dados (só no treino):** brilho de −20% a +20%, exposição de −10% a +10%, rotação de −5° a +5°,
  desfoque de até 1 px. Não use espelhamento vertical.
- Gere 2 ou 3 versões de cada imagem de treino.

### Exportar

Em **Export Dataset**, escolha o formato **YOLOv11** e baixe o arquivo .zip. Descompacte em `data/dataset/`,
de modo que exista o arquivo `data/dataset/data.yaml`.

## 4. Treinar e conferir

Rode o notebook `notebooks/treino.ipynb`. Depois do primeiro treino:

1. Olhe a matriz de confusão: quais classes o modelo confunde?
2. Rode a aplicação em um vídeo de demonstração e anote onde ela erra.
3. Grave mais vídeos **desses casos** (por exemplo, porta entreaberta ou reflexo no vidro), anote e
   treine a versão 2.

A comparação entre a versão 1 e a versão 2 entra na metodologia do relatório.

## 5. Cuidado com a avaliação

O Roboflow divide as imagens ao acaso. Como imagens vizinhas do mesmo vídeo são muito parecidas, parte
delas cai no treino e parte no teste, e as métricas ficam mais altas do que o desempenho real. Por isso:

- diga isso no relatório;
- mostre também o resultado nos vídeos reservados, que o modelo nunca viu.

## 6. Amostras para o repositório

Copie de 10 a 20 imagens variadas, sem pessoas identificáveis, para `data/amostras/`. Os vídeos e o
conjunto completo não vão para o repositório.
