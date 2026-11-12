"""Desenho das caixas, cronômetros e alertas sobre o vídeo.

Usa a Pillow para escrever o texto, porque o OpenCV não desenha acentos.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from monitor import ABERTO, FECHADO, Monitor

VERDE = (46, 160, 67)
LARANJA = (232, 150, 0)
VERMELHO = (215, 40, 40)
CINZA = (140, 140, 140)
FUNDO = (20, 26, 34)
BRANCO = (255, 255, 255)

_FONTES = (
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
)


@lru_cache(maxsize=32)
def _fonte(tamanho: int) -> ImageFont.ImageFont:
    for caminho in _FONTES:
        try:
            return ImageFont.truetype(caminho, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)


def _fonte_que_cabe(desenho: ImageDraw.ImageDraw, texto: str, tamanho: int, largura: float) -> ImageFont.ImageFont:
    """Maior fonte, até `tamanho`, com que o texto cabe na largura dada."""
    while tamanho > 6 and desenho.textlength(texto, font=_fonte(tamanho)) > largura:
        tamanho -= 1
    return _fonte(tamanho)


def formatar_duracao(segundos: float) -> str:
    segundos = int(max(segundos, 0))
    return f"{segundos // 60:02d}:{segundos % 60:02d}"


def formatar_reais(valor: float) -> str:
    casas = 2 if valor >= 1 else 4
    return f"R$ {valor:.{casas}f}".replace(".", ",")


def _etiqueta(desenho: ImageDraw.ImageDraw, caixa, linhas: list[str], cor: tuple, tamanho: int) -> None:
    """Etiqueta de várias linhas no canto superior esquerdo, dentro da caixa do equipamento."""
    x1, y1, x2, _ = caixa
    margem = max(tamanho // 3, 3)
    mais_longa = max(linhas, key=len)
    fonte = _fonte_que_cabe(desenho, mais_longa, tamanho, max(x2 - x1 - 2 * margem, 40))
    passo = int(fonte.size * 1.25)
    largura = max(desenho.textlength(linha, font=fonte) for linha in linhas)
    desenho.rectangle([x1, y1, x1 + largura + 2 * margem, y1 + passo * len(linhas) + margem], fill=cor)
    for i, linha in enumerate(linhas):
        desenho.text((x1 + margem, y1 + margem // 2 + i * passo), linha, fill=BRANCO, font=fonte)


def desenhar(quadro_bgr: np.ndarray, monitor: Monitor, agora: float, fps: float | None = None) -> np.ndarray:
    """Devolve uma cópia do quadro com as caixas, a barra de resumo e o alerta desenhados."""
    imagem = Image.fromarray(np.ascontiguousarray(quadro_bgr[:, :, ::-1]))
    desenho = ImageDraw.Draw(imagem)
    altura, largura = quadro_bgr.shape[:2]
    tamanho = max(altura // 34, 13)
    espessura = max(altura // 240, 2)

    equipamentos = monitor.equipamentos(agora)
    em_alerta = []
    for eq in equipamentos:
        tempo = monitor.tempo_aberto(eq, agora)
        if eq.estado == ABERTO:
            alerta = monitor.em_alerta(eq, agora)
            cor = VERMELHO if alerta else LARANJA
            linhas = [eq.id, f"ABERTO há {formatar_duracao(tempo)}",
                      formatar_reais(monitor.custo.calcular(eq.tipo, tempo))]
            if alerta:
                em_alerta.append((eq, tempo))
        elif eq.estado == FECHADO:
            cor, linhas = VERDE, [eq.id, "fechado"]
        else:
            cor, linhas = CINZA, [eq.id, "verificando"]
        desenho.rectangle(list(eq.caixa), outline=cor, width=espessura)
        _etiqueta(desenho, eq.caixa, linhas, cor, tamanho)

    # Barra de resumo no topo
    abertos = sum(1 for eq in equipamentos if eq.estado == ABERTO)
    partes = [
        f"Energia que dá Lucro · {monitor.loja_id}",
        f"Equipamentos: {len(equipamentos)} · abertos: {abertos}",
        f"Desperdício estimado: {formatar_reais(monitor.custo_total_rs(agora))}",
        f"Eventos: {monitor.total_eventos}",
    ]
    if fps:
        partes.append(f"{fps:.0f} qps")
    resumo = "     ".join(partes)
    fonte = _fonte_que_cabe(desenho, resumo, tamanho, largura - 2 * tamanho)
    barra = int(tamanho * 1.9)
    desenho.rectangle([0, 0, largura, barra], fill=FUNDO)
    desenho.text((tamanho, (barra - int(fonte.size * 1.25)) // 2), resumo, fill=BRANCO, font=fonte)

    # Faixa de alerta na parte de baixo
    if em_alerta:
        eq, tempo = max(em_alerta, key=lambda item: item[1])
        texto = f"ALERTA: {eq.id} aberto há {formatar_duracao(tempo)}. Feche a porta."
        if len(em_alerta) > 1:
            texto += f" (+{len(em_alerta) - 1})"
        fonte = _fonte_que_cabe(desenho, texto, int(tamanho * 1.5), largura - 2 * tamanho)
        faixa = int(tamanho * 3)
        desenho.rectangle([0, altura - faixa, largura, altura], fill=VERMELHO)
        desenho.text((tamanho, altura - faixa + (faixa - int(fonte.size * 1.25)) // 2), texto,
                     fill=BRANCO, font=fonte)

    return np.asarray(imagem)[:, :, ::-1].copy()
