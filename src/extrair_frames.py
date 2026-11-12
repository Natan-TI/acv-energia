"""Extrai imagens dos vídeos gravados, para anotar e treinar o modelo.

Exemplos:
    python src/extrair_frames.py --entrada data/videos --saida data/frames
    python src/extrair_frames.py --entrada data/videos/padaria.mp4 --intervalo 1 --borrar-rostos

Salva um quadro a cada `--intervalo` segundos e pula os que são quase iguais ao
último salvo, para o conjunto não ficar cheio de imagens repetidas.

Vídeos dentro de uma pasta chamada `demo` são ignorados: eles ficam reservados
para a demonstração e não podem entrar no treino.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

EXTENSOES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}
PASTA_RESERVADA = "demo"


def listar_videos(entrada: Path) -> list[Path]:
    if entrada.is_file():
        return [entrada]
    return sorted(
        p for p in entrada.rglob("*")
        if p.suffix.lower() in EXTENSOES and PASTA_RESERVADA not in p.relative_to(entrada).parts[:-1]
    )


def _miniatura(quadro: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(quadro, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.int16)


def regiao_do_rosto(pontos: list[list[float]], confiancas: list[float], caixa: tuple[float, float, float, float],
                    largura_quadro: int, altura_quadro: int, limiar: float = 0.4) -> tuple[int, int, int, int] | None:
    """Retângulo em volta do rosto de uma pessoa, a partir dos pontos do corpo (padrão COCO).

    Pontos usados: 0 nariz, 1 e 2 olhos, 3 e 4 orelhas, 5 e 6 ombros. Devolve None quando nenhum
    ponto do rosto está visível (pessoa de costas ou com a cabeça fora da imagem).
    """
    rosto = [pontos[i] for i in range(5) if confiancas[i] >= limiar]
    if not rosto:
        return None
    cx = sum(p[0] for p in rosto) / len(rosto)
    cy = sum(p[1] for p in rosto) / len(rosto)
    x1, y1, x2, y2 = caixa
    largura, altura = x2 - x1, y2 - y1

    def distancia(a, b) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # Três estimativas do tamanho da cabeça; vale a maior, limitada pelo tamanho da pessoa.
    candidatos = [0.06 * altura]
    if confiancas[5] >= limiar and confiancas[6] >= limiar:
        candidatos.append(0.45 * distancia(pontos[5], pontos[6]))
    if len(rosto) >= 2:
        candidatos.append(1.3 * max(distancia(a, b) for a in rosto for b in rosto))
    raio = max(min(max(candidatos), 0.35 * min(largura, altura)), 10)
    return (
        max(int(cx - raio), 0),
        max(int(cy - 1.3 * raio), 0),
        min(int(cx + raio), largura_quadro),
        min(int(cy + 1.3 * raio), altura_quadro),
    )


class BorradorDeRostos:
    """Borra o rosto de cada pessoa encontrada na imagem (LGPD).

    Usa o modelo YOLO de pose pré-treinado, que localiza nariz, olhos, orelhas e ombros.
    Assim só o rosto é coberto, e não o equipamento ao lado da pessoa. O OpenCV 5 não traz
    mais o detector de rostos antigo (CascadeClassifier), por isso ele não é usado aqui.
    """

    def __init__(self, confianca: float = 0.3) -> None:
        # Evita o erro "Fatal Error: HW capability found" em alguns computadores com Windows.
        os.environ.pop("OPENSSL_ia32cap", None)
        from ultralytics import YOLO

        self.modelo = YOLO("yolo11n-pose.pt")
        self.confianca = confianca

    def borrar(self, quadro: np.ndarray) -> np.ndarray:
        altura, largura = quadro.shape[:2]
        resultado = self.modelo.predict(quadro, conf=self.confianca, verbose=False)[0]
        if resultado.keypoints is None or resultado.keypoints.conf is None:
            return quadro
        for caixa, pontos, confiancas in zip(resultado.boxes.xyxy.tolist(), resultado.keypoints.xy.tolist(),
                                             resultado.keypoints.conf.tolist()):
            regiao = regiao_do_rosto(pontos, confiancas, tuple(caixa), largura, altura)
            if regiao is None:
                continue
            x1, y1, x2, y2 = regiao
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            lado = max(31, (max(x2 - x1, y2 - y1) // 2) | 1)
            quadro[y1:y2, x1:x2] = cv2.GaussianBlur(quadro[y1:y2, x1:x2], (lado, lado), 0)
        return quadro


def extrair(video: Path, saida: Path, intervalo: float = 0.5, largura_max: int = 1280,
            diferenca_minima: float = 4.0, borrador: BorradorDeRostos | None = None) -> int:
    """Extrai os quadros de um vídeo e devolve quantas imagens foram salvas."""
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        print(f"  não consegui abrir {video.name}, pulando")
        return 0
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    passo = max(int(round(fps * intervalo)), 1)

    saida.mkdir(parents=True, exist_ok=True)
    salvos, indice, ultima = 0, 0, None
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        if indice % passo == 0:
            altura, largura = quadro.shape[:2]
            if largura > largura_max:
                quadro = cv2.resize(quadro, (largura_max, int(altura * largura_max / largura)))
            miniatura = _miniatura(quadro)
            if ultima is None or float(np.abs(miniatura - ultima).mean()) >= diferenca_minima:
                if borrador is not None:
                    quadro = borrador.borrar(quadro)
                cv2.imwrite(str(saida / f"{video.stem}_{indice:06d}.jpg"), quadro, [cv2.IMWRITE_JPEG_QUALITY, 92])
                ultima = miniatura
                salvos += 1
        indice += 1
    captura.release()
    return salvos


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai quadros de vídeos para anotação.")
    parser.add_argument("--entrada", default="data/videos", help="vídeo ou pasta com vídeos")
    parser.add_argument("--saida", default="data/frames", help="pasta onde as imagens são salvas")
    parser.add_argument("--intervalo", type=float, default=0.5, help="segundos entre um quadro e outro")
    parser.add_argument("--largura-max", type=int, default=1280, help="reduz imagens mais largas que isso")
    parser.add_argument("--diferenca-minima", type=float, default=4.0,
                        help="pula quadros quase iguais ao último salvo (0 desliga)")
    parser.add_argument("--borrar-rostos", action="store_true",
                        help="borra o rosto das pessoas que aparecem na imagem (LGPD)")
    args = parser.parse_args()

    entrada = Path(args.entrada)
    videos = listar_videos(entrada)
    if not videos:
        raise SystemExit(f"Nenhum vídeo encontrado em {args.entrada}")
    if entrada.is_dir() and (entrada / PASTA_RESERVADA).is_dir():
        print(f"Ignorando a pasta {entrada / PASTA_RESERVADA} (vídeos reservados para a demonstração).")
    borrador = BorradorDeRostos() if args.borrar_rostos else None

    total = 0
    for video in videos:
        salvos = extrair(video, Path(args.saida), args.intervalo, args.largura_max,
                         args.diferenca_minima, borrador)
        print(f"  {video.name}: {salvos} imagens")
        total += salvos
    print(f"Total: {total} imagens em {args.saida}")


if __name__ == "__main__":
    main()
