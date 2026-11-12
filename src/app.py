"""Detecção em tempo real de portas de refrigeração abertas.

Exemplos:
    python src/app.py                               # webcam
    python src/app.py --fonte data/videos/teste.mp4 # vídeo gravado
    python src/app.py --fonte rtsp://usuario@ip/stream --salvar results/demo.mp4

Teclas: Q ou Esc para sair.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Em alguns computadores com Windows existe a variável de sistema OPENSSL_ia32cap, que faz uma
# biblioteca usada pelo Ultralytics (polars) abortar com "Fatal Error: HW capability found".
# Removê-la só deste processo evita o problema e não altera nada no sistema.
os.environ.pop("OPENSSL_ia32cap", None)

import cv2  # noqa: E402

import config  # noqa: E402
from eventos import GravadorEventos  # noqa: E402
from hud import desenhar, formatar_duracao, formatar_reais  # noqa: E402
from monitor import Deteccao  # noqa: E402

log = logging.getLogger("app")


class DetectorYolo:
    """Roda um modelo YOLO (Ultralytics) e devolve as detecções no formato do monitor."""

    def __init__(self, caminho: str | Path, confianca: float, classes: list[int] | None = None,
                 tamanho: int = 640) -> None:
        from ultralytics import YOLO  # importado aqui para os testes rodarem sem o pacote

        self.modelo = YOLO(str(caminho))
        self.confianca = confianca
        self.classes = classes
        self.tamanho = tamanho

    @property
    def nomes(self) -> list[str]:
        return list(self.modelo.names.values())

    def detectar(self, quadro) -> list[Deteccao]:
        resultado = self.modelo.predict(quadro, conf=self.confianca, classes=self.classes,
                                        imgsz=self.tamanho, verbose=False)[0]
        caixas = resultado.boxes
        return [
            Deteccao(resultado.names[int(classe)], float(confianca), tuple(float(v) for v in xyxy))
            for xyxy, confianca, classe in zip(caixas.xyxy.tolist(), caixas.conf.tolist(), caixas.cls.tolist())
        ]


def _tocar_alerta() -> None:
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONHAND)  # não bloqueia o vídeo
    except ImportError:
        print("\a", end="", flush=True)


def _abrir_fonte(fonte: str):
    ao_vivo = fonte.isdigit() or "://" in fonte
    captura = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)
    if not captura.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte de vídeo: {fonte}")
    return captura, ao_vivo


def executar(args, detector=None, detector_pessoas=None) -> dict:
    """Roda a detecção até o vídeo acabar ou o usuário sair. Devolve um resumo."""
    cfg = config.carregar(args.config)
    if args.loja:
        cfg["loja_id"] = args.loja

    if detector is None:
        caminho_modelo = Path(args.modelo or cfg["modelo"])
        if not caminho_modelo.is_absolute():
            caminho_modelo = config.RAIZ / caminho_modelo
        if not caminho_modelo.exists():
            raise SystemExit(
                f"Modelo não encontrado em {caminho_modelo}.\n"
                "Treine o modelo (notebooks/treino.ipynb) e copie o best.pt para a pasta models/."
            )
        detector = DetectorYolo(caminho_modelo, cfg["confianca_minima"], tamanho=cfg.get("tamanho_imagem", 640))
        desconhecidas = sorted(set(detector.nomes) - set(cfg["classes"]))
        if desconhecidas:
            log.warning("Classes do modelo sem regra no config.yaml (serão ignoradas): %s", desconhecidas)
        if not set(detector.nomes) & set(cfg["classes"]):
            raise SystemExit(
                f"Nenhuma classe do modelo ({detector.nomes}) aparece no config.yaml. "
                "Os nomes precisam ser iguais aos usados na anotação."
            )

    pessoas_cfg = cfg.get("pessoas", {})
    if detector_pessoas is None and pessoas_cfg.get("ativar"):
        # Classe 0 do COCO = pessoa
        detector_pessoas = DetectorYolo(pessoas_cfg.get("modelo", "yolo11n.pt"),
                                        pessoas_cfg.get("confianca_minima", 0.4), classes=[0],
                                        tamanho=cfg.get("tamanho_imagem", 640))

    eventos_cfg = cfg.get("eventos", {})
    pasta_eventos = Path(eventos_cfg.get("pasta", "eventos"))
    if not pasta_eventos.is_absolute():
        pasta_eventos = config.RAIZ / pasta_eventos
    enviar = eventos_cfg.get("enviar_para_api") or args.enviar
    gravador = GravadorEventos(
        pasta_eventos,
        api_url=eventos_cfg.get("api_url") if enviar else None,
        api_key=os.environ.get("PAINEL_API_KEY"),
    )

    captura, ao_vivo = _abrir_fonte(str(args.fonte))
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30.0
    if not 1 <= fps_video <= 240:
        fps_video = 30.0
    monitor = config.criar_monitor(cfg)
    alerta_cfg = cfg.get("alerta", {})

    gravacao = None
    janela = "Energia que da Lucro - deteccao (Q para sair)"
    indice, agora, fps_medido = 0, 0.0, None
    inicio_real = ultimo = time.monotonic()
    try:
        while True:
            ok, quadro = captura.read()
            if not ok:
                break
            # Em vídeo gravado o relógio é o do vídeo, para as durações não dependerem
            # da velocidade de processamento. Ao vivo, é o relógio do computador.
            agora = time.monotonic() - inicio_real if ao_vivo else indice / fps_video
            indice += 1

            deteccoes = detector.detectar(quadro)
            pessoas = [d.caixa for d in detector_pessoas.detectar(quadro)] if detector_pessoas else None
            for evento in monitor.atualizar(deteccoes, agora, pessoas):
                gravador.gravar(evento)
                log.info("Evento: %s aberto por %s (%s)", evento["equipamento_id"],
                         formatar_duracao(evento["duracao_s"]), formatar_reais(evento["custo_estimado_rs"]))

            if alerta_cfg.get("som", True) and not args.sem_som:
                for eq in monitor.equipamentos(agora):
                    if monitor.em_alerta(eq, agora) and (
                        eq.ultimo_som_em is None
                        or agora - eq.ultimo_som_em >= alerta_cfg.get("repetir_a_cada_s", 10)
                    ):
                        eq.ultimo_som_em = agora
                        _tocar_alerta()

            instante = time.monotonic()
            atual = 1.0 / max(instante - ultimo, 1e-6)
            fps_medido = atual if fps_medido is None else 0.9 * fps_medido + 0.1 * atual
            ultimo = instante

            saida = desenhar(quadro, monitor, agora, fps_medido)
            if args.salvar:
                if gravacao is None:
                    Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
                    altura, largura = saida.shape[:2]
                    gravacao = cv2.VideoWriter(str(args.salvar), cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_video, (largura, altura))
                gravacao.write(saida)
            if not args.sem_janela:
                cv2.imshow(janela, saida)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    finally:
        for evento in monitor.finalizar(agora):
            gravador.gravar(evento)
        captura.release()
        if gravacao is not None:
            gravacao.release()
        if not args.sem_janela:
            cv2.destroyAllWindows()

    resumo = {
        "quadros": indice,
        "duracao_s": round(agora, 1),
        "eventos": monitor.total_eventos,
        "custo_estimado_rs": round(monitor.custo_eventos_rs, 4),
        "pasta_eventos": str(pasta_eventos),
    }
    log.info("Fim. %d quadros, %d eventos, desperdício estimado %s. Eventos em %s",
             resumo["quadros"], resumo["eventos"], formatar_reais(resumo["custo_estimado_rs"]), pasta_eventos)
    return resumo


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detecção em tempo real de portas de refrigeração abertas.")
    parser.add_argument("--fonte", default="0", help="número da webcam, arquivo de vídeo ou endereço da câmera")
    parser.add_argument("--config", default=None, help="caminho do config.yaml")
    parser.add_argument("--modelo", default=None, help="pesos do modelo (padrão: o do config.yaml)")
    parser.add_argument("--loja", default=None, help="loja_id (padrão: o do config.yaml)")
    parser.add_argument("--salvar", default=None, help="grava o vídeo com as marcações neste arquivo .mp4")
    parser.add_argument("--sem-janela", action="store_true", help="não abre janela (útil para só gravar)")
    parser.add_argument("--sem-som", action="store_true", help="não toca o alerta sonoro")
    parser.add_argument("--enviar", action="store_true", help="envia os eventos para a API do painel")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    executar(criar_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
