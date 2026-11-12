"""Testa a aplicação de ponta a ponta com um vídeo sintético e um detector falso.

Cobre a leitura do vídeo, o relógio, o desenho na tela, a gravação do vídeo de saída
e a gravação dos eventos. O modelo YOLO de verdade não é necessário.
"""
import argparse
import json

import cv2
import numpy as np
import pytest
import yaml

import app
import config
from eventos import GravadorEventos, ler_eventos
from extrair_frames import extrair, listar_videos, regiao_do_rosto
from monitor import Deteccao
from resumo import resumir

FPS = 20
CAIXA = (60.0, 40.0, 200.0, 220.0)


class DetectorFalso:
    """Fechada nos 2 primeiros segundos, aberta por 8 segundos, fechada no fim."""

    def __init__(self):
        self.quadro = 0

    def detectar(self, _quadro):
        segundo = self.quadro / FPS
        self.quadro += 1
        classe = "geladeira_aberta" if 2 <= segundo < 10 else "geladeira_fechada"
        return [Deteccao(classe, 0.9, CAIXA)]


def criar_video(caminho, segundos=12, estatico=False):
    gravacao = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    assert gravacao.isOpened()
    for i in range(segundos * FPS):
        quadro = np.full((240, 320, 3), 60, np.uint8)
        if not estatico:
            cv2.rectangle(quadro, (0, 0), ((i * 7) % 320, 240), (200, 120, (i * 3) % 255), -1)
        gravacao.write(quadro)
    gravacao.release()


@pytest.fixture
def ambiente(tmp_path):
    cfg = config.carregar()
    cfg["eventos"]["pasta"] = str(tmp_path / "eventos")
    cfg["regras"]["limite_alerta_s"] = 3
    arquivo_cfg = tmp_path / "config.yaml"
    arquivo_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    video = tmp_path / "entrada.mp4"
    criar_video(video)
    return tmp_path, arquivo_cfg, video


def test_aplicacao_de_ponta_a_ponta(ambiente):
    pasta, arquivo_cfg, video = ambiente
    saida = pasta / "saida.mp4"
    args = argparse.Namespace(fonte=str(video), config=str(arquivo_cfg), modelo=None, loja="mercado-01",
                              salvar=str(saida), sem_janela=True, sem_som=True, enviar=False)

    resumo = app.executar(args, detector=DetectorFalso())

    assert resumo["quadros"] == 12 * FPS
    assert resumo["eventos"] == 1
    eventos = ler_eventos(pasta / "eventos")
    assert len(eventos) == 1
    assert eventos[0]["loja_id"] == "mercado-01"
    assert eventos[0]["equipamento_id"] == "expositor-1"
    assert eventos[0]["duracao_s"] == 8
    assert saida.exists() and saida.stat().st_size > 0

    gravado = cv2.VideoCapture(str(saida))
    assert int(gravado.get(cv2.CAP_PROP_FRAME_COUNT)) == 12 * FPS
    gravado.release()


def test_modelo_ausente_explica_o_que_fazer(ambiente):
    _, arquivo_cfg, video = ambiente
    args = argparse.Namespace(fonte=str(video), config=str(arquivo_cfg), modelo="models/nao_existe.pt", loja=None,
                              salvar=None, sem_janela=True, sem_som=True, enviar=False)
    with pytest.raises(SystemExit, match="Modelo não encontrado"):
        app.executar(args)


def test_gravador_valida_o_contrato_e_sobrevive_a_api_fora_do_ar(tmp_path):
    evento = {"id": "1", "loja_id": "padaria-01", "equipamento_id": "freezer-1", "tipo": "porta_aberta",
              "origem": "acv", "inicio": "2026-11-12T14:03:10-03:00", "duracao_s": 95, "custo_estimado_rs": 0.0036}
    gravador = GravadorEventos(tmp_path, api_url="http://127.0.0.1:9", api_key="teste")
    arquivo = gravador.gravar(evento)  # a API não existe: o evento precisa ser salvo mesmo assim
    assert arquivo.name == "eventos-2026-11-12.jsonl"
    assert json.loads(arquivo.read_text(encoding="utf-8")) == evento
    with pytest.raises(ValueError, match="custo_estimado_rs"):
        gravador.gravar({k: v for k, v in evento.items() if k != "custo_estimado_rs"})

    linha = resumir(ler_eventos(tmp_path))["padaria-01 / freezer-1"]
    assert linha == {"eventos": 1, "duracao_s": 95, "custo_estimado_rs": 0.0036}


def test_extracao_de_frames_pula_imagens_repetidas(tmp_path):
    criar_video(tmp_path / "movimento.mp4", segundos=4)
    criar_video(tmp_path / "parado.mp4", segundos=4, estatico=True)
    assert extrair(tmp_path / "movimento.mp4", tmp_path / "a", intervalo=0.5) >= 6
    assert extrair(tmp_path / "parado.mp4", tmp_path / "b", intervalo=0.5) == 1
    assert len(list((tmp_path / "a").glob("movimento_*.jpg"))) >= 6


def test_videos_da_pasta_demo_nao_entram_na_extracao(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "loja").mkdir()
    for nome in ("video1.mp4", "demo/reservado.mp4", "loja/video2.MOV", "notas.txt"):
        (tmp_path / nome).write_bytes(b"")
    assert [v.name for v in listar_videos(tmp_path)] == ["video2.MOV", "video1.mp4"]
    # Um arquivo indicado diretamente é sempre aceito.
    assert listar_videos(tmp_path / "demo" / "reservado.mp4") == [tmp_path / "demo" / "reservado.mp4"]


def _pessoa(rosto, ombros=None, visiveis=(0, 1, 2, 3, 4)):
    """17 pontos no padrão COCO; só os índices em `visiveis` têm confiança alta."""
    pontos = [[0.0, 0.0] for _ in range(17)]
    confiancas = [0.0] * 17
    for i in visiveis:
        pontos[i], confiancas[i] = list(rosto[i]), 0.9
    if ombros:
        pontos[5], pontos[6] = list(ombros[0]), list(ombros[1])
        confiancas[5] = confiancas[6] = 0.9
    return pontos, confiancas


def test_regiao_do_rosto_cobre_so_o_rosto():
    rosto = {0: (150, 100), 1: (158, 92), 2: (142, 92), 3: (168, 96), 4: (132, 96)}
    # Pessoa em pé, de frente: orelhas a 36 px, ombros a 80 px
    pontos, confiancas = _pessoa(rosto, ombros=((190, 150), (110, 150)))
    x1, y1, x2, y2 = regiao_do_rosto(pontos, confiancas, (90, 60, 210, 420), 640, 480)
    assert x1 <= 132 and x2 >= 168 and y1 <= 92 and y2 >= 100   # cobre olhos, nariz e orelhas
    assert x2 - x1 <= 100 and y2 - y1 <= 130                    # e não a pessoa inteira
    # Pessoa bem perto da câmera, cortada pela borda: a região continua limitada
    grande = {i: (x * 3, y * 3) for i, (x, y) in rosto.items()}
    pontos, confiancas = _pessoa(grande, ombros=((640, 470), (250, 470)))
    x1, y1, x2, y2 = regiao_do_rosto(pontos, confiancas, (200, 150, 640, 480), 640, 480)
    assert (x2 - x1) * (y2 - y1) < 0.25 * 640 * 480
    assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 480


def test_pessoa_de_costas_ou_sem_cabeca_visivel_nao_e_borrada():
    pontos, confiancas = _pessoa({}, ombros=((190, 150), (110, 150)), visiveis=())
    assert regiao_do_rosto(pontos, confiancas, (90, 60, 210, 420), 640, 480) is None


def test_borrador_e_aplicado_so_nas_imagens_salvas(tmp_path):
    class BorradorFalso:
        chamadas = 0

        def borrar(self, quadro):
            self.chamadas += 1
            quadro[:] = 0
            return quadro

    criar_video(tmp_path / "movimento.mp4", segundos=2)
    borrador = BorradorFalso()
    salvos = extrair(tmp_path / "movimento.mp4", tmp_path / "saida", intervalo=0.5, borrador=borrador)
    assert salvos == borrador.chamadas >= 3
    assert cv2.imread(str(next((tmp_path / "saida").glob("*.jpg")))).max() < 10
