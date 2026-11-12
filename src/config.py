"""Leitura do config.yaml e montagem do monitor."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from monitor import FUSO_BRASILIA, Custo, Monitor, Regras

RAIZ = Path(__file__).resolve().parent.parent


def carregar(caminho: str | Path | None = None) -> dict:
    caminho = Path(caminho) if caminho else RAIZ / "config.yaml"
    with open(caminho, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def criar_monitor(cfg: dict, inicio: datetime | None = None) -> Monitor:
    regras_cfg = cfg.get("regras", {})
    pessoas_cfg = cfg.get("pessoas", {})
    regras = Regras(
        limite_alerta_s=float(regras_cfg.get("limite_alerta_s", 60)),
        confirmacao_s=float(regras_cfg.get("confirmacao_s", 0.7)),
        tempo_perdido_s=float(regras_cfg.get("tempo_perdido_s", 15)),
        iou_mesmo_equipamento=float(regras_cfg.get("iou_mesmo_equipamento", 0.3)),
        iou_tipo_confundido=float(regras_cfg.get("iou_tipo_confundido", 0.6)),
        duracao_minima_evento_s=float(regras_cfg.get("duracao_minima_evento_s", 2)),
        pessoas_ativas=bool(pessoas_cfg.get("ativar", False)),
        tolerancia_pessoa_s=float(pessoas_cfg.get("tolerancia_s", 5)),
    )
    custo = Custo(
        tarifa_rs_kwh=float(cfg["custo"]["tarifa_rs_kwh"]),
        equipamentos={
            tipo: {"potencia_kw": float(d["potencia_kw"]), "fator_perda": float(d["fator_perda"])}
            for tipo, d in cfg["custo"]["equipamentos"].items()
        },
    )
    classes = {nome: (d["equipamento"], d["estado"]) for nome, d in cfg["classes"].items()}
    return Monitor(
        loja_id=cfg["loja_id"],
        classes=classes,
        custo=custo,
        regras=regras,
        inicio_relogio=inicio or datetime.now(FUSO_BRASILIA),
    )
