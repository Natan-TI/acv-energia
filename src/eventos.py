"""Gravação dos eventos em arquivo e envio opcional para a API do painel."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("eventos")

CAMPOS = ("id", "loja_id", "equipamento_id", "tipo", "origem", "inicio", "duracao_s", "custo_estimado_rs")


class GravadorEventos:
    """Grava um evento por linha (JSON Lines), em um arquivo por dia.

    O arquivo é sempre gravado. O envio para a API é opcional e nunca derruba a
    aplicação: se a API estiver fora do ar, o erro vai para o log e o evento
    continua salvo em disco.
    """

    def __init__(self, pasta: str | Path, api_url: str | None = None, api_key: str | None = None) -> None:
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.api_url = api_url.rstrip("/") if api_url else None
        self.api_key = api_key

    def gravar(self, evento: dict) -> Path:
        faltando = [campo for campo in CAMPOS if campo not in evento]
        if faltando:
            raise ValueError(f"Evento fora do contrato, faltam os campos: {faltando}")
        arquivo = self.pasta / f"eventos-{evento['inicio'][:10]}.jsonl"
        with open(arquivo, "a", encoding="utf-8") as saida:
            saida.write(json.dumps(evento, ensure_ascii=False) + "\n")
        if self.api_url:
            self._enviar(evento)
        return arquivo

    def _enviar(self, evento: dict) -> None:
        try:
            import requests

            cabecalhos = {"X-API-Key": self.api_key} if self.api_key else {}
            resposta = requests.post(f"{self.api_url}/events", json=evento, headers=cabecalhos, timeout=3)
            resposta.raise_for_status()
        except Exception as erro:  # a detecção não pode parar por causa da rede
            log.warning("Evento %s salvo em disco, mas não enviado à API: %s", evento["id"], erro)


def ler_eventos(pasta: str | Path) -> list[dict]:
    eventos: list[dict] = []
    for arquivo in sorted(Path(pasta).glob("eventos-*.jsonl")):
        with open(arquivo, encoding="utf-8") as entrada:
            eventos += [json.loads(linha) for linha in entrada if linha.strip()]
    return eventos
