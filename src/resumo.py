"""Mostra o resumo dos eventos gravados: quanto tempo cada equipamento ficou aberto e quanto custou.

Exemplo:
    python src/resumo.py
    python src/resumo.py --pasta eventos
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import config
from eventos import ler_eventos
from hud import formatar_duracao, formatar_reais


def resumir(eventos: list[dict]) -> dict[str, dict]:
    por_equipamento: dict[str, dict] = defaultdict(lambda: {"eventos": 0, "duracao_s": 0, "custo_estimado_rs": 0.0})
    for evento in eventos:
        linha = por_equipamento[f"{evento['loja_id']} / {evento['equipamento_id']}"]
        linha["eventos"] += 1
        linha["duracao_s"] += evento["duracao_s"]
        linha["custo_estimado_rs"] += evento["custo_estimado_rs"]
    return dict(por_equipamento)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumo dos eventos de porta aberta.")
    parser.add_argument("--pasta", default=str(config.RAIZ / "eventos"))
    args = parser.parse_args()

    eventos = ler_eventos(Path(args.pasta))
    if not eventos:
        raise SystemExit(f"Nenhum evento encontrado em {args.pasta}")
    resumo = resumir(eventos)
    print(f"{'Loja / equipamento':<32}{'Eventos':>9}{'Tempo aberto':>15}{'Custo estimado':>17}")
    for nome, linha in sorted(resumo.items()):
        print(f"{nome:<32}{linha['eventos']:>9}{formatar_duracao(linha['duracao_s']):>15}"
              f"{formatar_reais(linha['custo_estimado_rs']):>17}")
    total_tempo = sum(l["duracao_s"] for l in resumo.values())
    total_custo = sum(l["custo_estimado_rs"] for l in resumo.values())
    print(f"{'Total':<32}{len(eventos):>9}{formatar_duracao(total_tempo):>15}{formatar_reais(total_custo):>17}")


if __name__ == "__main__":
    main()
