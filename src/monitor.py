"""Regras de negócio da detecção: transforma detecções quadro a quadro em eventos de desperdício.

Este módulo não depende de OpenCV nem do YOLO. Ele recebe uma lista de detecções e o
instante atual, e decide:

- qual equipamento cada detecção representa (as câmeras e as geladeiras são fixas,
  então a associação é feita pela posição da caixa);
- se a porta está aberta ou fechada, ignorando oscilações rápidas do modelo;
- quando uma porta aberta vira alerta;
- quando um evento termina e quanto ele custou.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

FUSO_BRASILIA = timezone(timedelta(hours=-3))

ABERTO = "aberto"
FECHADO = "fechado"

Caixa = tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels


@dataclass(frozen=True)
class Deteccao:
    classe: str
    confianca: float
    caixa: Caixa


@dataclass
class Regras:
    limite_alerta_s: float = 60.0
    confirmacao_s: float = 0.7
    tempo_perdido_s: float = 15.0
    iou_mesmo_equipamento: float = 0.3
    iou_tipo_confundido: float = 0.6
    duracao_minima_evento_s: float = 2.0
    pessoas_ativas: bool = False
    tolerancia_pessoa_s: float = 5.0


@dataclass
class Custo:
    """custo = potencia_kw x fator_perda x horas x tarifa_rs_kwh"""

    tarifa_rs_kwh: float
    equipamentos: dict[str, dict[str, float]]

    def calcular(self, tipo: str, duracao_s: float) -> float:
        dados = self.equipamentos.get(tipo)
        if dados is None:
            return 0.0
        horas = max(duracao_s, 0.0) / 3600.0
        return dados["potencia_kw"] * dados["fator_perda"] * horas * self.tarifa_rs_kwh


@dataclass
class Equipamento:
    tipo: str
    caixa: Caixa
    criado_em: float
    visto_em: float
    id: str | None = None  # definido quando o equipamento é confirmado
    estado: str | None = None
    candidato: str | None = None
    candidato_desde: float = 0.0
    aberto_desde: float | None = None
    pessoa_perto_em: float | None = None
    ultimo_som_em: float | None = None


def iou(a: Caixa, b: Caixa) -> float:
    largura = min(a[2], b[2]) - max(a[0], b[0])
    altura = min(a[3], b[3]) - max(a[1], b[1])
    if largura <= 0 or altura <= 0:
        return 0.0
    intersecao = largura * altura
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersecao / (area_a + area_b - intersecao)


def _centro_dentro(a: Caixa, b: Caixa) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _expandir(caixa: Caixa, fracao: float) -> Caixa:
    dx = (caixa[2] - caixa[0]) * fracao
    dy = (caixa[3] - caixa[1]) * fracao
    return (caixa[0] - dx, caixa[1] - dy, caixa[2] + dx, caixa[3] + dy)


@dataclass
class Monitor:
    loja_id: str
    classes: dict[str, tuple[str, str]]  # nome da classe -> (tipo do equipamento, estado)
    custo: Custo
    regras: Regras = field(default_factory=Regras)
    inicio_relogio: datetime = field(default_factory=lambda: datetime.now(FUSO_BRASILIA))

    def __post_init__(self) -> None:
        self._equipamentos: list[Equipamento] = []
        self._contador: dict[str, int] = {}
        self.custo_eventos_rs = 0.0
        self.total_eventos = 0

    # ------------------------------------------------------------------ consulta
    def equipamentos(self, agora: float) -> list[Equipamento]:
        """Equipamentos confirmados e vistos recentemente (para desenhar na tela)."""
        return [
            e for e in self._equipamentos
            if e.id is not None and agora - e.visto_em <= self.regras.tempo_perdido_s
        ]

    def tempo_aberto(self, eq: Equipamento, agora: float) -> float:
        if eq.estado != ABERTO or eq.aberto_desde is None:
            return 0.0
        return max(agora - eq.aberto_desde, 0.0)

    def em_alerta(self, eq: Equipamento, agora: float) -> bool:
        if self.tempo_aberto(eq, agora) < self.regras.limite_alerta_s:
            return False
        if self.regras.pessoas_ativas and eq.pessoa_perto_em is not None:
            # Com um cliente na frente da geladeira a porta aberta é atendimento, não desperdício.
            return agora - eq.pessoa_perto_em > self.regras.tolerancia_pessoa_s
        return True

    def custo_em_aberto_rs(self, agora: float) -> float:
        return sum(self.custo.calcular(e.tipo, self.tempo_aberto(e, agora)) for e in self._equipamentos)

    def custo_total_rs(self, agora: float) -> float:
        return self.custo_eventos_rs + self.custo_em_aberto_rs(agora)

    # --------------------------------------------------------------- atualização
    def atualizar(self, deteccoes: list[Deteccao], agora: float, pessoas: list[Caixa] | None = None) -> list[dict]:
        """Processa as detecções de um quadro e devolve os eventos encerrados nele."""
        eventos: list[dict] = []
        usados: set[int] = set()

        for det in sorted(deteccoes, key=lambda d: d.confianca, reverse=True):
            mapa = self.classes.get(det.classe)
            if mapa is None:
                continue
            tipo, estado = mapa
            eq = self._melhor_equipamento(det.caixa, tipo)
            if eq is None:
                eq = Equipamento(tipo=tipo, caixa=det.caixa, criado_em=agora, visto_em=agora)
                self._equipamentos.append(eq)
            elif id(eq) in usados:
                # Segunda detecção sobre o mesmo equipamento no mesmo quadro (por exemplo,
                # "aberta" e "fechada" ao mesmo tempo): vale a de maior confiança.
                continue
            usados.add(id(eq))
            eq.caixa = det.caixa
            eq.visto_em = agora
            eventos += self._observar(eq, estado, agora)

        for caixa_pessoa in pessoas or []:
            for eq in self._equipamentos:
                if iou(_expandir(eq.caixa, 0.2), caixa_pessoa) > 0:
                    eq.pessoa_perto_em = agora

        eventos += self._limpar(agora)
        return eventos

    def finalizar(self, agora: float) -> list[dict]:
        """Encerra os eventos ainda abertos quando o vídeo ou a câmera termina."""
        eventos: list[dict] = []
        for eq in self._equipamentos:
            if eq.estado == ABERTO:
                eventos += self._fechar_evento(eq, agora)
                eq.estado, eq.aberto_desde = None, None
        return eventos

    # ------------------------------------------------------------------ internos
    def _melhor_equipamento(self, caixa: Caixa, tipo: str) -> Equipamento | None:
        melhor, melhor_nota = None, 0.0
        for eq in self._equipamentos:
            nota = iou(caixa, eq.caixa)
            if eq.tipo != tipo:
                # Tipo diferente no mesmo lugar: se as caixas quase coincidem, é o mesmo objeto e o
                # modelo confundiu geladeira com freezer. Mantém o equipamento que já existe.
                if nota < self.regras.iou_tipo_confundido:
                    continue
            elif nota < self.regras.iou_mesmo_equipamento:
                # A caixa muda bastante quando a porta abre; o centro ajuda a reconhecer o mesmo equipamento.
                if not (_centro_dentro(caixa, eq.caixa) or _centro_dentro(eq.caixa, caixa)):
                    continue
                nota = max(nota, 0.01)
            if nota > melhor_nota:
                melhor, melhor_nota = eq, nota
        return melhor

    def _observar(self, eq: Equipamento, estado: str, agora: float) -> list[dict]:
        if estado == eq.estado:
            eq.candidato = None
            return []
        if eq.candidato != estado:
            eq.candidato, eq.candidato_desde = estado, agora
        if agora - eq.candidato_desde < self.regras.confirmacao_s:
            return []
        return self._mudar_estado(eq, estado, eq.candidato_desde)

    def _mudar_estado(self, eq: Equipamento, novo: str, quando: float) -> list[dict]:
        eventos: list[dict] = []
        if eq.id is None:
            numero = self._contador.get(eq.tipo, 0) + 1
            self._contador[eq.tipo] = numero
            eq.id = f"{eq.tipo}-{numero}"
        if eq.estado == ABERTO and novo == FECHADO:
            eventos = self._fechar_evento(eq, quando)
        eq.estado, eq.candidato = novo, None
        eq.aberto_desde = quando if novo == ABERTO else None
        eq.ultimo_som_em = None
        return eventos

    def _fechar_evento(self, eq: Equipamento, fim: float) -> list[dict]:
        if eq.aberto_desde is None:
            return []
        duracao = fim - eq.aberto_desde
        if duracao < self.regras.duracao_minima_evento_s:
            return []
        custo = self.custo.calcular(eq.tipo, duracao)
        self.custo_eventos_rs += custo
        self.total_eventos += 1
        inicio = self.inicio_relogio + timedelta(seconds=eq.aberto_desde)
        return [{
            "id": str(uuid.uuid4()),
            "loja_id": self.loja_id,
            "equipamento_id": eq.id,
            "tipo": "porta_aberta",
            "origem": "acv",
            "inicio": inicio.astimezone(FUSO_BRASILIA).isoformat(timespec="seconds"),
            "duracao_s": max(int(round(duracao)), 1),
            "custo_estimado_rs": round(custo, 4),
        }]

    def _limpar(self, agora: float) -> list[dict]:
        eventos: list[dict] = []
        for eq in list(self._equipamentos):
            sumido_ha = agora - eq.visto_em
            if sumido_ha <= self.regras.tempo_perdido_s:
                continue
            if eq.id is None:
                # Detecção passageira que nunca se confirmou: descarta.
                self._equipamentos.remove(eq)
            elif eq.estado is not None:
                # Sumiu por muito tempo (câmera tapada, por exemplo). Encerra o evento no
                # último instante em que foi visto e volta a confirmar o estado quando reaparecer.
                if eq.estado == ABERTO:
                    eventos += self._fechar_evento(eq, eq.visto_em)
                eq.estado, eq.candidato, eq.aberto_desde = None, None, None
        return eventos
