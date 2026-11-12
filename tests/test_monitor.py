"""Testes das regras de negócio. Não precisam de câmera, de GPU nem do modelo treinado."""
import uuid
from datetime import datetime

from monitor import ABERTO, FECHADO, FUSO_BRASILIA, Custo, Deteccao, Monitor, Regras

CLASSES = {
    "geladeira_aberta": ("expositor", ABERTO),
    "geladeira_fechada": ("expositor", FECHADO),
    "freezer_aberto": ("freezer", ABERTO),
    "freezer_fechado": ("freezer", FECHADO),
}
G1 = (100, 100, 300, 500)
G2 = (400, 100, 600, 500)
F1 = (100, 550, 500, 700)


def novo_monitor(**regras) -> Monitor:
    custo = Custo(0.90, {
        "expositor": {"potencia_kw": 0.50, "fator_perda": 0.30},
        "freezer": {"potencia_kw": 0.50, "fator_perda": 0.30},
    })
    return Monitor("padaria-01", CLASSES, custo, Regras(**regras),
                   inicio_relogio=datetime(2026, 11, 12, 14, 0, 0, tzinfo=FUSO_BRASILIA))


def rodar(monitor, inicio, fim, classe=None, caixa=G1, pessoas=None):
    """Simula 10 quadros por segundo entre `inicio` e `fim` (em segundos)."""
    eventos = []
    for i in range(int(inicio * 10), int(fim * 10)):
        deteccoes = [Deteccao(classe, 0.9, caixa)] if classe else []
        eventos += monitor.atualizar(deteccoes, i / 10, pessoas)
    return eventos


def test_porta_aberta_gera_evento_no_formato_do_contrato():
    m = novo_monitor()
    eventos = rodar(m, 0, 5, "geladeira_fechada")
    eventos += rodar(m, 5, 100, "geladeira_aberta")
    eventos += rodar(m, 100, 105, "geladeira_fechada")

    assert len(eventos) == 1
    evento = eventos[0]
    assert set(evento) == {"id", "loja_id", "equipamento_id", "tipo", "origem", "inicio",
                           "duracao_s", "custo_estimado_rs"}
    uuid.UUID(evento["id"])
    assert evento["loja_id"] == "padaria-01"
    assert evento["equipamento_id"] == "expositor-1"
    assert evento["tipo"] == "porta_aberta"
    assert evento["origem"] == "acv"
    assert evento["inicio"] == "2026-11-12T14:00:05-03:00"
    assert evento["duracao_s"] == 95
    # 0,50 kW x 0,30 x (95 / 3600) h x R$ 0,90 = R$ 0,0035625
    assert evento["custo_estimado_rs"] == 0.0036
    assert m.total_eventos == 1


def test_oscilacao_de_um_quadro_nao_fecha_o_evento():
    m = novo_monitor()
    eventos = rodar(m, 0, 10, "geladeira_aberta")
    eventos += m.atualizar([Deteccao("geladeira_fechada", 0.9, G1)], 10.0)
    eventos += rodar(m, 10.1, 30, "geladeira_aberta")
    assert eventos == []
    eventos = m.finalizar(30.0)
    assert len(eventos) == 1 and eventos[0]["duracao_s"] == 30


def test_abertura_muito_curta_nao_vira_evento():
    m = novo_monitor()
    eventos = rodar(m, 0, 3, "geladeira_fechada")
    eventos += rodar(m, 3, 4.5, "geladeira_aberta")
    eventos += rodar(m, 4.5, 8, "geladeira_fechada")
    assert eventos == []


def test_cada_equipamento_recebe_um_identificador():
    m = novo_monitor()
    for i in range(20):
        m.atualizar([
            Deteccao("geladeira_fechada", 0.9, G1),
            Deteccao("geladeira_aberta", 0.8, G2),
            Deteccao("freezer_fechado", 0.7, F1),
        ], i / 10)
    estados = {e.id: e.estado for e in m.equipamentos(2.0)}
    assert estados == {"expositor-1": FECHADO, "expositor-2": ABERTO, "freezer-1": FECHADO}


def test_duas_deteccoes_no_mesmo_equipamento_vale_a_mais_confiante():
    m = novo_monitor()
    for i in range(20):
        m.atualizar([
            Deteccao("geladeira_fechada", 0.6, G1),
            Deteccao("geladeira_aberta", 0.9, (95, 100, 310, 505)),
        ], i / 10)
    equipamentos = m.equipamentos(2.0)
    assert [(e.id, e.estado) for e in equipamentos] == [("expositor-1", ABERTO)]


def test_caixa_maior_com_a_porta_aberta_continua_sendo_o_mesmo_equipamento():
    m = novo_monitor()
    eventos = rodar(m, 0, 3, "geladeira_fechada", G1)
    eventos += rodar(m, 3, 13, "geladeira_aberta", (0, 50, 700, 550))  # IoU baixo, mas o centro coincide
    eventos += rodar(m, 13, 16, "geladeira_fechada", G1)
    assert [e["equipamento_id"] for e in eventos] == ["expositor-1"]
    assert eventos[0]["duracao_s"] == 10


def test_oclusao_curta_mantem_o_cronometro():
    m = novo_monitor(tempo_perdido_s=15)
    eventos = rodar(m, 0, 10, "geladeira_aberta")
    eventos += rodar(m, 10, 20)  # alguém passou na frente: nenhuma detecção
    eventos += rodar(m, 20, 25, "geladeira_aberta")
    eventos += rodar(m, 25, 28, "geladeira_fechada")
    assert len(eventos) == 1 and eventos[0]["duracao_s"] == 25


def test_sumico_longo_encerra_o_evento_no_ultimo_instante_visto():
    m = novo_monitor(tempo_perdido_s=15)
    eventos = rodar(m, 0, 10, "geladeira_aberta")
    eventos += rodar(m, 10, 40)
    assert len(eventos) == 1 and eventos[0]["duracao_s"] == 10
    # Quando reaparece, o estado é confirmado de novo e o identificador é mantido.
    rodar(m, 40, 43, "geladeira_fechada")
    assert [(e.id, e.estado) for e in m.equipamentos(43.0)] == [("expositor-1", FECHADO)]


def test_alerta_dispara_depois_do_limite():
    m = novo_monitor(limite_alerta_s=60)
    rodar(m, 0, 61, "geladeira_aberta")
    equipamento = m.equipamentos(61.0)[0]
    assert not m.em_alerta(equipamento, 30.0)
    assert m.em_alerta(equipamento, 61.0)


def test_cliente_na_frente_da_geladeira_adia_o_alerta():
    m = novo_monitor(limite_alerta_s=60, pessoas_ativas=True, tolerancia_pessoa_s=5)
    cliente = (250, 150, 380, 520)
    rodar(m, 0, 70, "geladeira_aberta", pessoas=[cliente])
    equipamento = m.equipamentos(70.0)[0]
    assert not m.em_alerta(equipamento, 70.0)
    rodar(m, 70, 77, "geladeira_aberta", pessoas=[])
    assert m.em_alerta(equipamento, 77.0)


def test_deteccao_passageira_e_classe_desconhecida_sao_ignoradas():
    m = novo_monitor()
    m.atualizar([Deteccao("geladeira_aberta", 0.9, G2), Deteccao("cachorro", 0.9, G1)], 0.0)
    rodar(m, 0.1, 20, "geladeira_fechada", G1)
    assert [e.id for e in m.equipamentos(20.0)] == ["expositor-1"]


def test_custo_acumulado_inclui_portas_ainda_abertas():
    m = novo_monitor()
    rodar(m, 0, 3600, "geladeira_aberta")
    # 1 hora aberta: 0,50 x 0,30 x 1 x 0,90 = R$ 0,135
    assert abs(m.custo_total_rs(3600.0) - 0.135) < 1e-6


def test_mesmo_objeto_visto_como_geladeira_e_como_freezer_vira_um_equipamento_so():
    m = novo_monitor()
    for i in range(20):
        m.atualizar([
            Deteccao("geladeira_fechada", 0.98, G1),
            Deteccao("freezer_fechado", 0.96, (98, 99, 303, 502)),
        ], i / 10)
    assert [(e.id, e.estado) for e in m.equipamentos(2.0)] == [("expositor-1", FECHADO)]
    # Um freezer de verdade, em outro lugar, continua sendo outro equipamento.
    rodar(m, 2, 4, "freezer_aberto", F1)
    assert sorted(e.id for e in m.equipamentos(4.0)) == ["expositor-1", "freezer-1"]
