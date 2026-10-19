"""
nr_modelo.py - código compartilhado entre o TREINO e o BACKEND.

As features são calculadas pela mesma função nos dois lugares, para o modelo
não ver no ar algo diferente do que viu no treino.

Modelo: RSRP = P0 - PL_38901_UMa(d, h) + RF(features)
  - PL_38901_UMa: perda de percurso 3GPP TR 38.901, urbano macro, média de
    LOS e NLOS ponderada pela probabilidade de LOS.
  - P0: constante em dBm calibrada nos dados de treino (potência por RE e
    ganho de antena não são conhecidos).
  - RF: Random Forest treinado no resíduo (medido - baseline).
"""
import numpy as np

FC_GHZ = 3.5          # referência da n78
H_UT = 1.5            # altura do receptor (m)
C_LUZ = 3e8
R_TERRA = 6371000.0

# Variante "plano": só usa o que o app envia (lat, lon, azimute, altura). Vai pro ar.
FEATURES_PLANO = ["log10_d2d", "ang_off_az"]
# Variante "terreno": usa altitudes. Só para o relatório (mostra o ganho de um DEM).
FEATURES_TERRENO = ["log10_d2d", "ang_off_az", "h_rel_m", "elev_graus"]

# Escala do produto (não é padrão 3GPP)
LIMIAR_BOA = -90.0
LIMIAR_MEDIANA = -110.0

# Guardrail de região: o modelo só foi treinado em Juiz de Fora
CENTRO_JF = (-21.7642, -43.3496)
RAIO_MAX_JF_M = 30000


def distancia_2d(lat1, lon1, lat2, lon2):
    """Haversine vetorizado, em metros."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TERRA * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def rumo(lat1, lon1, lat2, lon2):
    """Rumo de 1 para 2, em graus (0 = norte, sentido horário)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def prob_los_uma(d2d):
    """Probabilidade de LOS, 38.901 Tabela 7.4.2-1 (UMa, h_UT <= 13 m)."""
    d = np.maximum(d2d, 1e-6)
    p = 18 / d + np.exp(-d / 63) * (1 - 18 / d)
    return np.where(d2d <= 18, 1.0, p)


def pl_38901_uma(d2d, h_bs, h_ut=H_UT, fc_ghz=FC_GHZ):
    """Perda de percurso esperada (dB), 38.901 Tabela 7.4.1-1, UMa."""
    d2d = np.clip(d2d, 10, 5000)
    h_bs = np.clip(h_bs, 10, 150)
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    lf = 20 * np.log10(fc_ghz)
    d_bp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc_ghz * 1e9 / C_LUZ
    pl1 = 28.0 + 22 * np.log10(d3d) + lf
    pl2 = 28.0 + 40 * np.log10(d3d) + lf - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos = np.maximum(pl_los, 13.54 + 39.08 * np.log10(d3d) + lf - 0.6 * (h_ut - 1.5))
    p = prob_los_uma(d2d)
    return p * pl_los + (1 - p) * pl_nlos


def calcular_features(lat_t, lon_t, azimute, h_ant, lat_p, lon_p,
                      alt_terreno_torre=None, alt_p=None, usar_terreno=False):
    """Features e baseline para cada ponto. Aceita escalares ou arrays."""
    lat_p = np.asarray(lat_p, dtype=float)
    d2d = distancia_2d(lat_t, lon_t, lat_p, lon_p)
    brg = rumo(lat_t, lon_t, lat_p, lon_p)
    ang = np.abs(((brg - np.asarray(azimute) + 180) % 360) - 180)

    if usar_terreno:
        # altura do topo da antena em relação ao receptor
        h_rel = (np.asarray(alt_terreno_torre) + np.asarray(h_ant)) - np.asarray(alt_p)
        h_bs = h_rel + H_UT
    else:
        h_rel = np.broadcast_to(np.asarray(h_ant, dtype=float) - H_UT, d2d.shape)
        h_bs = np.broadcast_to(np.asarray(h_ant, dtype=float), d2d.shape)

    d3d = np.sqrt(np.maximum(d2d, 1.0) ** 2 + h_rel ** 2)
    return {
        "d2d": d2d,
        "d3d": d3d,
        "pl_base": pl_38901_uma(d2d, h_bs),
        "log10_d2d": np.log10(np.clip(d2d, 10, None)),
        "ang_off_az": ang,
        "h_rel_m": np.asarray(h_rel, dtype=float),
        "elev_graus": np.degrees(np.arctan2(h_rel, np.maximum(d2d, 1.0))),
    }


def montar_X(feats, nomes):
    return np.column_stack([np.asarray(feats[n], dtype=float) for n in nomes])


def checar_regiao(lat, lon):
    d = float(distancia_2d(CENTRO_JF[0], CENTRO_JF[1], lat, lon))
    if d > RAIO_MAX_JF_M:
        raise ValueError(f"Ponto a {d/1000:.1f} km de Juiz de Fora: fora da região de treino do modelo.")


def baseline_rsrp(modelo, f):
    """Baseline físico em dBm: 38.901 com P0 calibrado ou log-distância calibrado."""
    if modelo.get("baseline", "38901") == "logdist":
        return modelo["ld_a0"] + modelo["ld_a1"] * np.log10(f["d3d"])
    return modelo["p0_dbm"] - f["pl_base"]


def prever_rsrp(modelo, lat_t, lon_t, azimute, h_ant, lat_p, lon_p):
    """RSRP previsto (dBm) para pontos, com o modelo salvo (variante plano)."""
    f = calcular_features(lat_t, lon_t, azimute, h_ant, lat_p, lon_p, usar_terreno=False)
    X = montar_X(f, modelo["features"])
    return baseline_rsrp(modelo, f) + modelo["rf"].predict(X), f


def _resumo(rsrp, f, raio_m, abertura_graus):
    borda = (f["d2d"] >= 0.9 * raio_m) & (f["ang_off_az"] <= abertura_graus / 2)
    return {
        "rsrp_pico_dbm": round(float(np.max(rsrp)), 1),
        "rsrp_borda_dbm": round(float(np.median(rsrp[borda])), 1) if borda.any() else None,
        "area_boa_percentual": round(100 * float(np.mean(rsrp > LIMIAR_BOA)), 1),
        "area_mediana_percentual": round(100 * float(np.mean((rsrp <= LIMIAR_BOA) & (rsrp >= LIMIAR_MEDIANA))), 1),
    }


def simular(modelo, latitude, longitude, azimute, altura_antena_m=30.0,
            modo="setor_unico", raio_m=800, passo_m=25, abertura_graus=65):
    """Gera a resposta no formato do contrato de API combinado com o front."""
    checar_regiao(latitude, longitude)
    azs = [azimute % 360] if modo == "setor_unico" else [(azimute + k) % 360 for k in (0, 120, 240)]

    # grade de pontos dentro do círculo do raio analisado
    eixo = np.arange(-raio_m, raio_m + passo_m, passo_m)
    dx, dy = np.meshgrid(eixo, eixo)
    dentro = dx ** 2 + dy ** 2 <= raio_m ** 2
    dx, dy = dx[dentro], dy[dentro]
    lat_p = latitude + np.degrees(dy / R_TERRA)
    lon_p = longitude + np.degrees(dx / (R_TERRA * np.cos(np.radians(latitude))))

    setores, todos = [], []
    for az in azs:
        rsrp, f = prever_rsrp(modelo, latitude, longitude, az, altura_antena_m, lat_p, lon_p)
        todos.append(rsrp)
        s = {"azimute": round(float(az), 1), "abertura_graus": abertura_graus, "raio_analisado_m": raio_m}
        s.update(_resumo(rsrp, f, raio_m, abertura_graus))
        s["heatmap_pontos"] = [{"lat": round(float(a), 6), "lng": round(float(b), 6), "rsrp_dbm": round(float(r), 1)}
                               for a, b, r in zip(lat_p, lon_p, rsrp)]
        setores.append(s)

    resposta = {"torre": {"lat": latitude, "lng": longitude}, "setores": setores}
    if len(azs) > 1:  # melhor servidor entre os 3 setores
        melhor = np.max(np.vstack(todos), axis=0)
        resposta["combinado"] = {
            "rsrp_pico_dbm": round(float(melhor.max()), 1),
            "area_boa_percentual": round(100 * float(np.mean(melhor > LIMIAR_BOA)), 1),
            "area_mediana_percentual": round(100 * float(np.mean((melhor <= LIMIAR_BOA) & (melhor >= LIMIAR_MEDIANA))), 1),
        }
    return resposta
