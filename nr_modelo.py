"""
nr_modelo.py - código compartilhado entre o TREINO e o BACKEND.

As features são calculadas pela mesma função nos dois lugares, para o modelo
não ver no ar algo diferente do que viu no treino.

Modelo v2 (com relevo): RSRP = baseline físico + RF(features)
  - baseline "fisica_relevo": a0 + a1*log10(d3D) + a2*L_difracao, calibrado nos dados
  - L_difracao: perda por difração no obstáculo dominante do perfil de terreno
    (gume de faca, ITU-R P.526), calculada com o DEM Copernicus GLO-30
  - RF: Random Forest no resíduo (medido - baseline)
Também aceita o modelo v1 (sem DEM), para compatibilidade.
"""
import numpy as np

FC_GHZ = 3.5
H_UT = 1.5
C_LUZ = 3e8
LAMBDA_M = C_LUZ / (FC_GHZ * 1e9)
R_TERRA = 6371000.0
K_TERRA = 4 / 3            # raio efetivo da Terra (refração padrão)
PASSO_PERFIL_M = 25.0      # espaçamento das amostras do perfil de terreno
MARGEM_PONTA_M = 30.0      # ignora o terreno colado na torre e no receptor

FEATURES_PLANO = ["log10_d2d", "ang_off_az"]
FEATURES_DEM = ["log10_d2d", "ang_off_az", "h_rel_m", "elev_graus", "difracao_db", "h_obst_m"]

LIMIAR_BOA = -90.0
LIMIAR_MEDIANA = -110.0
CENTRO_JF = (-21.7642, -43.3496)
RAIO_MAX_JF_M = 30000


# ============================ DEM ============================
class DEM:
    """Relevo de Juiz de Fora. 'superficie' = DSM (inclui parte de prédios/árvores),
    usada para obstáculos; 'chao' = mínimo 3x3 da superfície, aproximação do solo,
    usada para a base da torre e para o receptor."""

    def __init__(self, caminho="dem_jf.npz"):
        z = np.load(caminho)
        self.sup = z["elev_dm"].astype(np.float32) / 10
        self.lat0, self.lon0 = float(z["lat0"]), float(z["lon0"])
        self.dlat, self.dlon = float(z["dlat"]), float(z["dlon"])
        p = np.pad(self.sup, 1, mode="edge")
        self.chao_grade = np.min([p[i:i + self.sup.shape[0], j:j + self.sup.shape[1]]
                                  for i in range(3) for j in range(3)], axis=0)
        self.lat_min = self.lat0 - (self.sup.shape[0] - 1) * self.dlat
        self.lon_max = self.lon0 + (self.sup.shape[1] - 1) * self.dlon

    def cobre(self, lat, lon):
        lat, lon = np.asarray(lat), np.asarray(lon)
        return (lat <= self.lat0) & (lat >= self.lat_min) & (lon >= self.lon0) & (lon <= self.lon_max)

    def _bilinear(self, g, lat, lon):
        r = np.clip((self.lat0 - np.asarray(lat)) / self.dlat, 0, g.shape[0] - 1.001)
        c = np.clip((np.asarray(lon) - self.lon0) / self.dlon, 0, g.shape[1] - 1.001)
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        return (g[r0, c0] * (1 - fr) * (1 - fc) + g[r0 + 1, c0] * fr * (1 - fc)
                + g[r0, c0 + 1] * (1 - fr) * fc + g[r0 + 1, c0 + 1] * fr * fc)

    def superficie(self, lat, lon):
        return self._bilinear(self.sup, lat, lon)

    def chao(self, lat, lon):
        return self._bilinear(self.chao_grade, lat, lon)


# ======================= geometria =======================
def distancia_2d(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TERRA * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def rumo(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def perda_gume_faca(nu):
    """ITU-R P.526: perda (dB) de um obstáculo em gume de faca."""
    nu = np.asarray(nu, dtype=float)
    j = 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    return np.where(nu > -0.78, j, 0.0)


def _perfil(dem, lat_t, lon_t, h_tx, lat_p, lon_p, h_rx, d2d, bloco=2000):
    """Para cada caminho torre->ponto: maior obstrução acima da linha de visada (m)
    e perda por difração do obstáculo dominante (dB)."""
    n = len(d2d)
    h_obst = np.full(n, -50.0)
    nu_max = np.full(n, -5.0)
    k = int(np.ceil(np.max(d2d) / PASSO_PERFIL_M)) + 2 if n else 2
    t = np.linspace(0, 1, max(k, 3))[1:-1]
    for i in range(0, n, bloco):
        s = slice(i, i + bloco)
        lat_s = lat_t[s, None] + (lat_p[s] - lat_t[s])[:, None] * t
        lon_s = lon_t[s, None] + (lon_p[s] - lon_t[s])[:, None] * t
        z = dem.superficie(lat_s, lon_s)
        d1 = d2d[s, None] * t
        d2 = d2d[s, None] - d1
        linha = h_tx[s, None] + (h_rx[s] - h_tx[s])[:, None] * t
        h = z + d1 * d2 / (2 * K_TERRA * R_TERRA) - linha
        ok = (d1 > MARGEM_PONTA_M) & (d2 > MARGEM_PONTA_M)
        nu = h * np.sqrt(2 * (d1 + d2) / (LAMBDA_M * np.maximum(d1, 1) * np.maximum(d2, 1)))
        tem = ok.any(axis=1)
        h_obst[s] = np.where(tem, np.max(np.where(ok, h, -np.inf), axis=1), -50.0)
        nu_max[s] = np.where(tem, np.max(np.where(ok, nu, -np.inf), axis=1), -5.0)
    return np.clip(h_obst, -50, 300), perda_gume_faca(nu_max)


def calcular_features(lat_t, lon_t, azimute, h_ant, lat_p, lon_p, dem=None):
    """Features e grandezas físicas de cada ponto. Aceita escalares ou arrays."""
    lat_p = np.atleast_1d(np.asarray(lat_p, dtype=float))
    lon_p = np.atleast_1d(np.asarray(lon_p, dtype=float))
    lat_t = np.broadcast_to(np.asarray(lat_t, dtype=float), lat_p.shape)
    lon_t = np.broadcast_to(np.asarray(lon_t, dtype=float), lat_p.shape)
    h_ant = np.broadcast_to(np.asarray(h_ant, dtype=float), lat_p.shape)

    d2d = distancia_2d(lat_t, lon_t, lat_p, lon_p)
    ang = np.abs(((rumo(lat_t, lon_t, lat_p, lon_p) - np.asarray(azimute) + 180) % 360) - 180)
    f = {"d2d": d2d, "log10_d2d": np.log10(np.clip(d2d, 10, None)), "ang_off_az": ang}

    if dem is None:
        h_rel = h_ant - H_UT
        f["difracao_db"] = np.zeros_like(d2d)
        f["h_obst_m"] = np.full_like(d2d, -50.0)
    else:
        h_tx = dem.chao(lat_t, lon_t) + h_ant          # topo da antena (m acima do mar)
        h_rx = dem.chao(lat_p, lon_p) + H_UT           # receptor
        h_rel = h_tx - h_rx                            # torre no morro: grande; no vale: pequeno/negativo
        f["h_obst_m"], f["difracao_db"] = _perfil(dem, lat_t, lon_t, h_tx, lat_p, lon_p, h_rx, d2d)

    f["h_rel_m"] = np.asarray(h_rel, dtype=float)
    f["elev_graus"] = np.degrees(np.arctan2(h_rel, np.maximum(d2d, 1.0)))
    f["d3d"] = np.sqrt(np.maximum(d2d, 1.0) ** 2 + h_rel ** 2)
    return f


def montar_X(feats, nomes):
    return np.column_stack([np.asarray(feats[n], dtype=float) for n in nomes])


def baseline_rsrp(modelo, f):
    b = modelo.get("baseline", "38901")
    if b == "fisica_relevo":
        a0, a1, a2 = modelo["coef_fisica"]
        return a0 + a1 * np.log10(f["d3d"]) + a2 * f["difracao_db"]
    if b == "logdist":
        return modelo["ld_a0"] + modelo["ld_a1"] * np.log10(f["d3d"])
    raise ValueError(f"baseline desconhecido: {b}")


def prever_rsrp(modelo, lat_t, lon_t, azimute, h_ant, lat_p, lon_p, dem=None):
    f = calcular_features(lat_t, lon_t, azimute, h_ant, lat_p, lon_p,
                          dem=dem if modelo.get("usa_dem") else None)
    return baseline_rsrp(modelo, f) + modelo["rf"].predict(montar_X(f, modelo["features"])), f


# ======================= simulação (backend) =======================
def _resumo(rsrp, f, raio_m, abertura_graus):
    borda = (f["d2d"] >= 0.9 * raio_m) & (f["ang_off_az"] <= abertura_graus / 2)
    return {
        "rsrp_pico_dbm": round(float(np.max(rsrp)), 1),
        "rsrp_borda_dbm": round(float(np.median(rsrp[borda])), 1) if borda.any() else None,
        "area_boa_percentual": round(100 * float(np.mean(rsrp > LIMIAR_BOA)), 1),
        "area_mediana_percentual": round(100 * float(np.mean((rsrp <= LIMIAR_BOA) & (rsrp >= LIMIAR_MEDIANA))), 1),
    }


def simular(modelo, latitude, longitude, azimute, altura_antena_m=30.0,
            modo="setor_unico", raio_m=800, passo_m=25, abertura_graus=65, dem=None):
    """Resposta no formato do contrato de API combinado com o front."""
    if float(distancia_2d(CENTRO_JF[0], CENTRO_JF[1], latitude, longitude)) > RAIO_MAX_JF_M:
        raise ValueError("Ponto fora da região de Juiz de Fora, onde o modelo foi treinado.")
    azs = [azimute % 360] if modo == "setor_unico" else [(azimute + k) % 360 for k in (0, 120, 240)]

    eixo = np.arange(-raio_m, raio_m + passo_m, passo_m)
    dx, dy = np.meshgrid(eixo, eixo)
    dentro = dx ** 2 + dy ** 2 <= raio_m ** 2
    dx, dy = dx[dentro], dy[dentro]
    lat_p = latitude + np.degrees(dy / R_TERRA)
    lon_p = longitude + np.degrees(dx / (R_TERRA * np.cos(np.radians(latitude))))
    if modelo.get("usa_dem") and not np.all(dem.cobre(lat_p, lon_p)):
        raise ValueError("A área analisada sai do mapa de relevo carregado (Juiz de Fora e arredores).")

    setores, todos = [], []
    for az in azs:
        rsrp, f = prever_rsrp(modelo, latitude, longitude, az, altura_antena_m, lat_p, lon_p, dem=dem)
        todos.append(rsrp)
        s = {"azimute": round(float(az), 1), "abertura_graus": abertura_graus, "raio_analisado_m": raio_m}
        s.update(_resumo(rsrp, f, raio_m, abertura_graus))
        s["heatmap_pontos"] = [{"lat": round(float(a), 6), "lng": round(float(b), 6), "rsrp_dbm": round(float(r), 1)}
                               for a, b, r in zip(lat_p, lon_p, rsrp)]
        setores.append(s)

    resposta = {"torre": {"lat": latitude, "lng": longitude}, "setores": setores}
    if len(azs) > 1:
        melhor = np.max(np.vstack(todos), axis=0)
        resposta["combinado"] = {
            "rsrp_pico_dbm": round(float(melhor.max()), 1),
            "area_boa_percentual": round(100 * float(np.mean(melhor > LIMIAR_BOA)), 1),
            "area_mediana_percentual": round(100 * float(np.mean((melhor <= LIMIAR_BOA) & (melhor >= LIMIAR_MEDIANA))), 1),
        }
    return resposta
