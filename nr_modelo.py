"""
nr_modelo.py - código compartilhado entre o TREINO e o BACKEND.

As features são calculadas pela mesma função nos dois lugares, para o modelo
não ver no ar algo diferente do que viu no treino.

Modelo v2 (com relevo): RSRP = baseline físico + RF(features)
  - baseline "fisica_relevo": a0 + a1*log10(d3D) + a2*L_difracao + a3*G_antena, calibrado nos dados
  - G_antena: padrão horizontal de antena setorial 3GPP (65°), dá o formato do setor
  - baseline "fisica_38901": P0 - PL_38901_UMa(d, h) + a2*L_difracao + a3*G_antena + G_vertical,
    com a inclinação da distância e o padrão vertical (downtilt) FIXOS pela física (3GPP);
    só o nível P0 e os pesos de difração e do padrão horizontal são calibrados
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


def prob_los_uma(d2d):
    """Probabilidade de LOS, 3GPP TR 38.901 Tabela 7.4.2-1 (UMa, h_UT <= 13 m)."""
    d = np.maximum(d2d, 1e-6)
    return np.where(d2d <= 18, 1.0, 18 / d + np.exp(-d / 63) * (1 - 18 / d))


def pl_38901_uma(d2d, h_bs, h_ut=H_UT, fc_ghz=FC_GHZ):
    """Perda de percurso esperada (dB), 38.901 Tabela 7.4.1-1, UMa: LOS e NLOS
    ponderados pela probabilidade de LOS. Inclinação física: ~22 dB/década perto
    da torre (LOS), ~39 dB/década em NLOS."""
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


def ganho_padrao_h(ang_graus, abertura=65.0, a_max=30.0):
    """Padrão horizontal de antena setorial (3GPP TR 38.901, Tabela 7.3-1), em dB relativo ao eixo."""
    return -np.minimum(12 * (np.asarray(ang_graus) / abertura) ** 2, a_max)


# padrão vertical de antena de macrocélula (3GPP TR 36.873): feixe estreito com downtilt.
# Logo embaixo da torre o receptor fica fora do feixe vertical e o sinal cai.
TILT_GRAUS = 6.0
ABERTURA_V_GRAUS = 10.0
SLA_V_DB = 20.0


def ganho_padrao_v(elev_graus, tilt=TILT_GRAUS, abertura=ABERTURA_V_GRAUS, sla=SLA_V_DB):
    """Atenuação vertical (dB, <= 0) para um receptor elev_graus abaixo do horizonte da antena."""
    return -np.minimum(12 * ((np.asarray(elev_graus) - tilt) / abertura) ** 2, sla)


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
    f = {"d2d": d2d, "log10_d2d": np.log10(np.clip(d2d, 10, None)), "ang_off_az": ang,
         "ganho_antena_db": ganho_padrao_h(ang)}

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
    f["pl_38901"] = pl_38901_uma(d2d, f["h_rel_m"] + H_UT)
    f["ganho_vertical_db"] = ganho_padrao_v(f["elev_graus"])
    f["d3d"] = np.sqrt(np.maximum(d2d, 1.0) ** 2 + h_rel ** 2)
    return f


def montar_X(feats, nomes):
    return np.column_stack([np.asarray(feats[n], dtype=float) for n in nomes])


def baseline_rsrp(modelo, f):
    b = modelo.get("baseline", "38901")
    if b == "fisica_relevo":
        c = modelo["coef_fisica"]
        r = c[0] + c[1] * np.log10(f["d3d"]) + c[2] * f["difracao_db"]
        return r + c[3] * f["ganho_antena_db"] if len(c) > 3 else r
    if b == "fisica_38901":
        c = modelo["coef_fisica"]
        return (c[0] - f["pl_38901"] + c[1] * f["difracao_db"] + c[2] * f["ganho_antena_db"]
                + f["ganho_vertical_db"])
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


# ======================= modo cidade (até 10 torres) =======================
# Ruído térmico por elemento de recurso (SCS 30 kHz) com figura de ruído de 7 dB: ~ -122 dBm.
RUIDO_RE_DBM = -174.0 + 10 * np.log10(30e3) + 7.0
LIMIAR_RELEVANTE_DBM = -110.0   # um setor só "chega" a um ponto acima disso
SINR_BAIXA_DB = 13.0            # acima: interferência baixa
SINR_ALTA_DB = 0.0              # abaixo: interferência alta
MAX_PONTOS_CIDADE = 15000   # mantém a resposta leve e rápida no servidor gratuito
FAIXAS_RSRP = [("excelente", -80.0, np.inf), ("bom", -90.0, -80.0), ("regular", -100.0, -90.0),
               ("fraco", -110.0, -100.0), ("sem_cobertura", -np.inf, -110.0)]


def simular_cidade(modelo, torres, raio_m=2000, passo_m=None, dem=None, incluir_pontos=True):
    """Várias torres na MESMA grade: RSRP da melhor torre e SINR em cada ponto.
    torres: lista de dicts com latitude, longitude, azimute, altura_antena_m, modo.
    Premissa: todas as torres são da mesma operadora e usam a mesma frequência (interferência co-canal)."""
    if not 1 <= len(torres) <= 10:
        raise ValueError("O modo cidade aceita de 1 a 10 torres.")
    for i, t in enumerate(torres, 1):
        if float(distancia_2d(CENTRO_JF[0], CENTRO_JF[1], t["latitude"], t["longitude"])) > RAIO_MAX_JF_M:
            raise ValueError(f"A torre {i} está fora da região de Juiz de Fora, onde o modelo foi treinado.")

    # grade comum em metros, ao redor de todas as torres
    lat_c = float(np.mean([t["latitude"] for t in torres]))
    lon_c = float(np.mean([t["longitude"] for t in torres]))
    k_lon = R_TERRA * np.cos(np.radians(lat_c))
    tx = np.array([np.radians(t["longitude"] - lon_c) * k_lon for t in torres])
    ty = np.array([np.radians(t["latitude"] - lat_c) * R_TERRA for t in torres])
    x0, x1, y0, y1 = tx.min() - raio_m, tx.max() + raio_m, ty.min() - raio_m, ty.max() + raio_m
    area_util = len(torres) * np.pi * raio_m ** 2
    passo = max(passo_m or 0, 25, int(np.ceil(np.sqrt(min((x1 - x0) * (y1 - y0), area_util) / MAX_PONTOS_CIDADE) / 5) * 5))
    gx, gy = np.meshgrid(np.arange(x0, x1 + passo, passo), np.arange(y0, y1 + passo, passo))
    gx, gy = gx.ravel(), gy.ravel()
    dist = np.sqrt((gx[None, :] - tx[:, None]) ** 2 + (gy[None, :] - ty[:, None]) ** 2)   # torres x pontos
    manter = (dist <= raio_m).any(axis=0)
    gx, gy, dist = gx[manter], gy[manter], dist[:, manter]
    lat_p = lat_c + np.degrees(gy / R_TERRA)
    lon_p = lon_c + np.degrees(gx / k_lon)
    if modelo.get("usa_dem") and not np.all(dem.cobre(lat_p, lon_p)):
        raise ValueError("A área analisada sai do mapa de relevo carregado (Juiz de Fora e arredores).")

    # RSRP de cada setor de cada torre (o perfil de terreno é calculado uma vez por torre)
    dem_uso = dem if modelo.get("usa_dem") else None
    rsrp_setores, torre_do_setor = [], []
    for i, t in enumerate(torres):
        azs = ([t["azimute"] % 360] if t.get("modo", "torre_completa") == "setor_unico"
               else [(t["azimute"] + k) % 360 for k in (0, 120, 240)])
        idx = np.where(dist[i] <= raio_m)[0]
        f = calcular_features(t["latitude"], t["longitude"], azs[0], t.get("altura_antena_m", 30.0),
                              lat_p[idx], lon_p[idx], dem=dem_uso)
        brg = rumo(t["latitude"], t["longitude"], lat_p[idx], lon_p[idx])
        for az in azs:
            g = dict(f)
            g["ang_off_az"] = np.abs(((brg - az + 180) % 360) - 180)
            g["ganho_antena_db"] = ganho_padrao_h(g["ang_off_az"])
            r = np.full(len(lat_p), -np.inf)
            r[idx] = baseline_rsrp(modelo, g) + modelo["rf"].predict(montar_X(g, modelo["features"]))
            rsrp_setores.append(r)
            torre_do_setor.append(i)
    S = np.vstack(rsrp_setores)                     # setores x pontos
    torre_do_setor = np.array(torre_do_setor)

    ordem = np.argsort(-S, axis=0)
    melhor_idx, segundo_idx = ordem[0], (ordem[1] if len(S) > 1 else ordem[0])
    melhor = S[melhor_idx, np.arange(S.shape[1])]
    torre_serv = torre_do_setor[melhor_idx]
    # interferência = setores das OUTRAS torres (a ideia do modo cidade é a interseção entre torres);
    # os setores da própria torre servidora não entram (o padrão efetivo calibrado subestima a separação entre eles)
    lin = np.where(np.isfinite(S), 10 ** (S / 10), 0.0)
    da_servidora = torre_do_setor[:, None] == torre_serv[None, :]
    interf = np.where(da_servidora, 0.0, lin).sum(axis=0)
    sinr = melhor - 10 * np.log10(interf + 10 ** (RUIDO_RE_DBM / 10))
    # quantas TORRES chegam ao ponto acima do limiar
    T = np.vstack([S[torre_do_setor == i].max(axis=0) for i in range(len(torres))])
    n_chegam = (T >= LIMIAR_RELEVANTE_DBM).sum(axis=0)
    sobrepoe = n_chegam >= 2
    # segunda torre mais forte (para identificar o par que disputa o ponto)
    T_sem = np.where(np.arange(len(torres))[:, None] == torre_serv[None, :], -np.inf, T)
    torre_seg = np.argmax(T_sem, axis=0) if len(torres) > 1 else torre_serv
    classe = np.where(~sobrepoe, "nenhuma", np.where(sinr > SINR_BAIXA_DB, "baixa",
                      np.where(sinr >= SINR_ALTA_DB, "media", "alta")))
    intensidade = np.where(sobrepoe, np.clip((SINR_BAIXA_DB - sinr) / (SINR_BAIXA_DB + 10.0), 0, 1), 0.0)

    # resumo para o painel e para o Diagnóstico IA
    n = len(melhor)
    pct = lambda m: round(100 * float(np.mean(m)), 1)
    faixas = {nome: pct((melhor >= lo) & (melhor < hi)) for nome, lo, hi in FAIXAS_RSRP}
    interf_pct = {c: pct(classe == c) for c in ("nenhuma", "baixa", "media", "alta")}
    pares = {}
    for k in np.where(np.isin(classe, ["media", "alta"]))[0]:
        par = tuple(sorted((int(torre_serv[k]) + 1, int(torre_seg[k]) + 1)))
        pares.setdefault(par, []).append(k)
    lista_pares = []
    for par, ks in pares.items():
        ks = np.array(ks)
        alta = ks[classe[ks] == "alta"]
        foco = alta if len(alta) else ks
        lista_pares.append({
            "torres": list(par),
            "area_com_interferencia_pct": pct(np.isin(np.arange(n), ks)),
            "area_interferencia_alta_pct": round(100 * len(alta) / n, 1),
            "sinr_medio_db": round(float(np.mean(sinr[ks])), 1),
            "ponto_critico": {"lat": round(float(lat_p[foco].mean()), 6), "lng": round(float(lon_p[foco].mean()), 6)},
        })
    lista_pares.sort(key=lambda p: (p["area_interferencia_alta_pct"], p["area_com_interferencia_pct"]), reverse=True)
    por_torre = [{"torre": i + 1, "area_servida_pct": pct(torre_serv == i),
                  "rsrp_pico_dbm": round(float(S[torre_do_setor == i].max()), 1)} for i in range(len(torres))]
    resumo = {
        "n_torres": len(torres), "raio_m": raio_m, "passo_m": passo, "n_pontos": n,
        "premissa": "Torres da mesma operadora na mesma frequência (interferência co-canal).",
        "faixas_rsrp_pct": faixas,
        "interferencia_pct": interf_pct,
        "sinr_mediano_sobreposicao_db": round(float(np.median(sinr[sobrepoe])), 1) if sobrepoe.any() else None,
        "pares_criticos": lista_pares[:3],
        "por_torre": por_torre,
    }
    resposta = {"torres": [{"torre": i + 1, **t} for i, t in enumerate(torres)], "resumo": resumo}
    if incluir_pontos:
        resposta["pontos"] = [
            {"lat": round(float(a), 6), "lng": round(float(b), 6), "rsrp_dbm": round(float(r), 1),
             "sinr_db": round(float(s), 1), "interferencia": str(c), "intensidade": round(float(it), 2),
             "torre": int(ts) + 1}
            for a, b, r, s, c, it, ts in zip(lat_p, lon_p, melhor, sinr, classe, intensidade, torre_serv)]
    return resposta
