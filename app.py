"""
app.py - backend do NR Coverage Predictor (FastAPI).
Local:  uvicorn app:app --reload   ->  http://127.0.0.1:8000/docs
"""
from typing import List, Literal, Optional

import joblib
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

import nr_modelo as nm

modelo = joblib.load("modelo_rf_nr.joblib")
dem = nm.DEM("dem_jf.npz") if modelo.get("usa_dem") else None  # relevo de Juiz de Fora
if modelo.get("sklearn_versao") != sklearn.__version__:
    print(f"AVISO: modelo treinado com scikit-learn {modelo.get('sklearn_versao')}, "
          f"servidor usa {sklearn.__version__}. Fixe a mesma versão no requirements.txt.")

app = FastAPI(title="NR Coverage Predictor API")
# o front do Lovable chama direto do navegador, então precisa de CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Pedido(BaseModel):
    model_config = ConfigDict(extra="ignore")  # aceita campos a mais (ex.: "setorizado")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    banda: str = "n78"
    azimute: float = Field(ge=0, le=360)
    altura_antena_m: float = Field(30, ge=5, le=100)
    modo: Literal["setor_unico", "torre_completa"] = "setor_unico"
    raio_m: int = Field(2000, ge=100, le=5000)
    # espaçamento da grade; se não vier, ~raio/40 (uns 5 mil pontos por setor)
    passo_m: Optional[int] = Field(None, ge=10, le=200)
    # o agente (n8n) pede False: só os números, sem os milhares de pontos do mapa
    incluir_heatmap: bool = True


class TorreCidade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    azimute: float = Field(0, ge=0, le=360)
    altura_antena_m: float = Field(30, ge=5, le=100)
    modo: Literal["setor_unico", "torre_completa"] = "torre_completa"


class PedidoCidade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    torres: List[TorreCidade] = Field(min_length=1, max_length=10)
    raio_m: int = Field(2000, ge=500, le=3000)
    passo_m: Optional[int] = Field(None, ge=25, le=200)
    # o agente/Diagnóstico pede False: só o resumo, sem os milhares de pontos
    incluir_pontos: bool = True


@app.get("/")
def status():
    return {"status": "ok", "modelo": modelo.get("baseline", "38901") + " + Random Forest",
            "usa_relevo": bool(modelo.get("usa_dem")),
            "treinado_em": modelo.get("treinado_em"), "n_locais_treino": modelo.get("n_locais")}


@app.post("/prever")
def prever(p: Pedido):
    if p.banda.lower() != "n78":
        raise HTTPException(422, "O modelo só cobre a banda n78.")
    try:
        r = nm.simular(modelo, p.latitude, p.longitude, p.azimute, p.altura_antena_m,
                       p.modo, p.raio_m, p.passo_m or max(25, round(p.raio_m / 40)), dem=dem)
    except ValueError as e:  # fora de Juiz de Fora ou do mapa de relevo
        raise HTTPException(422, str(e))
    if not p.incluir_heatmap:
        for s in r["setores"]:
            s.pop("heatmap_pontos")
    return r


@app.post("/prever_cidade")
def prever_cidade(p: PedidoCidade):
    """Modo cidade: até 10 torres na mesma grade, com RSRP da melhor torre e SINR (interferência)."""
    try:
        return nm.simular_cidade(modelo, [t.model_dump() for t in p.torres], p.raio_m, p.passo_m,
                                 dem=dem, incluir_pontos=p.incluir_pontos)
    except ValueError as e:
        raise HTTPException(422, str(e))
