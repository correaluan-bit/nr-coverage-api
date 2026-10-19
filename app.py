"""
app.py - backend do NR Coverage Predictor (FastAPI).
Local:  uvicorn app:app --reload   ->  http://127.0.0.1:8000/docs
"""
from typing import Literal

import joblib
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

import nr_modelo as nm

modelo = joblib.load("modelo_rf_nr.joblib")
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
    raio_m: int = Field(800, ge=100, le=2000)
    passo_m: int = Field(25, ge=10, le=100)
    # o agente (n8n) pede False: só os números, sem os milhares de pontos do mapa
    incluir_heatmap: bool = True


@app.get("/")
def status():
    return {"status": "ok", "modelo": modelo.get("baseline", "38901") + " + Random Forest",
            "treinado_em": modelo.get("treinado_em"), "n_locais_treino": modelo.get("n_locais")}


@app.post("/prever")
def prever(p: Pedido):
    if p.banda.lower() != "n78":
        raise HTTPException(422, "O modelo só cobre a banda n78.")
    try:
        r = nm.simular(modelo, p.latitude, p.longitude, p.azimute, p.altura_antena_m,
                       p.modo, p.raio_m, p.passo_m)
    except ValueError as e:  # fora de Juiz de Fora
        raise HTTPException(422, str(e))
    if not p.incluir_heatmap:
        for s in r["setores"]:
            s.pop("heatmap_pontos")
    return r
