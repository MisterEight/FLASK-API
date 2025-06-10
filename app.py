from __future__ import annotations

import json
import pathlib
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from dotenv import load_dotenv
from typing import Any, Dict, List
from flask import Flask, jsonify, request
from sqlalchemy import Boolean, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker



load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
PORT = os.getenv("PORT", "3307")
DB_NAME = os.getenv("DB_NAME")

# -----------------------------------------------------------------------------
# Config – Banco de dados (ajuste conforme ambiente)
# -----------------------------------------------------------------------------
DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}"
engine = create_engine(
    DATABASE_URI,
    echo=False,
    pool_pre_ping=True,   # checa o socket antes de usar
    pool_recycle=1800,    # força reconexão a cada 30 min
    connect_args={"connect_timeout": 10}
)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class PredicaoDefault(Base):
    __tablename__ = "predicoes_default"

    id = Column(Integer, primary_key=True)
    prob_default = Column(Float)
    score = Column(Float)
    approved = Column(Boolean)

# Cria a tabela caso não exista
Base.metadata.create_all(engine)

# -----------------------------------------------------------------------------
# Carrega booster e obtém feature names
# -----------------------------------------------------------------------------
BOOSTER_PATH = pathlib.Path("credit_default_xgb.json")
if not BOOSTER_PATH.exists():
    raise FileNotFoundError(
        "credit_default_xgb.json não encontrado – gere com train_credit_default_xgb.py")

booster = xgb.Booster()
booster.load_model(str(BOOSTER_PATH))
FEATURE_NAMES: List[str] = booster.feature_names  # ordem absoluta usada no treino

# Threshold de aprovação (<= 50% de risco → aprovado)
THR = 0.5

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/analisa_emprestimo", methods=["POST"])
def predict_default():
    """Recebe JSON com features; devolve probabilidade de default e score."""
    data: Dict[str, Any] | None = request.get_json(silent=True)
    if not data:
        return jsonify({"erro": "JSON inválido ou ausente"}), 400

    # ---- Garantir presença das features ----
    missing = [f for f in FEATURE_NAMES if f not in data]
    # PAY_AVG pode ser calculado se faltar
    if "PAY_AVG" in missing:
        missing.remove("PAY_AVG")
    if missing:
        return jsonify({"erro": f"Faltam features: {missing}"}), 400

    # Copia para DataFrame (garante coerência de tipos)
    df = pd.DataFrame([data])

    # Cria PAY_AVG se não veio
    if "PAY_AVG" not in df.columns:
        if not all(col in df.columns for col in PAY_COLS):
            return jsonify({"erro": "Necessário fornecer PAY_AVG ou todas as PAY_*"}), 400
        df["PAY_AVG"] = df[PAY_COLS].mean(axis=1)

    # Se o payload contém PAY_* e o modelo foi treinado SEM elas, simplesmente
    # dropamos para evitar coluna extra – o booster ignora colunas desconhecidas
    df = df[[c for c in FEATURE_NAMES]]  # garante ordem / remove extras

    # Inferência XGBoost
    dmat = xgb.DMatrix(df.values, feature_names=FEATURE_NAMES)
    prob_default_arr = booster.predict(dmat)
    prob_default = float(prob_default_arr[0])

    score = round((1.0 - prob_default) * 100.0, 2)
    approved = prob_default < THR

    # Persistência no banco
    session = Session()
    try:
        rec = PredicaoDefault(prob_default=prob_default, score=score, approved=approved)
        session.add(rec)
        session.commit()
    except Exception as exc:  # pragma: no cover
        session.rollback()
        app.logger.error("DB error: %s", exc)
    finally:
        session.close()

    return jsonify({
        "prob_default": round(prob_default, 4),
        "score": score,
        "approved": approved,
    })

# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)