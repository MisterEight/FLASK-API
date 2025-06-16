# -*- coding: utf-8 -*-
"""
train_credit_default_xgb.py · v3.2
=================================
*Versão final estável – early‑stopping apenas na CV; treino final fixo.*

Diferenças principais em relação à v3.1
--------------------------------------
1. **CV**: continua usando `early_stopping_rounds=50` para estimar o número
   ótimo de árvores por fold.
2. **Treino completo**: remove `early_stopping_rounds`; usa
   `n_estimators = round(média dos best_iterations)`. Assim evitamos o erro
   *"Must have at least 1 validation dataset"* e ficamos alinhados às boas
   práticas.
3. Mantém engenharia de features (PAY_AVG, UTIL_AVG, TOTAL_PAY, etc.) e limpeza
   de NaN/inf.

Uso:
-----
```bash
pip install pandas scikit-learn xgboost joblib
python train_credit_default_xgb.py
```
Gera `credit_default_xgb.json` + `credit_default_xgb.pkl`, prontos para servir
na API Flask.
"""

import pathlib
import statistics
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
DATA_PATH = pathlib.Path("UCI_Credit_Card.csv")
RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 800      # alto o suficiente; early‑stopping decide
EARLY_STOPPING = 50
EPS = 1e-6

# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError("UCI_Credit_Card.csv não encontrado.")
print("📊  Lendo CSV …")
df = pd.read_csv(DATA_PATH)
print(f"✔️  Linhas: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. Engenharia de features
# ---------------------------------------------------------------------------
PAY_COLS = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 6 + 1)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 6 + 1)]

print("➕  Features derivadas …")
df["PAY_AVG"] = df[PAY_COLS].mean(axis=1)
DROP_PAY_COLS = True
if DROP_PAY_COLS:
    df.drop(columns=PAY_COLS, inplace=True)

for col in BILL_COLS:
    df[f"UTIL_{col[-1]}"] = df[col] / (df["LIMIT_BAL"] + EPS)
df["UTIL_AVG"] = df[[f"UTIL_{i}" for i in range(1, 7)]].mean(axis=1)

df["TOTAL_PAY"] = df[PAY_AMT_COLS].sum(axis=1)
df["PAYMENT_RATIO"] = df["TOTAL_PAY"] / (df[BILL_COLS].sum(axis=1) + 1)

# log1p nas faturas (após clip >=0)
df[BILL_COLS] = df[BILL_COLS].clip(lower=0)
df[BILL_COLS] = np.log1p(df[BILL_COLS])

# Sanitizar NaN/inf
print("🧹  Limpando NaN/inf …")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ---------------------------------------------------------------------------
TARGET = "default.payment.next.month"
ID_COL = "ID"

print(df.columns)

X = df[["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "PAY_AVG",'UTIL_1', 'UTIL_2', 'UTIL_3',
       'UTIL_4', 'UTIL_5', 'UTIL_6', 'UTIL_AVG', 'TOTAL_PAY', 'PAYMENT_RATIO']]
y = df[TARGET]
feature_names: List[str] = X.columns.tolist()
print(f"📈  Features: {len(feature_names)}")

neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / pos

base_params = dict(
    objective="binary:logistic",
    eval_metric="auc",         # agora no construtor
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=EARLY_STOPPING,
)

# ---------------------------------------------------------------------------
# 3. Stratified K‑Fold CV
# ---------------------------------------------------------------------------
print("🚀  CV …")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
auc_scores, best_iters = [], []

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    model = xgb.XGBClassifier(**base_params)
    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[va], y.iloc[va])],
        verbose=False,
    )
    best_iters.append(model.best_iteration)
    auc = roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1])
    auc_scores.append(auc)
    print(f"fold {fold}: best_iter = {model.best_iteration:4d}, AUC = {auc:.4f}")

mean_auc = statistics.mean(auc_scores)
std_auc = statistics.stdev(auc_scores)
mean_best_iter = int(statistics.mean(best_iters))
print(f"✅  CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")
print(f"🏆  n_estimators final = {mean_best_iter}")

# ---------------------------------------------------------------------------
# 4. Treino final sem early‑stopping
# ---------------------------------------------------------------------------
final_params = base_params.copy()
final_params.pop("early_stopping_rounds", None)
final_params["n_estimators"] = mean_best_iter

print("🏁  Treinando modelo completo …")
final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X, y, verbose=False)

# ---------------------------------------------------------------------------
# 5. Salvar artefatos
# ---------------------------------------------------------------------------
print("💾  Salvando booster JSON …")
final_model.get_booster().save_model("credit_default_xgb.json")

try:
    import joblib

    joblib.dump(final_model, "credit_default_xgb.pkl")
    print("💾  credit_default_xgb.pkl salvo.")
except ImportError:
    print("⚠️  joblib ausente; .pkl não gerado.")

print("🎉  Concluído!")
