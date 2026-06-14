# -*- coding: utf-8 -*-
"""Aday cesitli modelleri tara: mevcut 4-model ensemble'ina EKLENINCE weighted-OOF
(LB'yi izleyen metrik) dusuyor mu? Ayni 10-fold split kullanilir (pipeline ile tutarli)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
tX = train.drop(columns=[P.TARGET]); comb = pd.concat([tX, test], ignore_index=True)
comb = P.create_features(comb); comb = P.encode_categoricals(comb)
code = ["university_tier_code"] + [f"{c}_code" for c in P.CAT_COLS if c != "university_tier"]
num = [c for c in comb.columns if c not in P.CAT_COLS + [P.ID_COL, P.TEXT_COL]
       and comb[c].dtype != object and c not in code]
feat = num + code
Xall = comb[feat].reset_index(drop=True)
X = Xall.iloc[:len(tX)].reset_index(drop=True)

# test-year importance weights (LB-tracking metric)
tw = test["application_year"].value_counts(normalize=True)
trw = train["application_year"].value_counts(normalize=True)
W = train["application_year"].map(tw / trw).values; W = W / W.mean()
def wmse(t, p, w): return float(np.average((t - p) ** 2, weights=w))

bins = pd.qcut(pd.Series(y), q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(10, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y)), bins))

def cv_oof(make, needs_prep=False):
    oof = np.zeros(len(y))
    for tr, va in folds:
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        if needs_prep:
            imp = SimpleImputer(strategy="median").fit(Xtr)
            sc = StandardScaler().fit(imp.transform(Xtr))
            Xtr2 = sc.transform(imp.transform(Xtr)); Xva2 = sc.transform(imp.transform(Xva))
        else:
            Xtr2, Xva2 = Xtr, Xva
        m = make(); m.fit(Xtr2, y[tr]); oof[va] = m.predict(Xva2)
    return oof

cands = {
 "histgb": (lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03,
              max_leaf_nodes=63, l2_regularization=2.0, random_state=42), False),
 "extratrees": (lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
              n_jobs=-1, random_state=42), True),
 "mlp": (lambda: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
              learning_rate_init=2e-3, max_iter=120, early_stopping=True, random_state=42), True),
 "ridge_full": (lambda: Ridge(alpha=20.0), True),
}
cand_oof = {}
for name, (make, prep) in cands.items():
    o = cv_oof(make, prep)
    cand_oof[name] = o
    print(f"  {name:11s} solo weighted-OOF = {wmse(y, o, W):.3f}")

# mevcut 4-model OOF
oofdf = pd.read_csv("pipeline_oof_backup.csv" if False else "oof_predictions.csv")
base_models = ["catboost", "lgbm", "xgb", "ridge"]
Pbase = np.column_stack([oofdf[f"oof_{m}"].values for m in base_models])

def best_blend(P_, w):
    n = P_.shape[1]; w0 = np.full(n, 1 / n)
    cons = ({"type": "eq", "fun": lambda v: v.sum() - 1},); bnds = [(0, 1)] * n
    r = minimize(lambda v: wmse(y, P_ @ v, w), w0, method="SLSQP", bounds=bnds, constraints=cons)
    return wmse(y, P_ @ r.x, w), r.x

base_score, base_w = best_blend(Pbase, W)
print(f"\nMevcut 4-model blend weighted-OOF = {base_score:.3f}  weights={np.round(base_w,3)}")
print("\nHer adayi ekleyince:")
for name in cands:
    Pp = np.column_stack([Pbase, cand_oof[name]])
    s, w_ = best_blend(Pp, W)
    print(f"  +{name:11s} -> {s:.3f}   (delta {s-base_score:+.3f})  w={np.round(w_,3)}")
# hepsini ekle
Pall = np.column_stack([Pbase] + [cand_oof[n] for n in cands])
s, w_ = best_blend(Pall, W)
print(f"\n+HEPSI -> {s:.3f}  (delta {s-base_score:+.3f})  w={np.round(w_,3)}  order={base_models+list(cands)}")
