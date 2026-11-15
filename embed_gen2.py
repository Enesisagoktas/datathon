# -*- coding: utf-8 -*-
"""Cache'lenmis embedding'lerden (emb_train/emb_test.npy) ek DECORRELATED modeller:
ExtraTrees, KNN, ElasticNet ve embedding+tabular BIRLESIK modeller (metin<->ozellik
etkilesimi). Cikti: embed2_oof.csv / embed2_test.csv."""
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
n_tr = len(train)
E = np.load("emb_train.npy"); Ete = np.load("emb_test.npy")
print("Embedding:", E.shape)

# FE ozellikleri (birlesik modeller icin)
tX = train.drop(columns=[P.TARGET]); comb = pd.concat([tX, test], ignore_index=True)
comb = P.create_features(comb); comb = P.encode_categoricals(comb)
codec = ["university_tier_code"] + [f"{c}_code" for c in P.CAT_COLS if c != "university_tier"]
num = [c for c in comb.columns if c not in P.CAT_COLS + [P.ID_COL, P.TEXT_COL]
       and comb[c].dtype != object and c not in codec]
feat = num + codec
F = comb[feat].iloc[:n_tr].reset_index(drop=True).values
Fte = comb[feat].iloc[n_tr:].reset_index(drop=True).values

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))
oof_store, test_store = {}, {}

def cvrun(name, make, Xtr_all, Xte_all, scale=False, impute=False):
    t0 = time.time(); oof = np.zeros(n_tr); tp = np.zeros(len(Xte_all))
    for tr, va in folds:
        a, b, c = Xtr_all[tr], Xtr_all[va], Xte_all
        if impute:
            im = SimpleImputer(strategy="median").fit(a); a, b, c = im.transform(a), im.transform(b), im.transform(c)
        if scale:
            sc = StandardScaler().fit(a); a, b, c = sc.transform(a), sc.transform(b), sc.transform(c)
        m = make(); m.fit(a, y[tr]); oof[va] = m.predict(b); tp += m.predict(c) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("embed2_oof.csv", index=False)
    pd.DataFrame(test_store).to_csv("embed2_test.csv", index=False)
    from sklearn.metrics import mean_squared_error
    print(f"  {name:16s} solo MSE={mean_squared_error(y,oof):.2f} ({time.time()-t0:.0f}s)")

# embedding-only ek tipler
cvrun("emb_extratrees", lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42), E, Ete)
cvrun("emb_knn", lambda: KNeighborsRegressor(n_neighbors=30, weights="distance"), E, Ete, scale=True)
cvrun("emb_enet", lambda: ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=3000), E, Ete, scale=True)
# BIRLESIK: embedding + tabular
EF = np.hstack([E, F]); EFte = np.hstack([Ete, Fte])
cvrun("embfe_lgbm", lambda: lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=127,
      min_child_samples=30, subsample=0.8, colsample_bytree=0.6, reg_lambda=2, n_jobs=-1, verbosity=-1), EF, EFte)
cvrun("embfe_ridge", lambda: Ridge(alpha=20.0), EF, EFte, scale=True, impute=True)
print("embed2_oof.csv / embed2_test.csv yazildi.")
