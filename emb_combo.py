# -*- coding: utf-8 -*-
"""Son tur: DOGRUSAL embedding+tabular birlesik modeller (her iki transformer + FE).
Agaclar embedding'i iyi kullanamiyor, dogrusal modeller kullaniyor -> decorrelated katki.
Cikti: combo_oof.csv / combo_test.csv."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
n_tr = len(train)
E1 = np.load("emb_train.npy"); E1t = np.load("emb_test.npy")
E2 = np.load("emb2_train.npy"); E2t = np.load("emb2_test.npy")
tX = train.drop(columns=[P.TARGET]); comb = pd.concat([tX, test], ignore_index=True)
comb = P.create_features(comb); comb = P.encode_categoricals(comb)
codec = ["university_tier_code"] + [f"{c}_code" for c in P.CAT_COLS if c != "university_tier"]
num = [c for c in comb.columns if c not in P.CAT_COLS + [P.ID_COL, P.TEXT_COL] and comb[c].dtype != object and c not in codec]
feat = num + codec
F = comb[feat].iloc[:n_tr].reset_index(drop=True).values
Ft = comb[feat].iloc[n_tr:].reset_index(drop=True).values

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))
oof_store, test_store = {}, {}

def cv(name, make, Xtr, Xte):
    oof = np.zeros(n_tr); tp = np.zeros(len(Xte))
    for tr, va in folds:
        im = SimpleImputer(strategy="median").fit(Xtr[tr]); a, b, c = im.transform(Xtr[tr]), im.transform(Xtr[va]), im.transform(Xte)
        sc = StandardScaler().fit(a); a, b, c = sc.transform(a), sc.transform(b), sc.transform(c)
        m = make(); m.fit(a, y[tr]); oof[va] = m.predict(b); tp += m.predict(c) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("combo_oof.csv", index=False)
    pd.DataFrame(test_store).to_csv("combo_test.csv", index=False)
    print(f"  {name:18s} solo MSE={mean_squared_error(y,oof):.2f}")

EF_both = np.hstack([E1, E2, F]); EFt_both = np.hstack([E1t, E2t, Ft])
EF1 = np.hstack([E1, F]); EF1t = np.hstack([E1t, Ft])
print("Dogrusal birlesik modeller...")
cv("combo_ridge_both", lambda: Ridge(alpha=30.0), EF_both, EFt_both)
cv("combo_enet_both", lambda: ElasticNet(alpha=0.005, l1_ratio=0.15, max_iter=4000), EF_both, EFt_both)
cv("combo_ridge1", lambda: Ridge(alpha=50.0), EF1, EF1t)
cv("combo_huber1", lambda: HuberRegressor(alpha=0.001, epsilon=1.5, max_iter=300), EF1, EF1t)
print("combo_oof.csv / combo_test.csv yazildi.")
