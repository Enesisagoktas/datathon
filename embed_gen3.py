# -*- coding: utf-8 -*-
"""Tur 2 embedding: FARKLI bir cok-dilli transformer (decorrelated semantik). Encode + cache +
fold-ici modeller. Cikti: embed3_oof.csv / embed3_test.csv (cache: emb2_train/emb2_test.npy)."""
import sys, os, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
txt_tr = train[P.TEXT_COL].fillna("").astype(str).tolist()
txt_te = test[P.TEXT_COL].fillna("").astype(str).tolist()
n_tr = len(txt_tr)

if os.path.exists("emb2_train.npy"):
    Etr = np.load("emb2_train.npy"); Ete = np.load("emb2_test.npy")
    print("cache yuklendi", Etr.shape)
else:
    from sentence_transformers import SentenceTransformer
    name = None
    for cand in ["distiluse-base-multilingual-cased-v2", "paraphrase-multilingual-mpnet-base-v2"]:
        try:
            print("Model:", cand); st = SentenceTransformer(cand); name = cand; break
        except Exception as e:
            print("  fail", cand, str(e)[:100])
    if name is None: raise SystemExit("model yuklenemedi")
    t0 = time.time()
    Etr = st.encode(txt_tr, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    Ete = st.encode(txt_te, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.save("emb2_train.npy", Etr); np.save("emb2_test.npy", Ete)
    print(f"encode bitti ({name}) {Etr.shape} {time.time()-t0:.0f}s")

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))
oof_store, test_store = {}, {}

def cv(name, make, scale=False):
    oof = np.zeros(n_tr); tp = np.zeros(len(Ete))
    for tr, va in folds:
        a, b, c = Etr[tr], Etr[va], Ete
        if scale:
            sc = StandardScaler().fit(a); a, b, c = sc.transform(a), sc.transform(b), sc.transform(c)
        m = make(); m.fit(a, y[tr]); oof[va] = m.predict(b); tp += m.predict(c) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("embed3_oof.csv", index=False)
    pd.DataFrame(test_store).to_csv("embed3_test.csv", index=False)
    from sklearn.metrics import mean_squared_error
    print(f"  {name:14s} solo MSE={mean_squared_error(y,oof):.2f}")

cv("emb2_ridge", lambda: Ridge(alpha=10.0), scale=True)
cv("emb2_lgbm", lambda: lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31,
   min_child_samples=40, subsample=0.8, colsample_bytree=0.6, reg_lambda=3, n_jobs=-1, verbosity=-1))
print("embed3 yazildi.")
