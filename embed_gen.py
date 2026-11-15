# -*- coding: utf-8 -*-
"""Cok-dilli sentence-transformer ile mentor_feedback_text embedding'leri uret (cache'li),
sonra embedding uzerinde fold-ici modeller (Ridge/LGBM/MLP) -> OOF+test. Stacker'a eklenecek.
Cikti: embed_oof.csv / embed_test.csv  (+ cache: emb_train.npy / emb_test.npy)"""
import sys, os, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
txt_tr = train[P.TEXT_COL].fillna("").astype(str).tolist()
txt_te = test[P.TEXT_COL].fillna("").astype(str).tolist()
n_tr = len(txt_tr)

if os.path.exists("emb_train.npy") and os.path.exists("emb_test.npy"):
    Etr = np.load("emb_train.npy"); Ete = np.load("emb_test.npy")
    print("Embedding cache yuklendi:", Etr.shape, Ete.shape)
else:
    from sentence_transformers import SentenceTransformer
    model_name = None
    for cand in ["paraphrase-multilingual-MiniLM-L12-v2",
                 "distiluse-base-multilingual-cased-v2",
                 "paraphrase-multilingual-mpnet-base-v2"]:
        try:
            print("Model yukleniyor:", cand)
            st = SentenceTransformer(cand); model_name = cand; break
        except Exception as e:
            print("  basarisiz:", cand, str(e)[:120])
    if model_name is None:
        raise SystemExit("Hicbir embedding modeli yuklenemedi (internet/erisim?).")
    t0 = time.time()
    Etr = st.encode(txt_tr, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    Ete = st.encode(txt_te, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.save("emb_train.npy", Etr); np.save("emb_test.npy", Ete)
    print(f"Embedding uretildi ({model_name}) {Etr.shape} ({time.time()-t0:.0f}s)")

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))
oof_store, test_store = {}, {}

def cv(name, make, scale=False):
    t0 = time.time(); oof = np.zeros(n_tr); tp = np.zeros(len(Ete))
    for tr, va in folds:
        Xtr, Xva, Xt = Etr[tr], Etr[va], Ete
        if scale:
            sc = StandardScaler().fit(Xtr); Xtr, Xva, Xt = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xt)
        m = make(); m.fit(Xtr, y[tr]); oof[va] = m.predict(Xva); tp += m.predict(Xt) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("embed_oof.csv", index=False)
    pd.DataFrame(test_store).to_csv("embed_test.csv", index=False)
    from sklearn.metrics import mean_squared_error
    print(f"  {name:14s} solo MSE={mean_squared_error(y,oof):.2f} ({time.time()-t0:.0f}s)")

print("Embedding modelleri...")
cv("emb_ridge", lambda: Ridge(alpha=10.0), scale=True)
cv("emb_lgbm", lambda: lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31,
   min_child_samples=40, subsample=0.8, colsample_bytree=0.6, reg_lambda=3, n_jobs=-1, verbosity=-1))
cv("emb_mlp", lambda: MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-3, batch_size=128,
   learning_rate_init=1.5e-3, max_iter=150, early_stopping=True, random_state=42), scale=True)
print("embed_oof.csv / embed_test.csv yazildi.")
