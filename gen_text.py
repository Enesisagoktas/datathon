# -*- coding: utf-8 -*-
"""Tur 3: mentor_feedback_text'ten DECORRELATED metin modelleri (fold-ici TF-IDF, ayni split).
Word/Char/Combined Ridge + ElasticNet + SVD->LightGBM. Cikti: text_oof.csv / text_test.csv."""
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, ElasticNet
from scipy.sparse import hstack
import lightgbm as lgb
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
txt_tr = train[P.TEXT_COL].fillna("").astype(str).reset_index(drop=True)
txt_te = test[P.TEXT_COL].fillna("").astype(str).reset_index(drop=True)
n_tr = len(txt_tr)
bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))

mods = ["txt_word_ridge", "txt_char_ridge", "txt_comb_ridge", "txt_enet", "txt_svd_lgbm", "txt_svd_ridge"]
oof = {m: np.zeros(n_tr) for m in mods}
tep = {m: np.zeros(len(txt_te)) for m in mods}

t0 = time.time()
for k, (tr, va) in enumerate(folds, 1):
    wv = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=3, sublinear_tf=True).fit(txt_tr.iloc[tr])
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), max_features=40000, min_df=3, sublinear_tf=True).fit(txt_tr.iloc[tr])
    Wtr, Wva, Wte = wv.transform(txt_tr.iloc[tr]), wv.transform(txt_tr.iloc[va]), wv.transform(txt_te)
    Ctr, Cva, Cte = cv.transform(txt_tr.iloc[tr]), cv.transform(txt_tr.iloc[va]), cv.transform(txt_te)
    Btr, Bva, Bte = hstack([Wtr, Ctr]).tocsr(), hstack([Wva, Cva]).tocsr(), hstack([Wte, Cte]).tocsr()
    ytr = y[tr]
    # word ridge
    m = Ridge(alpha=8).fit(Wtr, ytr); oof["txt_word_ridge"][va] = m.predict(Wva); tep["txt_word_ridge"] += m.predict(Wte) / 10
    # char ridge
    m = Ridge(alpha=8).fit(Ctr, ytr); oof["txt_char_ridge"][va] = m.predict(Cva); tep["txt_char_ridge"] += m.predict(Cte) / 10
    # combined ridge (farkli alpha)
    m = Ridge(alpha=30).fit(Btr, ytr); oof["txt_comb_ridge"][va] = m.predict(Bva); tep["txt_comb_ridge"] += m.predict(Bte) / 10
    # elasticnet
    m = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=2000).fit(Btr, ytr); oof["txt_enet"][va] = m.predict(Bva); tep["txt_enet"] += m.predict(Bte) / 10
    # SVD -> lgbm ve ridge
    svd = TruncatedSVD(n_components=200, random_state=42).fit(Btr)
    Str, Sva, Ste = svd.transform(Btr), svd.transform(Bva), svd.transform(Bte)
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2, n_jobs=-1, verbosity=-1)
    g.fit(Str, ytr, eval_set=[(Sva, y[va])], eval_metric="l2",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    oof["txt_svd_lgbm"][va] = g.predict(Sva); tep["txt_svd_lgbm"] += g.predict(Ste) / 10
    m = Ridge(alpha=20).fit(Str, ytr); oof["txt_svd_ridge"][va] = m.predict(Sva); tep["txt_svd_ridge"] += m.predict(Ste) / 10
    pd.DataFrame(oof).to_csv("text_oof.csv", index=False); pd.DataFrame(tep).to_csv("text_test.csv", index=False)
    print(f"  fold {k}/10 done ({time.time()-t0:.0f}s)")

from sklearn.metrics import mean_squared_error
print("\nMetin modelleri solo MSE (uniform):")
for m in mods:
    print(f"  {m:16s} {mean_squared_error(y, oof[m]):.2f}")
print("text_oof.csv / text_test.csv yazildi.")
