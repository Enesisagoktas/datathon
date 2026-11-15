# -*- coding: utf-8 -*-
"""Tur 2: stacker'a en cok katan DECORRELATED modeller — farkli kayip fonksiyonlari
(Huber/Fair/Quantile), ozellik-alt-kumesi LightGBM'leri, KNN, ek seed'ler. FE ozellikleri,
ayni 10-fold split. Cikti: diverse2_oof.csv / diverse2_test.csv (checkpoint'li)."""
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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
n_tr = len(tX)
X = comb[feat].iloc[:n_tr].reset_index(drop=True)
Xte = comb[feat].iloc[n_tr:].reset_index(drop=True)

# ozellik alt-kumeleri (decorrelated alt-modeller icin)
tech = [c for c in feat if c in P.TECH_COLS or c.startswith("technical_") or c in
        ("coding_score","problem_solving_score","data_structures_score","sql_score",
         "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score","role_match_score")]
soft = [c for c in feat if c in P.SOFT_COLS or c.startswith("soft_skill_") or c in
        ("communication_score","teamwork_score","leadership_score","presentation_score",
         "interview_mean","interview_gap","technical_interview_score","hr_interview_score")]
expf = [c for c in feat if any(k in c for k in ("project","github","hackathon","internship",
        "freelance","portfolio","opensource","experience","certification","bootcamp","client"))]
acad = [c for c in feat if any(k in c for k in ("cgpa","english","attendance","failed","academic",
        "graduation","application_year","age","university_tier","department"))]

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))
oof_store, test_store = {}, {}

def run_lgb(name, params, cols=None):
    t0 = time.time(); oof = np.zeros(n_tr); tp = np.zeros(len(Xte))
    Xc = X[cols] if cols else X; Xc_te = Xte[cols] if cols else Xte
    catf = [c for c in code if (cols is None or c in cols)]
    for tr, va in folds:
        m = lgb.LGBMRegressor(**params)
        m.fit(Xc.iloc[tr], y[tr], eval_set=[(Xc.iloc[va], y[va])], eval_metric="l2",
              categorical_feature=catf if catf else "auto",
              callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(Xc.iloc[va]); tp += m.predict(Xc_te) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("diverse2_oof.csv", index=False)
    pd.DataFrame(test_store).to_csv("diverse2_test.csv", index=False)
    print(f"  {name:16s} done ({time.time()-t0:.0f}s)")

def run_knn(name, k):
    t0 = time.time(); oof = np.zeros(n_tr); tp = np.zeros(len(Xte))
    for tr, va in folds:
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        f = lambda d: sc.transform(imp.transform(d))
        m = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(f(X.iloc[tr]), y[tr])
        oof[va] = m.predict(f(X.iloc[va])); tp += m.predict(f(Xte)) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("diverse2_oof.csv", index=False)
    pd.DataFrame(test_store).to_csv("diverse2_test.csv", index=False)
    print(f"  {name:16s} done ({time.time()-t0:.0f}s)")

base = dict(n_estimators=2500, learning_rate=0.03, num_leaves=127, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.75, reg_lambda=1.0,
            n_jobs=-1, verbosity=-1)
print("Tur 2 cesitli modeller...")
# farkli kayip fonksiyonlari (decorrelated hatalar)
run_lgb("lgb_huber", dict(base, objective="huber", alpha=0.9, random_state=42))
run_lgb("lgb_fair", dict(base, objective="fair", random_state=42))
run_lgb("lgb_quantile", dict(base, objective="quantile", alpha=0.5, random_state=42))
# ek seed'ler
run_lgb("lgb_seed7", dict(base, random_state=7))
run_lgb("lgb_seed123", dict(base, random_state=123, num_leaves=160))
# ozellik alt-kumesi modelleri (decorrelated)
run_lgb("lgb_tech", dict(base, random_state=42), cols=tech)
run_lgb("lgb_soft", dict(base, random_state=42), cols=soft)
run_lgb("lgb_exp", dict(base, random_state=42), cols=expf)
run_lgb("lgb_acad", dict(base, random_state=42), cols=acad)
# KNN (cok farkli model tipi)
run_knn("knn25", 25)
print("\ndiverse2 yazildi. Model sayisi:", len(oof_store))
print("tech=%d soft=%d exp=%d acad=%d kolon" % (len(tech), len(soft), len(expf), len(acad)))
