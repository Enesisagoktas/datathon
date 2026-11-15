# -*- coding: utf-8 -*-
"""Cesitli + daha az regularize base modellerin OOF+test tahminlerini uret (FE ozellikleri,
ayni 10-fold split). Stacker bunlari kullanacak. Sonuc: diverse_oof.csv / diverse_test.csv"""
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
tX = train.drop(columns=[P.TARGET]); comb = pd.concat([tX, test], ignore_index=True)
comb = P.create_features(comb); comb = P.encode_categoricals(comb)
code = ["university_tier_code"] + [f"{c}_code" for c in P.CAT_COLS if c != "university_tier"]
num = [c for c in comb.columns if c not in P.CAT_COLS + [P.ID_COL, P.TEXT_COL]
       and comb[c].dtype != object and c not in code]
gbm_feat = num + code
n_tr = len(tX)
Xg = comb[gbm_feat].iloc[:n_tr].reset_index(drop=True)
Xg_te = comb[gbm_feat].iloc[n_tr:].reset_index(drop=True)
# CatBoost icin ham kategorik
cat_feat = num + P.CAT_COLS
Xc = comb[cat_feat].iloc[:n_tr].reset_index(drop=True).copy()
Xc_te = comb[cat_feat].iloc[n_tr:].reset_index(drop=True).copy()
for c in P.CAT_COLS:
    Xc[c] = Xc[c].astype(str).fillna("NA"); Xc_te[c] = Xc_te[c].astype(str).fillna("NA")

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n_tr), bins))

oof_store, test_store = {}, {}

def run_gbm(name, kind, params):
    t0 = time.time(); oof = np.zeros(n_tr); tp = np.zeros(len(Xg_te))
    for tr, va in folds:
        if kind == "lgb":
            m = lgb.LGBMRegressor(**params)
            m.fit(Xg.iloc[tr], y[tr], eval_set=[(Xg.iloc[va], y[va])], eval_metric="l2",
                  categorical_feature=code, callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
            oof[va] = m.predict(Xg.iloc[va]); tp += m.predict(Xg_te) / len(folds)
        elif kind == "xgb":
            m = xgb.XGBRegressor(**params)
            m.fit(Xg.iloc[tr], y[tr], eval_set=[(Xg.iloc[va], y[va])], verbose=False)
            oof[va] = m.predict(Xg.iloc[va]); tp += m.predict(Xg_te) / len(folds)
        elif kind == "cat":
            m = CatBoostRegressor(**params)
            m.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va]), cat_features=P.CAT_COLS,
                  use_best_model=True, verbose=0)
            oof[va] = m.predict(Xc.iloc[va]); tp += m.predict(Xc_te) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("diverse_oof.csv", index=False)   # checkpoint
    pd.DataFrame(test_store).to_csv("diverse_test.csv", index=False)
    print(f"  {name:14s} done ({time.time()-t0:.0f}s)")

def run_sk(name, make, prep):
    t0 = time.time(); oof = np.zeros(n_tr); tp = np.zeros(len(Xg_te))
    for tr, va in folds:
        Xtr, Xva, Xt = Xg.iloc[tr], Xg.iloc[va], Xg_te
        if prep:
            imp = SimpleImputer(strategy="median").fit(Xtr)
            Xtr, Xva, Xt = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xt)
        m = make(); m.fit(Xtr, y[tr]); oof[va] = m.predict(Xva); tp += m.predict(Xt) / len(folds)
    oof_store[name] = oof; test_store[name] = tp
    pd.DataFrame(oof_store).to_csv("diverse_oof.csv", index=False)   # checkpoint
    pd.DataFrame(test_store).to_csv("diverse_test.csv", index=False)
    print(f"  {name:14s} done ({time.time()-t0:.0f}s)")

print("Cesitli modeller uretiliyor (leaner/hizli konfig)...")
# az regularize / cesitli GBM'ler (makul derinlik -> hizli)
run_gbm("lgbm_strong", "lgb", dict(n_estimators=2500, learning_rate=0.03, num_leaves=255,
        min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=0.1, reg_alpha=0.0, random_state=42, n_jobs=-1, verbosity=-1))
run_gbm("lgbm_mid", "lgb", dict(n_estimators=2500, learning_rate=0.03, num_leaves=127,
        min_child_samples=25, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.2, random_state=42, n_jobs=-1, verbosity=-1))
run_gbm("lgbm_seed99", "lgb", dict(n_estimators=2500, learning_rate=0.02, num_leaves=200,
        min_child_samples=40, subsample=0.7, subsample_freq=1, colsample_bytree=0.65,
        reg_lambda=0.2, reg_alpha=0.04, random_state=99, n_jobs=-1, verbosity=-1))
run_gbm("xgb_strong", "xgb", dict(n_estimators=2500, learning_rate=0.03, max_depth=8,
        min_child_weight=2, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, reg_alpha=0.0,
        random_state=42, n_jobs=-1, tree_method="hist", early_stopping_rounds=120, eval_metric="rmse"))
run_gbm("cat_strong", "cat", dict(iterations=2500, learning_rate=0.04, depth=8, l2_leaf_reg=1.5,
        loss_function="RMSE", eval_metric="RMSE", random_seed=42, early_stopping_rounds=120,
        thread_count=-1, bootstrap_type="Bernoulli", subsample=0.85))
run_gbm("cat_seed99", "cat", dict(iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=2.0,
        loss_function="RMSE", eval_metric="RMSE", random_seed=99, early_stopping_rounds=120,
        thread_count=-1, bootstrap_type="Bernoulli", subsample=0.8))
# tip cesitliligi
run_sk("histgb_strong", lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03,
       max_leaf_nodes=200, l2_regularization=0.3, random_state=42), False)
run_sk("extratrees", lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3,
       n_jobs=-1, random_state=42), True)
run_sk("randomforest", lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=4,
       n_jobs=-1, random_state=42), True)
print("\ndiverse_oof.csv / diverse_test.csv yazildi. Model sayisi:", len(oof_store))
