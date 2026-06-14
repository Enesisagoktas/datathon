# -*- coding: utf-8 -*-
"""Hizli smoke test: tum pipeline parcalarini kucuk veride calistirir (API/bug yakalama)."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.stdout.reconfigure(encoding="utf-8")

import pipeline as P

# kucult
P.WORD_MAX_FEATURES = 2000
P.CHAR_MAX_FEATURES = 2000
P.WORD_SVD_COMP = 20
P.CHAR_SVD_COMP = 15

train, test, sample = P.load_data()
train = train.sample(1200, random_state=0).reset_index(drop=True)
y = train[P.TARGET].astype(float)
train_X = train.drop(columns=[P.TARGET])
n_train = len(train_X)
combined = pd.concat([train_X, test], ignore_index=True)
combined = P.create_features(combined)
combined = P.encode_categoricals(combined)
base_train = combined.iloc[:n_train].reset_index(drop=True)
base_test = combined.iloc[n_train:].reset_index(drop=True)
text_train = base_train[P.TEXT_COL]
text_test = base_test[P.TEXT_COL]

# role_match sanity
roles = train["target_role"].astype(str).unique()
print("ROLLER:", sorted(roles))
print("role_match_score NaN sayisi:", base_train["role_match_score"].isna().sum())
print("create_features kolon sayisi:", combined.shape[1])

bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(2, shuffle=True, random_state=0).split(np.zeros(n_train), bins))

lp = P.default_lgbm_params(); lp["n_estimators"] = 200
cp = P.default_cat_params(); cp["iterations"] = 200
xp = P.default_xgb_params(); xp["n_estimators"] = 200

oof, test_pred, fic, fil = P.run_cv(base_train, base_test, text_train, text_test, y, folds, lp, cp, xp)
w, ens, eo = P.optimize_ensemble_weights(oof, y)
print("\nSMOKE OK")
print("ensemble MSE:", round(ens, 3), " weights:", {k: round(v, 3) for k, v in w.items()})
print("test_pred lens:", {k: len(v) for k, v in test_pred.items()})
print("FI catboost top3:", list(fic.head(3).index))
print("FI lgbm top3:", list(fil.head(3).index))
print("any test NaN:", {k: bool(np.isnan(v).any()) for k, v in test_pred.items()})
