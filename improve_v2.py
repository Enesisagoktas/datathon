# -*- coding: utf-8 -*-
"""v2: mevcut 4 base modelin KAYITLI OOF+test tahminlerine HistGB+MLP ekle, blend'i
test-yili agirlikli (LB-izleyen) metrikle optimize et, yeni submission yaz. Yeniden
egitim YOK (base modeller tekrar kosulmaz)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, sample = P.load_data()
y = train[P.TARGET].astype(float).values
tX = train.drop(columns=[P.TARGET]); comb = pd.concat([tX, test], ignore_index=True)
comb = P.create_features(comb); comb = P.encode_categoricals(comb)
code = ["university_tier_code"] + [f"{c}_code" for c in P.CAT_COLS if c != "university_tier"]
num = [c for c in comb.columns if c not in P.CAT_COLS + [P.ID_COL, P.TEXT_COL]
       and comb[c].dtype != object and c not in code]
feat = num + code
X = comb[feat].iloc[:len(tX)].reset_index(drop=True)
Xte = comb[feat].iloc[len(tX):].reset_index(drop=True)

tw = test["application_year"].value_counts(normalize=True)
trw = train["application_year"].value_counts(normalize=True)
W = train["application_year"].map(tw / trw).values; W = W / W.mean()
def wmse(t, p, w): return float(np.average((t - p) ** 2, weights=w))

bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop")
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(len(y)), bins))

def cv(make, prep):
    oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
    for tr, va in folds:
        Xtr, Xva = X.iloc[tr], X.iloc[va]; Xt = Xte
        if prep:
            imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
            f = lambda d: sc.transform(imp.transform(d))
            Xtr, Xva, Xt = f(Xtr), f(Xva), f(Xt)
        m = make(); m.fit(Xtr, y[tr])
        oof[va] = m.predict(Xva); tp += m.predict(Xt) / len(folds)
    return oof, tp

print("HistGB + MLP OOF/test uretiliyor (yeniden egitim yok, sadece 2 ek model)...")
hist_oof, hist_te = cv(lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03,
                        max_leaf_nodes=63, l2_regularization=2.0, random_state=42), False)
mlp_oof, mlp_te = cv(lambda: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
                        learning_rate_init=2e-3, max_iter=120, early_stopping=True, random_state=42), True)

# ek model tahminlerini cache'le (tekrar hesaplamamak icin)
pd.DataFrame({"hist_oof": hist_oof, "mlp_oof": mlp_oof}).to_csv("extra_oof.csv", index=False)
pd.DataFrame({"hist_te": hist_te, "mlp_te": mlp_te}).to_csv("extra_test.csv", index=False)

oofdf = pd.read_csv("oof_predictions.csv"); tpdf = pd.read_csv("test_predictions.csv")
base = ["catboost", "lgbm", "xgb", "ridge"]
Po = np.column_stack([oofdf[f"oof_{m}"].values for m in base] + [hist_oof, mlp_oof])
Pt = np.column_stack([tpdf[f"test_{m}"].values for m in base] + [hist_te, mlp_te])
names = base + ["histgb", "mlp"]

def best_blend(Pm, w, yy):
    n = Pm.shape[1]; w0 = np.full(n, 1 / n)
    cons = ({"type": "eq", "fun": lambda v: v.sum() - 1},); bnds = [(0, 1)] * n
    r = minimize(lambda v: wmse(yy, Pm @ v, w), w0, method="SLSQP", bounds=bnds, constraints=cons)
    ww = np.clip(r.x, 0, None); ww /= ww.sum()
    return ww

# 4-model (v1) referans, weighted
w4 = best_blend(Po[:, :4], W, y)
print(f"\nv1 (4 model) weighted-OOF = {wmse(y, Po[:, :4] @ w4, W):.3f}")
# 6-model weighted
w6 = best_blend(Po, W, y)
print(f"v2 (6 model) weighted-OOF = {wmse(y, Po @ w6, W):.3f}  weights={dict(zip(names, np.round(w6,3)))}")
# nested-CV overfit kontrolu
kf = KFold(5, shuffle=True, random_state=1); os_ = np.zeros(len(y))
for tr, va in kf.split(Po):
    ww = best_blend(Po[tr], W[tr], y[tr]); os_[va] = Po[va] @ ww
print(f"v2 nested-CV weighted-OOF = {wmse(y, os_, W):.3f}  (overfit kontrolu)")

# submission v2
pred = np.clip(Pt @ w6, 0, 100)
sub = pd.DataFrame({P.ID_COL: test[P.ID_COL].values, P.TARGET: pred})
assert len(sub) == len(test) and not sub[P.TARGET].isna().any() and sub[P.TARGET].between(0,100).all()
sub.to_csv("submission_v2.csv", index=False, encoding="utf-8")
print(f"\nsubmission_v2.csv yazildi. pred mean/std = {pred.mean():.2f}/{pred.std():.2f}")
print(f"v1 vs v2 tahmin ortalama farki = {abs(pred.mean() - tpdf['test_ensemble'].mean()):.3f}")
