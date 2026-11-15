# -*- coding: utf-8 -*-
"""Tum base modelleri (mevcut 6 + diverse 9) topla, unconstrained Ridge stacker'i
multi-seed nested-CV (weighted metrik = LB izleyici) ile dogrula, en robust olani sec,
submission yaz. Asiri model -> overfit riski; nested-CV bunu yakalar."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
yr = train["application_year"].values
tw = test["application_year"].value_counts(normalize=True)
trw = train["application_year"].value_counts(normalize=True)
W = train["application_year"].map(tw / trw).values; W = W / W.mean()
wmse = lambda p, w=W: float(np.average((y - p) ** 2, weights=w))

# --- tum OOF / test tahminlerini topla ---
oof = pd.read_csv("oof_predictions.csv"); ex = pd.read_csv("extra_oof.csv")
tp = pd.read_csv("test_predictions.csv"); ext = pd.read_csv("extra_test.csv")
dv_o = pd.read_csv("diverse_oof.csv"); dv_t = pd.read_csv("diverse_test.csv")

OOF = {}; TEST = {}
for m in ["catboost", "lgbm", "xgb", "ridge"]:
    OOF[m] = oof[f"oof_{m}"].values; TEST[m] = tp[f"test_{m}"].values
OOF["histgb6"] = ex["hist_oof"].values; TEST["histgb6"] = ext["hist_te"].values
OOF["mlp6"] = ex["mlp_oof"].values; TEST["mlp6"] = ext["mlp_te"].values
for c in dv_o.columns:
    OOF[c] = dv_o[c].values; TEST[c] = dv_t[c].values

names = list(OOF.keys())
Po = np.column_stack([OOF[n] for n in names]); Pt = np.column_stack([TEST[n] for n in names])
print(f"Toplam {len(names)} base model: {names}")
print("Solo weighted-OOF:")
for n in names: print(f"  {n:14s} {wmse(OOF[n]):.3f}")

def nested(Xm, alpha, seeds=(1, 7, 13, 21, 42), clip=True):
    sc = []
    for s in seeds:
        kf = KFold(5, shuffle=True, random_state=s); pr = np.zeros(len(y))
        for tr, va in kf.split(Xm):
            pr[va] = Ridge(alpha=alpha).fit(Xm[tr], y[tr]).predict(Xm[va])
        sc.append(wmse(np.clip(pr, 0, 100) if clip else pr))
    return np.mean(sc), np.std(sc)

yd = pd.get_dummies(yr).values.astype(float)
print("\n=== Stacker konfig taramasi (multi-seed nested-CV weighted-OOF) ===")
results = {}
for label, Xm in {
    "all-15": Po,
    "all-15 + year_dummy": np.column_stack([Po, yd]),
    "all-15 + year*ens": np.column_stack([Po, yd, yd * oof["oof_ensemble"].values[:, None]]),
}.items():
    for a in [5, 20, 50, 100]:
        mu, sd = nested(Xm, a)
        results[(label, a)] = (mu, sd, Xm)
        print(f"  {label:24s} alpha={a:4d}  {mu:.3f} +/- {sd:.3f}")

# en robust: en dusuk mu (stabilite icin sd kucuk olani tercih ederek)
best = min(results.items(), key=lambda kv: kv[1][0] + 0.5 * kv[1][1])
(blab, ba), (bmu, bsd, bXm) = best
print(f"\nSECILEN: {blab} alpha={ba}  weighted-OOF={bmu:.3f} +/- {bsd:.3f}")
print(f"(referans: eski 6-model stacker 87.92, convex ensemble 89.56)")

# test matrisini ayni sekilde kur
if blab == "all-15":
    XmTe = Pt
elif blab == "all-15 + year_dummy":
    ydt = pd.get_dummies(test["application_year"]).reindex(columns=sorted(set(yr)), fill_value=0).values.astype(float)
    XmTe = np.column_stack([Pt, ydt])
else:
    ydt = pd.get_dummies(test["application_year"]).reindex(columns=sorted(set(yr)), fill_value=0).values.astype(float)
    ens_te = tp["test_ensemble"].values
    XmTe = np.column_stack([Pt, ydt, ydt * ens_te[:, None]])

final = Ridge(alpha=ba).fit(bXm, y)
pred = np.clip(final.predict(XmTe), 0, 100)
sub = pd.DataFrame({"student_id": test["student_id"].values, "career_success_score": pred})
assert len(sub) == len(test) and not sub.iloc[:, 1].isna().any() and sub.iloc[:, 1].between(0, 100).all()
import shutil
shutil.copy("submission.csv", "submission_prev_backup.csv")
sub.to_csv("submission.csv", index=False, encoding="utf-8")
sub.to_csv("submission_v4_stack15.csv", index=False, encoding="utf-8")
print(f"\nsubmission.csv GUNCELLENDI. pred mean/std={pred.mean():.2f}/{pred.std():.2f} range[{pred.min():.2f},{pred.max():.2f}]")
print("onceki yedek: submission_prev_backup.csv")
