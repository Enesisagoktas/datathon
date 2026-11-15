# -*- coding: utf-8 -*-
"""Kanonik stacker: TUM base model tahminlerini topla (genisletilebilir kaynak listesi),
unconstrained Ridge stacker'i 7-seed nested-CV (weighted = LB izleyici) ile dogrula,
en robust alpha'yi sec, submission yaz."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
sys.stdout.reconfigure(encoding="utf-8")
import pipeline as P

train, test, _ = P.load_data()
y = train[P.TARGET].astype(float).values
tw = test["application_year"].value_counts(normalize=True)
trw = train["application_year"].value_counts(normalize=True)
W = train["application_year"].map(tw / trw).values; W = W / W.mean()
wmse = lambda p, w=W: float(np.average((y - p) ** 2, weights=w))

OOF, TEST = {}, {}
# kaynak 1: ana pipeline (oof_<m> / test_<m>)
oof = pd.read_csv("oof_predictions.csv"); tp = pd.read_csv("test_predictions.csv")
for m in ["catboost", "lgbm", "xgb", "ridge"]:
    OOF[m] = oof[f"oof_{m}"].values; TEST[m] = tp[f"test_{m}"].values
# kaynak 2: extra (histgb6, mlp6)
ex = pd.read_csv("extra_oof.csv"); ext = pd.read_csv("extra_test.csv")
OOF["histgb6"] = ex["hist_oof"].values; TEST["histgb6"] = ext["hist_te"].values
OOF["mlp6"] = ex["mlp_oof"].values; TEST["mlp6"] = ext["mlp_te"].values
# kaynak 3+: ayni kolon adli oof/test dosyalari
for fo, ft in [("diverse_oof.csv", "diverse_test.csv"),
               ("diverse2_oof.csv", "diverse2_test.csv"),
               ("text_oof.csv", "text_test.csv"),
               ("embed_oof.csv", "embed_test.csv"),
               ("embed2_oof.csv", "embed2_test.csv")]:
    try:
        do = pd.read_csv(fo); dt = pd.read_csv(ft)
        for c in do.columns:
            OOF[c] = do[c].values; TEST[c] = dt[c].values
    except FileNotFoundError:
        pass

names = list(OOF.keys())
Po = np.column_stack([OOF[n] for n in names]); Pt = np.column_stack([TEST[n] for n in names])
print(f"Toplam {len(names)} base model")

def nested(Xm, alpha, seeds=(1, 7, 13, 21, 42, 100, 2024)):
    sc = []
    for s in seeds:
        kf = KFold(5, shuffle=True, random_state=s); pr = np.zeros(len(y))
        for tr, va in kf.split(Xm):
            pr[va] = Ridge(alpha=alpha).fit(Xm[tr], y[tr]).predict(Xm[va])
        sc.append(wmse(np.clip(pr, 0, 100)))
    return float(np.mean(sc)), float(np.std(sc))

print("\n=== alpha taramasi (7-seed nested-CV weighted-OOF) ===")
best = None
for a in [10, 30, 60, 100, 150, 250, 400]:
    mu, sd = nested(Po, a)
    print(f"  all-{len(names)}  alpha={a:4d}  {mu:.3f} +/- {sd:.3f}")
    if best is None or mu + 0.5 * sd < best[0]:
        best = (mu + 0.5 * sd, a, mu, sd)
_, ba, bmu, bsd = best
print(f"\nSECILEN alpha={ba}  weighted-OOF={bmu:.3f} +/- {bsd:.3f}")
print("(trajektori: convex 89.56 -> 6:87.92 -> 15:86.30 -> 25:86.24 ; LB 89.52->87->85.25)")

final = Ridge(alpha=ba).fit(Po, y)
pred = np.clip(final.predict(Pt), 0, 100)
sub = pd.DataFrame({"student_id": test["student_id"].values, "career_success_score": pred})
assert len(sub) == len(test) and not sub.iloc[:, 1].isna().any() and sub.iloc[:, 1].between(0, 100).all()
import shutil
shutil.copy("submission.csv", "submission_prev_backup.csv")
sub.to_csv("submission.csv", index=False, encoding="utf-8")
co = pd.Series(final.coef_, index=names).sort_values(key=abs, ascending=False)
print("\nEn etkili 12 model (|katsayi|):"); print(co.head(12).round(3).to_string())
print(f"\nsubmission.csv GUNCELLENDI. pred mean/std={pred.mean():.2f}/{pred.std():.2f} range[{pred.min():.2f},{pred.max():.2f}]")
