# -*- coding: utf-8 -*-
"""
BTK / Kaggle Datathon 2026  -  career_success_score regression pipeline
========================================================================

Gorev : test_x.csv icindeki ogrenciler icin `career_success_score` (0-100, surekli)
        tahmin etmek. Metrik: MSE.

Tasarim ilkeleri
----------------
* Kurallara uygun: test elle etiketleme yok, hedef-tahmin hilesi yok, data leakage yok,
  dis/gizli veri yok, baska takim cozumu yok, public LB'ye manuel ayar yok.
* `student_id` modelde KULLANILMAZ (leakage analizi: korelasyon ~ -0.009, sinyalsiz).
* Tum TF-IDF / SVD / metin modeli FOLD ICINDE fit edilir -> validasyon/ test sizmasi olmaz.
  (Satir-bazli muhendislik ozellikleri ve nominal kategori kodlari capraz-satir istatistik
   icermedigi icin bir kez global hesaplanir; bu leakage yaratmaz.)
* Agac modelleri (CatBoost / LightGBM / XGBoost) eksik degerleri NATIVE isler; ek olarak
  bilgi tasiyan eksiklik icin `*_was_missing` bayraklari uretilir.
* Tahminler [0, 100] araligina clip edilir (hedef bu aralikta sinirli, 100'de yigilma var).

Calistirma
----------
    python -m pip install -r requirements.txt
    python pipeline.py

Uretilen dosyalar: submission.csv, oof_predictions.csv, model_scores.csv,
feature_importance_catboost.csv, feature_importance_lgbm.csv, README.md
"""

from __future__ import annotations

import os
import sys
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------
# Konfigurasyon
# ----------------------------------------------------------------------------------------
SEED = 42
N_FOLDS = 10                # "Kapsamli" butce: 10-fold
N_INNER_FOLDS = 3           # Optuna tuning icin ic CV
TARGET = "career_success_score"
ID_COL = "student_id"
TEXT_COL = "mentor_feedback_text"
CLIP_LOW, CLIP_HIGH = 0.0, 100.0

# Metin (her fold icinde fit edilir)
WORD_MAX_FEATURES = 20000
CHAR_MAX_FEATURES = 20000
WORD_SVD_COMP = 120
CHAR_SVD_COMP = 80
TFIDF_MIN_DF = 3
RIDGE_ALPHA = 10.0   # CV ile secildi (word+char, olceklemesiz). bkz. train_text_ridge

# Optuna
DO_OPTUNA = True
LGBM_TRIALS = 30
CAT_TRIALS = 15
LGBM_OPTUNA_TIMEOUT = 600   # saniye (guvenlik tavani)
CAT_OPTUNA_TIMEOUT = 700

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.csv"
TEST_PATH = HERE / "test_x.csv"
SAMPLE_PATH = HERE / "sample_submission.csv"

# Kategorik kolonlar
CAT_COLS = ["department", "university_tier", "target_role", "hobby",
            "preferred_social_media_platform"]

# Bilgi tasiyan eksiklik bayragi uretilecek kolonlar
MISSING_FLAG_COLS = ["english_exam_score", "github_avg_stars", "open_source_contribution_count",
                     "hr_interview_score", "linkedin_profile_score", "portfolio_score",
                     "internship_duration_months"]

# Beceri kolon gruplari
TECH_COLS = ["coding_score", "problem_solving_score", "data_structures_score", "sql_score",
             "machine_learning_score", "backend_score", "frontend_score", "cloud_score",
             "devops_score"]
SOFT_COLS = ["communication_score", "teamwork_score", "leadership_score", "presentation_score"]
COUNT_COLS = ["real_client_project_count", "internship_count", "freelance_project_count",
              "hackathon_count", "hackathon_awards", "github_repo_count",
              "open_source_contribution_count"]

# Veri sozlesmesinde beklenen tum kolonlar (zorunlu kolon kontrolu icin)
EXPECTED_COLS = [
    "student_id", "application_year", "age", "graduation_year", "department",
    "university_tier", "cgpa", "english_exam_score", "attendance_rate", "failed_courses_count",
    "target_role", "coding_score", "problem_solving_score", "data_structures_score", "sql_score",
    "machine_learning_score", "backend_score", "frontend_score", "cloud_score", "devops_score",
    "project_quality_score", "real_client_project_count", "internship_count",
    "internship_duration_months", "freelance_project_count", "hackathon_count", "hackathon_awards",
    "portfolio_score", "github_repo_count", "github_avg_stars", "open_source_contribution_count",
    "linkedin_profile_score", "cv_quality_score", "technical_interview_score", "hr_interview_score",
    "communication_score", "teamwork_score", "leadership_score", "presentation_score",
    "certification_count", "bootcamp_count", "applications_sent", "interviews_attended", "hobby",
    "preferred_social_media_platform", "mentor_feedback_text",
]


def set_seed(seed: int = SEED) -> None:
    """Tum kaynaklarda tekrar-uretilebilirlik icin seed sabitle."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ----------------------------------------------------------------------------------------
# 1) Veri yukleme
# ----------------------------------------------------------------------------------------
def load_data():
    """train / test / sample_submission dosyalarini oku, zorunlu kolonlari dogrula."""
    for p in (TRAIN_PATH, TEST_PATH, SAMPLE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Beklenen veri dosyasi bulunamadi: {p}")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")

    # Zorunlu kolon kontrolu (acik, aciklayici hata)
    missing_train = [c for c in EXPECTED_COLS + [TARGET] if c not in train.columns]
    if missing_train:
        raise ValueError(f"train.csv'de eksik zorunlu kolon(lar): {missing_train}")
    missing_test = [c for c in EXPECTED_COLS if c not in test.columns]
    if missing_test:
        raise ValueError(f"test_x.csv'de eksik zorunlu kolon(lar): {missing_test}")
    if TARGET in test.columns:
        raise ValueError("test_x.csv hedef kolonu icermemeli; data leakage riski!")

    return train, test, sample


# ----------------------------------------------------------------------------------------
# 2) Ozellik muhendisligi  (satir-bazli -> leakage'siz, bir kez global hesaplanabilir)
# ----------------------------------------------------------------------------------------
def _role_match_score(df: pd.DataFrame) -> pd.Series:
    """target_role'a gore ilgili teknik skorlarin ortalamasi (rol uyumu)."""
    role = df["target_role"].astype(str).str.lower()
    tech_mean = df[TECH_COLS].mean(axis=1)

    def grp(cols):
        return df[cols].mean(axis=1)

    is_data = role.str.contains("data|scientist|machine") | role.str.contains(r"\bml\b") \
        | role.str.contains(r"\bai\b") | role.str.contains("ml engineer")
    is_backend = role.str.contains("backend")
    is_frontend = role.str.contains("frontend|front-end|front end")
    is_devops = role.str.contains("devops|cloud|sre")
    is_full = role.str.contains("fullstack|full stack|full-stack")

    score = tech_mean.copy()  # bilinmeyen roller -> technical_mean (default)
    # Oncelik sirasi: full > data > backend > frontend > devops (cakismalari netlestirir)
    score = score.mask(is_devops, grp(["cloud_score", "devops_score", "backend_score"]))
    score = score.mask(is_frontend, grp(["frontend_score", "portfolio_score",
                                         "project_quality_score", "coding_score"]))
    score = score.mask(is_backend, grp(["backend_score", "sql_score", "cloud_score",
                                        "devops_score", "coding_score"]))
    score = score.mask(is_data, grp(["machine_learning_score", "sql_score",
                                     "problem_solving_score", "data_structures_score",
                                     "project_quality_score"]))
    score = score.mask(is_full, grp(["backend_score", "frontend_score", "sql_score",
                                     "cloud_score", "coding_score"]))
    return score


def create_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Spec'teki tum muhendislik ozelliklerini uret. Capraz-satir istatistik kullanmaz."""
    df = df_in.copy()

    # --- Bilgi tasiyan eksiklik bayraklari (doldurmadan ONCE) ---
    for c in MISSING_FLAG_COLS:
        df[f"{c}_was_missing"] = df[c].isna().astype("int8")

    # --- 1) Akademik ---
    df["years_to_graduation"] = df["graduation_year"] - df["application_year"]
    df["age_at_graduation"] = df["age"] + df["years_to_graduation"]
    df["academic_mean"] = df[["cgpa", "english_exam_score", "attendance_rate"]].mean(axis=1)
    df["academic_risk"] = df["failed_courses_count"]
    df["attendance_failed_interaction"] = df["attendance_rate"] / (df["failed_courses_count"] + 1)

    # --- 2) Teknik beceri ---
    tech = df[TECH_COLS]
    df["technical_mean"] = tech.mean(axis=1)
    df["technical_max"] = tech.max(axis=1)
    df["technical_min"] = tech.min(axis=1)
    df["technical_std"] = tech.std(axis=1)
    df["technical_range"] = df["technical_max"] - df["technical_min"]
    df["technical_sum"] = tech.sum(axis=1)

    # --- 3) Sosyal beceri ---
    soft = df[SOFT_COLS]
    df["soft_skill_mean"] = soft.mean(axis=1)
    df["soft_skill_max"] = soft.max(axis=1)
    df["soft_skill_min"] = soft.min(axis=1)
    df["soft_skill_std"] = soft.std(axis=1)
    df["soft_skill_sum"] = soft.sum(axis=1)

    # --- 4) Mulakat ve kariyer hazirligi ---
    df["interview_mean"] = df[["technical_interview_score", "hr_interview_score"]].mean(axis=1)
    df["career_readiness_mean"] = df[["linkedin_profile_score", "cv_quality_score"]].mean(axis=1)
    df["interview_gap"] = df["technical_interview_score"] - df["hr_interview_score"]

    # --- 5) Proje ve deneyim ---
    df["experience_count_sum"] = df[COUNT_COLS].sum(axis=1)
    df["project_experience_score"] = df["project_quality_score"] * np.log1p(df["experience_count_sum"])
    df["github_power"] = np.log1p(df["github_repo_count"]) * np.log1p(df["github_avg_stars"] + 1)
    df["opensource_power"] = np.log1p(df["open_source_contribution_count"])
    df["portfolio_project_interaction"] = df["portfolio_score"] * df["project_quality_score"]
    df["hackathon_success_rate"] = df["hackathon_awards"] / (df["hackathon_count"] + 1)
    df["internship_intensity"] = df["internship_duration_months"] / (df["internship_count"] + 1)

    # --- 6) Basvuru verimliligi ---
    df["interview_rate"] = df["interviews_attended"] / (df["applications_sent"] + 1)
    df["applications_per_interview"] = df["applications_sent"] / (df["interviews_attended"] + 1)

    # --- 7) Rol uyumu ---
    df["role_match_score"] = _role_match_score(df)

    # --- 8) Oran ve etkilesim ---
    df["technical_soft_ratio"] = df["technical_mean"] / (df["soft_skill_mean"] + 1)
    df["project_to_academic_ratio"] = df["project_quality_score"] / (df["academic_mean"] + 1)
    df["career_total_score"] = (df["technical_mean"] + df["soft_skill_mean"]
                                + df["interview_mean"] + df["portfolio_score"]
                                + df["project_quality_score"])
    # Carpimi log1p ile sikistir (patlamayi engelle)
    df["balanced_profile_score"] = np.log1p(df["technical_mean"]
                                            * df["soft_skill_mean"]
                                            * df["interview_mean"])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Nominal kategoriler -> tamsayi kod (LGBM/XGB icin). university_tier ordinal map."""
    out = df.copy()
    # university_tier ordinal: "Tier 1".."Tier 4" -> 1..4
    out["university_tier_code"] = (out["university_tier"].astype(str)
                                   .str.extract(r"(\d+)").astype("float")).fillna(-1).astype("int32")
    for c in CAT_COLS:
        if c == "university_tier":
            continue
        out[f"{c}_code"] = out[c].astype("category").cat.codes.astype("int32")
    return out


# ----------------------------------------------------------------------------------------
# 3) Metin ozellikleri (FOLD ICINDE fit)
# ----------------------------------------------------------------------------------------
def prepare_text_features(text_tr, text_val, text_test):
    """Word + Char TF-IDF -> SVD (agac modelleri icin) ve tam TF-IDF (Ridge icin).
    Tum vektorizer/SVD yalnizca fold-train metni uzerinde fit edilir (leakage'siz).
    """
    text_tr = text_tr.fillna("").astype(str)
    text_val = text_val.fillna("").astype(str)
    text_test = text_test.fillna("").astype(str)

    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=WORD_MAX_FEATURES,
                               min_df=TFIDF_MIN_DF, sublinear_tf=True, lowercase=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=CHAR_MAX_FEATURES, min_df=TFIDF_MIN_DF,
                               sublinear_tf=True, lowercase=True)

    Xw_tr = word_vec.fit_transform(text_tr)
    Xw_val = word_vec.transform(text_val)
    Xw_test = word_vec.transform(text_test)

    Xc_tr = char_vec.fit_transform(text_tr)
    Xc_val = char_vec.transform(text_val)
    Xc_test = char_vec.transform(text_test)

    # SVD (agac modellerine eklenecek yogun ozellikler)
    svd_w = TruncatedSVD(n_components=min(WORD_SVD_COMP, Xw_tr.shape[1] - 1),
                         random_state=SEED)
    svd_c = TruncatedSVD(n_components=min(CHAR_SVD_COMP, Xc_tr.shape[1] - 1),
                         random_state=SEED)
    sw_tr = svd_w.fit_transform(Xw_tr)
    sw_val = svd_w.transform(Xw_val)
    sw_test = svd_w.transform(Xw_test)
    sc_tr = svd_c.fit_transform(Xc_tr)
    sc_val = svd_c.transform(Xc_val)
    sc_test = svd_c.transform(Xc_test)

    def to_df(mat, prefix):
        return pd.DataFrame(mat, columns=[f"{prefix}_{i}" for i in range(mat.shape[1])])

    svd_tr = pd.concat([to_df(sw_tr, "svd_word"), to_df(sc_tr, "svd_char")], axis=1)
    svd_val = pd.concat([to_df(sw_val, "svd_word"), to_df(sc_val, "svd_char")], axis=1)
    svd_test = pd.concat([to_df(sw_test, "svd_word"), to_df(sc_test, "svd_char")], axis=1)

    # Ridge icin tam TF-IDF (word + char)
    tfidf_tr = sparse_hstack([Xw_tr, Xc_tr]).tocsr()
    tfidf_val = sparse_hstack([Xw_val, Xc_val]).tocsr()
    tfidf_test = sparse_hstack([Xw_test, Xc_test]).tocsr()

    return svd_tr, svd_val, svd_test, tfidf_tr, tfidf_val, tfidf_test


# ----------------------------------------------------------------------------------------
# 4) Tek-fold model egiticileri
# ----------------------------------------------------------------------------------------
def train_catboost(Xtr, ytr, Xval, yval, Xtest, cat_features, params):
    model = CatBoostRegressor(**params)
    model.fit(Xtr, ytr, eval_set=(Xval, yval), cat_features=cat_features,
              use_best_model=True, verbose=0)
    val_pred = model.predict(Xval)
    test_pred = model.predict(Xtest)
    return val_pred, test_pred, model


def train_lgbm(Xtr, ytr, Xval, yval, Xtest, cat_features, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="l2",
              categorical_feature=cat_features,
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    val_pred = model.predict(Xval)
    test_pred = model.predict(Xtest)
    return val_pred, test_pred, model


def train_xgb(Xtr, ytr, Xval, yval, Xtest, params):
    model = xgb.XGBRegressor(**params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    val_pred = model.predict(Xval)
    test_pred = model.predict(Xtest)
    return val_pred, test_pred, model


def train_text_ridge(tfidf_tr, ytr, tfidf_val, tfidf_test, alpha=RIDGE_ALPHA):
    # NOT: TF-IDF zaten L2-normalize. StandardScaler(with_mean=False) nadir ozellikleri
    # sisirip agir overfit'e yol aciyordu (valMSE ~318). Olcekleme YOK + makul alpha (~10)
    # ile fold valMSE ~143'e duser (varyans baseline ~227'nin cok altinda).
    ridge = Ridge(alpha=alpha)
    ridge.fit(tfidf_tr, ytr)
    val_pred = ridge.predict(tfidf_val)
    test_pred = ridge.predict(tfidf_test)
    return val_pred, test_pred, ridge


# ----------------------------------------------------------------------------------------
# Varsayilan (makul) hiperparametreler
# ----------------------------------------------------------------------------------------
def default_cat_params():
    return dict(iterations=2500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
                early_stopping_rounds=150, thread_count=-1, bootstrap_type="Bernoulli",
                subsample=0.85)


def default_lgbm_params():
    return dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                min_child_samples=40, subsample=0.85, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
                random_state=SEED, n_jobs=-1, verbosity=-1)


def default_xgb_params():
    return dict(n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5,
                subsample=0.85, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
                random_state=SEED, n_jobs=-1, tree_method="hist",
                early_stopping_rounds=150, eval_metric="rmse")


# ----------------------------------------------------------------------------------------
# 5) Optuna ile hiperparametre tuning  (sadece sayisal+kodlu kategorik; metin haric -> hizli)
# ----------------------------------------------------------------------------------------
def tune_lgbm(X, y, cat_features):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = dict(
            n_estimators=2000, learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 120),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1)
        scores = []
        for tr, va in inner.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                  eval_metric="l2", categorical_feature=cat_features,
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            scores.append(mean_squared_error(y.iloc[va], m.predict(X.iloc[va])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=LGBM_TRIALS, timeout=LGBM_OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    best = default_lgbm_params()
    best.update(study.best_params)
    print(f"    [Optuna LGBM] best inner-MSE={study.best_value:.4f}  params={study.best_params}")
    return best


def tune_catboost(X, y, cat_features):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = dict(
            iterations=1500, learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
            early_stopping_rounds=100, thread_count=-1, bootstrap_type="Bernoulli", verbose=0)
        scores = []
        for tr, va in inner.split(X):
            m = CatBoostRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]),
                  cat_features=cat_features, use_best_model=True, verbose=0)
            scores.append(mean_squared_error(y.iloc[va], m.predict(X.iloc[va])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=CAT_TRIALS, timeout=CAT_OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    best = default_cat_params()
    best.update(study.best_params)
    print(f"    [Optuna CatBoost] best inner-MSE={study.best_value:.4f}  params={study.best_params}")
    return best


# ----------------------------------------------------------------------------------------
# Baseline (FE/metin yok) -> referans CV MSE
# ----------------------------------------------------------------------------------------
def run_baseline(train_enc, y, folds):
    """Ham sayisal + kodlu kategorik (FE yok, metin yok) LightGBM baseline."""
    raw_num = [c for c in EXPECTED_COLS if c not in CAT_COLS + [ID_COL, TEXT_COL]]
    code_cols = ["university_tier_code"] + [f"{c}_code" for c in CAT_COLS if c != "university_tier"]
    feats = raw_num + code_cols
    X = train_enc[feats]
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=63,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], eval_metric="l2",
              categorical_feature=code_cols,
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
    return float(mean_squared_error(y, oof))


# ----------------------------------------------------------------------------------------
# 6) Ana CV dongusu (4 model, fold-ici metin)
# ----------------------------------------------------------------------------------------
def run_cv(base_train, base_test, text_train, text_test, y, folds,
           lgbm_params, cat_params, xgb_params):
    n_tr, n_te = len(y), len(base_test)

    # Agac modelleri icin temel ozellik kolonlari
    num_cols = [c for c in base_train.columns
                if c not in CAT_COLS + [ID_COL, TEXT_COL, TARGET]
                and base_train[c].dtype != object]
    code_cols = ["university_tier_code"] + [f"{c}_code" for c in CAT_COLS if c != "university_tier"]
    # num_cols zaten *_code kolonlarini icerir (int dtype) -> tekrarlamamak icin ayikla
    num_cols = [c for c in num_cols if c not in code_cols]
    lgb_feats = num_cols + code_cols              # LGBM / XGB
    cat_feats = num_cols + CAT_COLS               # CatBoost (native kategorik)

    models = ["catboost", "lgbm", "xgb", "ridge"]
    oof = {m: np.zeros(n_tr) for m in models}
    test_pred = {m: np.zeros(n_te) for m in models}
    fi_cat, fi_lgb = None, None

    for k, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        ytr, yval = y.iloc[tr], y.iloc[va]

        # --- fold-ici metin ozellikleri ---
        svd_tr, svd_val, svd_test, tf_tr, tf_val, tf_test = prepare_text_features(
            text_train.iloc[tr], text_train.iloc[va], text_test)
        svd_tr.index = tr; svd_val.index = va

        # --- LGBM / XGB matrisi ---
        Xlgb_tr = pd.concat([base_train[lgb_feats].iloc[tr].reset_index(drop=True),
                             svd_tr.reset_index(drop=True)], axis=1)
        Xlgb_val = pd.concat([base_train[lgb_feats].iloc[va].reset_index(drop=True),
                              svd_val.reset_index(drop=True)], axis=1)
        Xlgb_test = pd.concat([base_test[lgb_feats].reset_index(drop=True),
                               svd_test.reset_index(drop=True)], axis=1)

        # --- CatBoost matrisi (native kategorik string) ---
        Xcat_tr = pd.concat([base_train[cat_feats].iloc[tr].reset_index(drop=True),
                             svd_tr.reset_index(drop=True)], axis=1)
        Xcat_val = pd.concat([base_train[cat_feats].iloc[va].reset_index(drop=True),
                              svd_val.reset_index(drop=True)], axis=1)
        Xcat_test = pd.concat([base_test[cat_feats].reset_index(drop=True),
                               svd_test.reset_index(drop=True)], axis=1)
        for c in CAT_COLS:
            Xcat_tr[c] = Xcat_tr[c].astype(str).fillna("NA")
            Xcat_val[c] = Xcat_val[c].astype(str).fillna("NA")
            Xcat_test[c] = Xcat_test[c].astype(str).fillna("NA")

        # --- modeller ---
        cv_p, ct_p, cmodel = train_catboost(Xcat_tr, ytr.reset_index(drop=True),
                                            Xcat_val, yval.reset_index(drop=True),
                                            Xcat_test, CAT_COLS, cat_params)
        oof["catboost"][va] = cv_p; test_pred["catboost"] += ct_p / len(folds)

        lv_p, lt_p, lmodel = train_lgbm(Xlgb_tr, ytr.reset_index(drop=True),
                                        Xlgb_val, yval.reset_index(drop=True),
                                        Xlgb_test, code_cols, lgbm_params)
        oof["lgbm"][va] = lv_p; test_pred["lgbm"] += lt_p / len(folds)

        xv_p, xt_p, _ = train_xgb(Xlgb_tr, ytr.reset_index(drop=True),
                                  Xlgb_val, yval.reset_index(drop=True),
                                  Xlgb_test, xgb_params)
        oof["xgb"][va] = xv_p; test_pred["xgb"] += xt_p / len(folds)

        rv_p, rt_p, _ = train_text_ridge(tf_tr, ytr.reset_index(drop=True), tf_val, tf_test)
        oof["ridge"][va] = rv_p; test_pred["ridge"] += rt_p / len(folds)

        # --- feature importance birikim ---
        ci = pd.Series(cmodel.get_feature_importance(), index=Xcat_tr.columns)
        li = pd.Series(lmodel.feature_importances_, index=Xlgb_tr.columns)
        fi_cat = ci if fi_cat is None else fi_cat.add(ci, fill_value=0)
        fi_lgb = li if fi_lgb is None else fi_lgb.add(li, fill_value=0)

        fold_mse = {m: mean_squared_error(yval, oof[m][va]) for m in models}
        print(f"  Fold {k:>2}/{len(folds)}  "
              + "  ".join(f"{m}={fold_mse[m]:.3f}" for m in models)
              + f"   ({time.time()-t0:.1f}s)")

    fi_cat = (fi_cat / len(folds)).sort_values(ascending=False)
    fi_lgb = (fi_lgb / len(folds)).sort_values(ascending=False)
    return oof, test_pred, fi_cat, fi_lgb


# ----------------------------------------------------------------------------------------
# 7) Ensemble agirlik optimizasyonu (OOF uzerinde, konveks)
# ----------------------------------------------------------------------------------------
def optimize_ensemble_weights(oof: dict, y, init=None):
    names = list(oof.keys())
    P = np.column_stack([oof[m] for m in names])
    yv = y.values

    def loss(w):
        return mean_squared_error(yv, P @ w)

    n = len(names)
    if init is None:
        init = np.array([0.35, 0.25, 0.15, 0.15, 0.10][:n])
        init = init / init.sum()
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(loss, init, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    w = res.x if res.success else init
    w = np.clip(w, 0, None); w = w / w.sum()
    weights = {names[i]: float(w[i]) for i in range(n)}
    ens_oof = P @ w
    return weights, float(mean_squared_error(yv, ens_oof)), ens_oof


# ----------------------------------------------------------------------------------------
# 8) Submission + cikti dosyalari
# ----------------------------------------------------------------------------------------
def create_submission(test_ids, ensemble_test_pred, sample, n_expected_rows):
    pred = np.clip(ensemble_test_pred, CLIP_LOW, CLIP_HIGH)
    sub = pd.DataFrame({ID_COL: test_ids.values, TARGET: pred})

    # --- format dogrulama ---
    assert len(sub) == n_expected_rows, \
        f"Submission satir sayisi ({len(sub)}) test ({n_expected_rows}) ile eslesmiyor!"
    assert list(sub.columns) == [ID_COL, TARGET], f"Kolon isimleri hatali: {list(sub.columns)}"
    assert not sub[TARGET].isna().any(), "Submission'da NaN tahmin var!"
    assert sub[TARGET].between(CLIP_LOW, CLIP_HIGH).all(), "Tahminler [0,100] disinda!"
    # sample_submission ile ayni kolon basliklari
    assert list(sample.columns) == [ID_COL, TARGET], \
        f"sample_submission kolon basliklari beklenenden farkli: {list(sample.columns)}"

    out = HERE / "submission.csv"
    sub.to_csv(out, index=False, encoding="utf-8")
    return out, sub


def write_readme(report: dict, weights: dict, model_scores: pd.DataFrame):
    lines = []
    lines.append("# BTK Datathon 2026 — career_success_score Çözümü\n")
    lines.append("## Görev\n`test_x.csv` içindeki öğrenciler için `career_success_score` "
                 "(0–100, sürekli) tahmini. Metrik: **MSE**.\n")
    lines.append("## Kullanılan Modeller\n")
    lines.append("- **CatBoostRegressor** — sayısal + mühendislik özellikleri + **native kategorik** "
                 "+ TF-IDF SVD; RMSE loss, early stopping.")
    lines.append("- **LightGBMRegressor** — sayısal + mühendislik + kodlanmış kategorik + SVD; early stopping.")
    lines.append("- **XGBoostRegressor** — LightGBM ile aynı matris; early stopping.")
    lines.append("- **TF-IDF Ridge** — `mentor_feedback_text` üzerinde word(1,2)+char_wb(3,5) "
                 f"TF-IDF → Ridge (alpha={RIDGE_ALPHA:g}, ölçeklemesiz; ensemble'da stacking üyesi).\n")
    lines.append("## Özellik Mühendisliği (özet)\n")
    lines.append("Akademik (years_to_graduation, academic_mean, attendance_failed_interaction…), "
                 "teknik beceri agregasyonları (mean/max/min/std/range/sum), soft-skill agregasyonları, "
                 "mülakat & kariyer hazırlığı (interview_mean, interview_gap, career_readiness_mean), "
                 "proje/deneyim (github_power, opensource_power, hackathon_success_rate, "
                 "internship_intensity…), başvuru verimliliği (interview_rate…), "
                 "`role_match_score` (target_role'a göre ağırlıklı teknik skor), oran/etkileşim "
                 "(technical_soft_ratio, career_total_score, balanced_profile_score) ve "
                 "bilgi taşıyan eksiklik için `*_was_missing` bayrakları.\n")
    lines.append("## CV Stratejisi\n")
    lines.append(f"- `StratifiedKFold` (n_splits={N_FOLDS}), hedef `pd.qcut` ile 10 bine ayrılarak "
                 "stratifiye edildi.\n"
                 "- Tüm TF-IDF/SVD/metin modeli **fold içinde** fit edildi → data leakage yok.\n"
                 "- Test tahmini fold ortalaması; tahminler [0,100] aralığına clip edildi.\n"
                 "- Hiperparametreler Optuna ile (iç 3-fold CV) OOF MSE'ye göre seçildi; "
                 "public leaderboard'a göre ayar yapılmadı.\n")
    lines.append("## Skorlar (OOF / CV MSE)\n")
    lines.append("| Aşama / Model | CV MSE |")
    lines.append("|---|---|")
    lines.append(f"| Varyans baseline (ortalama tahmin) | {report['variance_baseline_mse']:.3f} |")
    lines.append(f"| Baseline LightGBM (FE/metin yok) | {report['baseline_mse']:.3f} |")
    for _, r in model_scores.iterrows():
        lines.append(f"| {r['model']} | {r['oof_mse']:.3f} |")
    lines.append(f"| **Ensemble (final)** | **{report['ensemble_mse']:.3f}** |")
    lines.append("")
    lines.append("## Final Ensemble Ağırlıkları\n")
    for m, w in weights.items():
        lines.append(f"- {m}: {w:.3f}")
    lines.append("")
    lines.append("## Kurallara Uygunluk\n")
    lines.append("Test seti elle etiketlenmedi, hedef-tahmin hilesi yapılmadı, data leakage yok "
                 "(tüm fit fold-içi), dışarıdan gizli veri kullanılmadı, başka takımlardan kod/veri "
                 "alınmadı, public leaderboard'a göre manuel ayar yapılmadı. `student_id` modelde "
                 "kullanılmadı (leakage analizi: korelasyon ≈ −0.009).\n")
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
def main():
    set_seed(SEED)
    sys.stdout.reconfigure(encoding="utf-8")
    t_start = time.time()

    print("=" * 78)
    print("BTK Datathon 2026 — career_success_score pipeline")
    print("=" * 78)

    # --- yukle ---
    train, test, sample = load_data()
    y = train[TARGET].astype(float)
    train_ids, test_ids = train[ID_COL], test[ID_COL]

    # (1) shape
    print(f"\n[1] Sekiller: train={train.shape}  test={test.shape}  "
          f"sample_submission={sample.shape}  (sample yalnizca format sablonu)")

    # (2) eksik ozet
    miss = train.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print("\n[2] Eksik deger ozeti (train):")
    for c, v in miss.items():
        print(f"    {c:34s} {v:6d}  ({100*v/len(train):.2f}%)")

    # (3) kolon tipleri
    num_orig = [c for c in EXPECTED_COLS if c not in CAT_COLS + [ID_COL, TEXT_COL]]
    print(f"\n[3] Kolonlar: {len(num_orig)} sayisal, {len(CAT_COLS)} kategorik "
          f"({CAT_COLS}), 1 metin ({TEXT_COL})")

    # (4) hedef dagilim
    print(f"\n[4] Hedef '{TARGET}': min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f} "
          f"std={y.std():.2f} median={y.median():.2f}")
    var_baseline = float(((y - y.mean()) ** 2).mean())
    print(f"    Varyans baseline MSE (ortalama tahmin) = {var_baseline:.3f}")

    # --- ozellik muhendisligi (train+test birlesik, satir-bazli -> leakage'siz) ---
    train_X = train.drop(columns=[TARGET])
    n_train = len(train_X)
    combined = pd.concat([train_X, test], ignore_index=True)
    combined = create_features(combined)
    combined = encode_categoricals(combined)
    base_train = combined.iloc[:n_train].reset_index(drop=True)
    base_test = combined.iloc[n_train:].reset_index(drop=True)
    text_train = base_train[TEXT_COL]
    text_test = base_test[TEXT_COL]
    print(f"\n    Ozellik muhendisligi sonrasi toplam kolon: {combined.shape[1]} "
          f"(sayisal+muhendislik+kod+metin)")

    # --- CV foldlari (StratifiedKFold, hedef qcut bin) ---
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(n_train), bins))

    # (5) baseline CV MSE
    print("\n[5] Baseline CV (LightGBM, FE/metin YOK) hesaplaniyor...")
    baseline_mse = run_baseline(base_train, y, folds)
    print(f"    Baseline CV MSE = {baseline_mse:.3f}")

    # --- Optuna tuning (sayisal+kodlu kategorik; metin haric -> hizli) ---
    lgbm_params, cat_params = default_lgbm_params(), default_cat_params()
    xgb_params = default_xgb_params()
    if DO_OPTUNA:
        code_cols = ["university_tier_code"] + [f"{c}_code" for c in CAT_COLS if c != "university_tier"]
        num_cols = [c for c in base_train.columns
                    if c not in CAT_COLS + [ID_COL, TEXT_COL] and base_train[c].dtype != object]
        num_cols = [c for c in num_cols if c not in code_cols]
        X_tune_lgb = base_train[num_cols + code_cols]
        X_tune_cat = base_train[num_cols + CAT_COLS].copy()
        for c in CAT_COLS:
            X_tune_cat[c] = X_tune_cat[c].astype(str).fillna("NA")
        print("\n[*] Optuna tuning (LightGBM)...")
        try:
            lgbm_params = tune_lgbm(X_tune_lgb, y, code_cols)
        except Exception as e:
            print(f"    Optuna LGBM atlandi ({e}); varsayilan parametreler kullanilacak.")
        print("[*] Optuna tuning (CatBoost)...")
        try:
            cat_params = tune_catboost(X_tune_cat, y, CAT_COLS)
        except Exception as e:
            print(f"    Optuna CatBoost atlandi ({e}); varsayilan parametreler kullanilacak.")

    # --- ana CV: 4 model, fold-ici metin (6,7) ---
    print(f"\n[6/7] {N_FOLDS}-fold ana CV (CatBoost + LightGBM + XGBoost + TF-IDF Ridge)...")
    oof, test_pred, fi_cat, fi_lgb = run_cv(base_train, base_test, text_train, text_test, y,
                                            folds, lgbm_params, cat_params, xgb_params)

    model_mse = {m: float(mean_squared_error(y, oof[m])) for m in oof}
    print("\n    Model OOF CV MSE:")
    for m, v in sorted(model_mse.items(), key=lambda kv: kv[1]):
        print(f"      {m:10s} MSE={v:.3f}  RMSE={np.sqrt(v):.3f}")

    fe_best_single = min(model_mse.values())

    # (8) ensemble
    weights, ens_mse, ens_oof = optimize_ensemble_weights(oof, y)
    print(f"\n[8] Ensemble agirliklari: " + "  ".join(f"{m}={w:.3f}" for m, w in weights.items()))
    print(f"    Ensemble OOF CV MSE = {ens_mse:.3f}  (RMSE={np.sqrt(ens_mse):.3f})")

    ens_test = sum(weights[m] * test_pred[m] for m in weights)

    # (9) submission + ciktilar
    sub_path, sub = create_submission(test_ids, ens_test, sample, len(test))
    print(f"\n[9] Submission yazildi: {sub_path}")

    # oof_predictions.csv
    oof_df = pd.DataFrame({ID_COL: train_ids.values})
    for m in oof:
        oof_df[f"oof_{m}"] = oof[m]
    oof_df["oof_ensemble"] = ens_oof
    oof_df[TARGET] = y.values
    oof_df.to_csv(HERE / "oof_predictions.csv", index=False, encoding="utf-8")

    # per-model test tahminleri (seffaflik + ileride ucuz ensemble denemesi icin)
    tp_df = pd.DataFrame({ID_COL: test_ids.values})
    for m in test_pred:
        tp_df[f"test_{m}"] = test_pred[m]
    tp_df["test_ensemble"] = np.clip(ens_test, CLIP_LOW, CLIP_HIGH)
    tp_df.to_csv(HERE / "test_predictions.csv", index=False, encoding="utf-8")

    # model_scores.csv
    rows = [{"model": m, "oof_mse": model_mse[m], "oof_rmse": np.sqrt(model_mse[m])} for m in oof]
    rows.append({"model": "ensemble", "oof_mse": ens_mse, "oof_rmse": np.sqrt(ens_mse)})
    rows.append({"model": "baseline_lgbm_no_fe", "oof_mse": baseline_mse,
                 "oof_rmse": np.sqrt(baseline_mse)})
    rows.append({"model": "variance_baseline_mean", "oof_mse": var_baseline,
                 "oof_rmse": np.sqrt(var_baseline)})
    model_scores = pd.DataFrame(rows)
    model_scores.to_csv(HERE / "model_scores.csv", index=False, encoding="utf-8")

    # feature importance
    fi_cat.rename("importance").to_csv(HERE / "feature_importance_catboost.csv",
                                       index_label="feature", encoding="utf-8")
    fi_lgb.rename("importance").to_csv(HERE / "feature_importance_lgbm.csv",
                                       index_label="feature", encoding="utf-8")

    # README
    report = dict(variance_baseline_mse=var_baseline, baseline_mse=baseline_mse,
                  fe_best_single_mse=fe_best_single, ensemble_mse=ens_mse)
    model_scores_for_readme = pd.DataFrame(
        [{"model": m, "oof_mse": model_mse[m]} for m in oof])
    write_readme(report, weights, model_scores_for_readme)

    # --- ozet rapor ---
    print("\n" + "=" * 78)
    print("OZET RAPOR")
    print("=" * 78)
    print(f"1) train/test shape           : {train.shape} / {test.shape}")
    print(f"2) eksik kolon sayisi (train) : {len(miss)}")
    print(f"3) sayisal/kategorik/metin    : {len(num_orig)} / {len(CAT_COLS)} / 1")
    print(f"4) hedef mean/std             : {y.mean():.2f} / {y.std():.2f}  (varyans MSE={var_baseline:.2f})")
    print(f"5) baseline CV MSE            : {baseline_mse:.3f}")
    print(f"6) FE sonrasi en iyi tek model: {fe_best_single:.3f}")
    print(f"7) her model CV MSE           : " + ", ".join(f"{m}={model_mse[m]:.3f}" for m in model_mse))
    print(f"8) ensemble CV MSE            : {ens_mse:.3f}")
    print(f"9) submission yolu            : {sub_path}")
    print(f"\nToplam sure: {(time.time()-t_start)/60:.1f} dk")
    print("Cikti dosyalari: submission.csv, oof_predictions.csv, model_scores.csv,")
    print("                 feature_importance_catboost.csv, feature_importance_lgbm.csv, README.md")


if __name__ == "__main__":
    main()
