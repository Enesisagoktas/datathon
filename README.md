# BTK Datathon 2026 — career_success_score Çözümü

## Görev
`test_x.csv` içindeki öğrenciler için `career_success_score` (0–100, sürekli) tahmini. Metrik: **MSE**.

## Kullanılan Modeller

- **CatBoostRegressor** — sayısal + mühendislik özellikleri + **native kategorik** + TF-IDF SVD; RMSE loss, early stopping.
- **LightGBMRegressor** — sayısal + mühendislik + kodlanmış kategorik + SVD; early stopping.
- **XGBoostRegressor** — LightGBM ile aynı matris; early stopping.
- **TF-IDF Ridge** — `mentor_feedback_text` üzerinde word(1,2)+char_wb(3,5) TF-IDF → Ridge (alpha=10, ölçeklemesiz; ensemble'da stacking üyesi).

## Özellik Mühendisliği (özet)

Akademik (years_to_graduation, academic_mean, attendance_failed_interaction…), teknik beceri agregasyonları (mean/max/min/std/range/sum), soft-skill agregasyonları, mülakat & kariyer hazırlığı (interview_mean, interview_gap, career_readiness_mean), proje/deneyim (github_power, opensource_power, hackathon_success_rate, internship_intensity…), başvuru verimliliği (interview_rate…), `role_match_score` (target_role'a göre ağırlıklı teknik skor), oran/etkileşim (technical_soft_ratio, career_total_score, balanced_profile_score) ve bilgi taşıyan eksiklik için `*_was_missing` bayrakları.

## CV Stratejisi

- `StratifiedKFold` (n_splits=10), hedef `pd.qcut` ile 10 bine ayrılarak stratifiye edildi.
- Tüm TF-IDF/SVD/metin modeli **fold içinde** fit edildi → data leakage yok.
- Test tahmini fold ortalaması; tahminler [0,100] aralığına clip edildi.
- Hiperparametreler Optuna ile (iç 3-fold CV) OOF MSE'ye göre seçildi; public leaderboard'a göre ayar yapılmadı.

## Skorlar (OOF / CV MSE)

| Aşama / Model | CV MSE |
|---|---|
| Varyans baseline (ortalama tahmin) | 230.612 |
| Baseline LightGBM (FE/metin yok) | 85.592 |
| catboost | 79.680 |
| lgbm | 79.306 |
| xgb | 80.728 |
| ridge | 145.025 |
| **Ensemble (final)** | **78.423** |

## Final Ensemble Ağırlıkları

- catboost: 0.424
- lgbm: 0.482
- xgb: 0.073
- ridge: 0.021

## Kurallara Uygunluk

Test seti elle etiketlenmedi, hedef-tahmin hilesi yapılmadı, data leakage yok (tüm fit fold-içi), dışarıdan gizli veri kullanılmadı, başka takımlardan kod/veri alınmadı, public leaderboard'a göre manuel ayar yapılmadı. `student_id` modelde kullanılmadı (leakage analizi: korelasyon ≈ −0.009).
