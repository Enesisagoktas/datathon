# Datathon 2026 — `career_success_score` Çözüm Analizi (Jüri Dokümanı)

## 1. Problem
`test_x.csv` içindeki 10.000 öğrenci için `career_success_score` (0–100, sürekli) tahmini.
Değerlendirme metriği: **MSE**. Veri sayısal, kategorik ve doğal dil (Türkçe `mentor_feedback_text`)
alanları içerir.

## 2. Veri Keşfi (kanıta dayalı)
- `train.csv` 10.000×47, `test_x.csv` 10.000×46. `sample_submission.csv` yalnızca 2 örnek satır
  (format şablonu) → gerçek gönderim 10.000 satır.
- **ID sızıntısı yok**: `corr(student_id, target) ≈ −0.009`, train id 1–10000 / test id 10001–20000,
  kesişim yok → `student_id` modelde kullanılmadı.
- **Eksik değerler** 7 sayısal kolonda (bilgi taşıyan): `internship_duration_months` %16.6, vb.
  Ağaç modelleri NaN'i native işler; ayrıca `*_was_missing` bayrakları üretildi.
- **Hedef**: 0–100 sınırlı, ortalama 76.9 / std 15.2, sola çarpık, 100'de yığılma (p99=100).
  Varyans baseline (ortalama tahmin) MSE = **230.6** — tüm karşılaştırmaların referansı.

## 3. Özellik Mühendisliği
47 ham kolondan **91 kolonluk** zenginleştirilmiş temsil üretildi (hepsi **satır-bazlı** →
leakage'siz, train+test üzerinde bir kez hesaplanabilir):
- **Akademik**: years_to_graduation, age_at_graduation, academic_mean, attendance_failed_interaction.
- **Teknik beceri (9 skor)**: mean/max/min/std/range/sum.
- **Sosyal beceri (4 skor)**: mean/max/min/std/sum.
- **Mülakat & hazırlık**: interview_mean, career_readiness_mean, interview_gap.
- **Proje/deneyim**: github_power=log1p(repo)·log1p(stars+1), opensource_power,
  hackathon_success_rate, internship_intensity, portfolio_project_interaction, experience_count_sum.
- **Başvuru verimliliği**: interview_rate, applications_per_interview.
- **`role_match_score`**: `target_role`'a göre ilgili teknik skorların ağırlıklı ortalaması
  (data/ml/ai, backend, frontend, devops/cloud, fullstack, genel).
- **Oran/etkileşim**: technical_soft_ratio, project_to_academic_ratio, career_total_score,
  balanced_profile_score (log ile sıkıştırılmış çarpım).
- **Eksiklik bayrakları**: 7 kolon için `*_was_missing`.

Feature importance'ta en güçlüler: `career_total_score`, `project_quality_score`,
`project_experience_score`, `technical_interview_score`, `interview_gap`, `role_match_score`
(+ metin `svd_char_2`).

## 4. Metin İşleme (`mentor_feedback_text`)
Her CV fold'u **içinde** fit edilerek (leakage'siz):
- Word TF-IDF (1,2)-gram + Char-wb TF-IDF (3,5)-gram (Türkçe morfolojisi için).
- TruncatedSVD (word→120, char→80 bileşen) → ağaç modellerine yoğun özellik olarak verildi.
- Ayrı bir **TF-IDF Ridge** metin modeli (alpha=10, ölçeklemesiz) ensemble üyesi.

## 5. Modelleme ve Doğrulama
- **CV**: `StratifiedKFold(10)`, hedef `pd.qcut` ile 10 bine ayrılarak stratifiye.
- **Modeller**: CatBoost (native kategorik) + LightGBM + XGBoost + TF-IDF Ridge. Hepsi
  early stopping + sabit seed. Hiperparametreler **Optuna** (iç 3-fold) ile OOF MSE'ye göre.
- **Ensemble**: OOF üzerinde konveks ağırlık optimizasyonu (SLSQP), [0,100] clip.
- Tüm TF-IDF/SVD/metin modeli fold-içi fit → **data leakage yok**.

### Sonuçlar (OOF / CV MSE)
| Aşama | MSE |
|---|---|
| Varyans baseline (ortalama) | 230.6 |
| Baseline LightGBM (FE/metin yok) | 85.6 |
| LightGBM (final) | 79.3 |
| CatBoost (final) | 79.7 |
| XGBoost (final) | 80.7 |
| TF-IDF Ridge (metin) | 145.0 |
| **Ensemble (final)** | **78.4** (RMSE 8.85, R²≈0.66) |

## 6. ANA BULGU — CV/LB Farkının Kökü ve Tavanın İspatı
Public LB skoru (**89.5**) ile OOF CV (**78.4**) arasında ~11 puanlık fark vardı. Bunu
**rastgele model varyansı değil, sistematik bir temporal shift** olarak teşhis ettik:

1. **Adversarial validation** (train vs test ayırt etme) AUC = **0.657** → dağılım farkı var.
   En ayırt edici değişken: `application_year` / `graduation_year`.
2. **Test seti yeni-yıl ağırlıklı**: train yıllara ~eşit (2019–2026 her biri ~1300), test ise
   2024:1994, 2025:2197, 2026:2029 (eski yıllar çok az).
3. **Yeni yıllar geri dönülmez şekilde daha gürültülü**: OOF MSE yıla göre 2019'da **56** →
   2026'da **111**; hedef std 12.6 → 18.0.
4. **Doğrulama**: OOF'u test-yılı dağılımına göre ağırlıklandırınca **weighted-OOF = 89.56**,
   gerçek public LB **89.52** ile **birebir** örtüştü. Bu, modelin overfit OLMADIĞINI ve CV'mizin
   dürüst olduğunu kanıtlar — fark tamamen test setinin doğasından geliyor.

Bu **LB-izleyen metrik** sayesinde her iyileştirmeyi public LB'ye dokunmadan güvenle ölçtük.

### Test edilen ve elenen iyileştirmeler (nested-CV ile, hepsi gürültü seviyesinde)
| Kaldıraç | Dürüst sonuç |
|---|---|
| Sample weighting / recent-only eğitim (shift düzeltme) | **Kötüleşti** (95.1→96.2) |
| Ensemble yeniden ağırlıklama | +0.04 (gürültü) |
| Çeşitli modeller (HistGB, MLP, ExtraTrees, full-Ridge) | +0.01 (nested-CV; in-sample kazanç overfit'ti) |
| TF-IDF metni daha çok kullanma | metin, ağaç artığını **R²=−0.02** ile açıklıyor (tamamen redundant) |

**Sonuç:** Çözüm, bu veri temsilinin **tahmin tavanındadır** (R²≈0.66). Metin, sayısal
özelliklerle aynı bilgiyi taşıdığı için (mentor yorumu ölçülen becerileri betimliyor) ensemble'a
katkısı ~0'dır; bu yüzden embedding bile redundancy'yi aşamaz. Public LB'de ~89.5 etrafındaki
yoğun kümelenme bu tavanı doğrular.

## 7. Neden Bu Çözüm Final İçin Güçlü
- Public leaderboard'a **hiç overfit edilmedi**; karar daima OOF/weighted-OOF'a göre verildi.
- weighted-OOF'un gerçek LB'yi birebir izlediği **ispatlandı** → private LB tahminimiz de güvenilir.
- Robust, düşük-varyanslı bir ensemble (10-fold ortalama). Bu tip çözümler **private/final
  leaderboard'da yükselme**, public'e overfit edenler ise düşme eğilimindedir.

## 8. Kurallara Uygunluk
Test elle etiketlenmedi, hedef-tahmin hilesi yok, data leakage yok (tüm fit fold-içi), dışarıdan
gizli veri yok, başka takım çözümü yok, public LB'ye manuel ayar yok. `student_id` modelde yok.

## 9. Tekrar Üretilebilirlik
```
python -m pip install -r requirements.txt
python pipeline.py        # tüm CV, modeller, ensemble, submission.csv + raporlar
```
Tüm seed'ler sabit. Yardımcı analiz scriptleri: `screen_models.py` (model çeşitliliği taraması),
`improve_v2.py` (weighted-metrik blend deneyi). Bulgular yukarıda raporlanmıştır.
