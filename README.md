# BTK Datathon 2026 — Kariyer Başarı Skoru Tahmini

BTK Datathon 2026 için 2 kişilik ekip olarak hazırladığımız çözüm. Amaç, öğrencilerin akademik, teknik ve sosyal verilerinden `career_success_score` (0–100) değerini tahmin etmek. Değerlendirme metriği MSE.

## Yaklaşım

- 47 ham kolondan 91 özellik türetildi: akademik ortalamalar, beceri skorlarının istatistikleri, mülakat ve proje göstergeleri, eksik değer bayrakları
- Mentor geri bildirim metni için TF-IDF + SVD
- CatBoost, LightGBM, XGBoost ve Ridge modelleri, 10 katlı stratified cross-validation
- Modeller ağırlıklı ortalama ile birleştirildi; tüm dönüşümler fold içinde yapıldı (veri sızıntısı yok)

## Sonuçlar (CV MSE)

| Model | MSE |
|---|---|
| Ortalama tahmin (baseline) | 230.61 |
| LightGBM, özellik mühendisliği olmadan | 85.59 |
| CatBoost | 79.68 |
| LightGBM | 79.31 |
| XGBoost | 80.73 |
| **Ensemble** | **78.42** |

Ayrıntılı analiz: [ANALYSIS.md](ANALYSIS.md)

## Çalıştırma

Yarışma verilerini (`train.csv`, `test_x.csv`, `sample_submission.csv`) proje klasörüne koyun. Veri seti repoya eklenmedi.

```bash
pip install -r requirements.txt
python pipeline.py      # modelleri eğitir, submission.csv üretir
python smoke_test.py    # küçük veriyle hızlı kontrol
```

## Dosyalar

| Dosya | İçerik |
|---|---|
| `pipeline.py` | Veri hazırlama, özellik mühendisliği, eğitim ve ensemble |
| `improve_v2.py` | Kayıtlı tahminlere ek modellerle blend denemesi |
| `screen_models.py` | Ensemble'a eklenecek aday modellerin taranması |
| `smoke_test.py` | Küçük veriyle hızlı test |
| `model_scores.csv` | Modellerin CV skorları |
| `feature_importance_*.csv` | Özellik önemleri |
