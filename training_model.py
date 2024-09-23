import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Veri Yükleme
file_path = 'train_veri_temizlenmis.xlsx' #temizlenmiş train excel dosyası
df = pd.read_excel(file_path)

# Label Encoding: Kategorik değişkenleri sayısal değerlere dönüştürme
label_encoders = {}
combined_df = df.drop(columns=['Degerlendirme Puani'])

# Eğitim veri setinde kategorik değişkenleri encode etme
for column in combined_df.select_dtypes(include=['object']).columns:
    label_enc = LabelEncoder()
    combined_df[column] = label_enc.fit_transform(combined_df[column].astype(str))
    label_encoders[column] = label_enc  # Her sütun için encoder saklanır

# Özellikler ve hedef değişken
X = combined_df
y = df['Degerlendirme Puani']

# Eksik Değerleri Doldurma (Imputation)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Eğitim verisindeki NaN değerleri doldur
y = y.fillna(y.mean())  # Eksik puanları ortalama ile doldurabilirsiniz

# Model Eğitimi
lgb_model = lgb.LGBMRegressor(
    learning_rate=0.1,   # Öğrenme oranı
    n_estimators=300,    # Ağaç sayısı (en iyi parametre)
    num_leaves=31,       # En iyi parametre
    max_depth=20,        # En iyi parametre
    random_state=42
)
lgb_model.fit(X_imputed, y)

# Test Verisini Yükleme ve Tahmin Yapma
test_file_path = 'test_veri_temizlenmis.xlsx'  #temizlenmiş train excel dosyası
test_df = pd.read_excel(test_file_path)

# Test verisindeki kategorik değişkenleri sayısal değerlere dönüştürme
test_combined_df = test_df.copy()

for column in test_combined_df.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        # Test verisindeki kategorik değerleri eğitim verisindeki etiketler ile eşleştirme
        encoder = label_encoders[column]
        # Eğitim veri setinde olmayan değerler test verisinde olabilir, bu yüzden bilinmeyenleri NaN yapacağız
        test_combined_df[column] = test_combined_df[column].apply(lambda x: x if x in encoder.classes_ else np.nan)
        test_combined_df[column] = encoder.transform(test_combined_df[column].astype(str))
    else:
        test_combined_df[column] = test_combined_df[column].astype(str)  # Bilinmeyen kategori varsa string olarak bırak

# Test verisindeki eksik değerleri doldurmak
test_df_imputed = imputer.transform(test_combined_df)

# Tahmin yapma
test_predictions = lgb_model.predict(test_df_imputed)

# Sonuçları CSV Dosyasına Kaydetme
results_df = pd.DataFrame({
    'id': test_df.index,  # id'yi 0'dan başlatmak
    'Degerlendirme Puani': np.round(test_predictions, 1)  # Virgülden sonra bir basamak
})

results_df.to_csv('test_predictions.csv', index=False) #test dosyasının model sonuçları çıktı olarak verilecektir.
