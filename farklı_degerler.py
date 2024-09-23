import pandas as pd

# CSV dosyasının yolunu belirleyin
csv_dosyasi = r"C:\Users\beyza\OneDrive\Masaüstü\datathon\temizlenmis_veri.csv"

# CSV dosyasını yükleyin
df = pd.read_csv(csv_dosyasi)

# Sonuçları tutacağımız bir DataFrame oluşturuyoruz
sonuc_df = pd.DataFrame(columns=["Sütun Adı", "Eşsiz Değer Sayısı", "Eşsiz Değerler"])

# Her sütun için eşsiz değer sayısını ve değerlerin kendilerini bulma
for column in df.columns:
    # Sütundaki eşsiz değerleri bul
    esiz_degerler = df[column].dropna().unique()
    
    # Sütundaki eşsiz değerlerin sayısını al
    esiz_deger_sayisi = len(esiz_degerler)
    
    # Eşsiz değerlerin sayısı 500'den küçükse tümünü yazdır, büyükse sadece sayısını yaz
    if esiz_deger_sayisi <= 500:
        esiz_degerler_str = ' // '.join(map(str, esiz_degerler))  # Eşsiz değerleri // ile ayır
    else:
        esiz_degerler_str = "500'den fazla değer var"
    
    # Sütun adı, eşsiz değer sayısı ve eşsiz değerleri yeni DataFrame'e ekle
    sonuc_df = sonuc_df._append({
        "Sütun Adı": column,
        "Eşsiz Değer Sayısı": esiz_deger_sayisi,
        "Eşsiz Değerler": esiz_degerler_str
    }, ignore_index=True)

# Sonuçları yeni bir CSV dosyasına kaydetme
output_file = r"C:\Users\beyza\OneDrive\Masaüstü\datathon\sutun_esiz_degerler.csv"
sonuc_df.to_csv(output_file, index=False)

# Sonuçları ekrana yazdırma
for index, row in sonuc_df.iterrows():
    print(f"Sütun: {row['Sütun Adı']}, Eşsiz Değer Sayısı: {row['Eşsiz Değer Sayısı']}")
    print(f"Eşsiz Değerler: {row['Eşsiz Değerler']}\n")

print(f"Eşsiz değerler ve sayıları '{output_file}' dosyasına kaydedildi.")

