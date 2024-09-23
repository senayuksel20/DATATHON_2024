import pandas as pd
import re

# CSV dosyasının yolunu belirtin
csv_dosyasi = r"temizlenmis_veri.csv" #temizlenmiş veri csv dosyasının yolunu buraya girmeniz gerekmemtedir
# CSV dosyasını oku
df = pd.read_csv(csv_dosyasi)


# Doğum tarihi yılını almak için saat bilgisini çıkar ve . ile böl
def temizle_dogum_yili(dogum_tarihi):
    try:
        # Saat bilgisini çıkar (örn: 00:00 veya başka saatler)
        temiz_tarih = re.sub(r'\s*\d{1,2}(:|.)\d{2}\s*$', '', dogum_tarihi).strip()
        # . ile böl ve 2. indexteki yılı al
        tarih_parcalari = temiz_tarih.split('.')
        return tarih_parcalari[2] if len(tarih_parcalari) >= 3 else None
    except:
        return None  # Hatalı format veya eksik veri varsa None döndür

# Yaş sütunu ekleme (Doğum yılı ve Başvuru Yılı kullanılarak)
def hesapla_yas(dogum_tarihi, basvuru_yili):
    try:
        # Doğum yılını al
        dogum_yili = int(temizle_dogum_yili(dogum_tarihi))
        yas = int(basvuru_yili) - dogum_yili
        return yas
    except:
        return None  # Hatalı format veya eksik veri varsa None döndür

# Yaşı hesapla ve yeni bir sütuna ekle
df['Yaş'] = df.apply(lambda row: hesapla_yas(row['Dogum Tarihi'], row['Basvuru Yili']), axis=1)

silinecek_sutunlar = [
    "Hangi STK'nin Uyesisiniz?",
    'Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?',
    'Dogum Tarihi',
    'Burslu ise Burs Yuzdesi',
    'Daha Once Baska Bir Universiteden Mezun Olmus',
    'Lise Adi Diger',
    'Lise Bolum Diger',
    'Uye Oldugunuz Kulubun Ismi',
    'Stk Projesine Katildiniz Mi?',
    'Ingilizce Seviyeniz?',
    'Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite',
    'id',
    'Unnamed: 43'

]

df = df.drop(columns=[sutun for sutun in silinecek_sutunlar if sutun in df.columns])

# Hücrelerdeki uzun metinleri kısaltma
df = df.applymap(lambda x: str(x)[:32767] if isinstance(x, str) else x)

# Excel dosyasına yaz, XlsxWriter kullan
#excel dosyasının kaydedilmesini istediğiniz dosya yolunu belirleyiniz
excel_dosyasi = r"train_veri_temizlenmis_artı_unk.xlsx" #yapılan değişiklikler sonrasında verinin son hali excel dosyasında kaydedilir
with pd.ExcelWriter(excel_dosyasi, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)

print("Excel dosyası başarıyla oluşturuldu!")
