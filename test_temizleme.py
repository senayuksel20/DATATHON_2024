import pandas as pd
from unidecode import unidecode
import re
import string
import jpype
from nltk.corpus import stopwords
from jpype import JClass, startJVM, getDefaultJVMPath
import numpy as np

# CSV dosyasının giriş ve çıkış yolunu belirtin
input_file_path = r"C:\Users\beyza\OneDrive\Masaüstü\datathon\test_x.csv"
output_file_path = r"C:\Users\beyza\OneDrive\Masaüstü\datathon\test_x_temizleme.csv"

# CSV dosyasını oku
df = pd.read_csv(input_file_path)

# Belirtilen sütunlar işlemden hariç tutulacak
columns_to_exclude = [
    "Hangi STK'nin Uyesisiniz?",
    "Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"
]

# Hariç tutulacak sütunlar dışındaki sütunları seçin
columns_to_process = [col for col in df.columns if col not in columns_to_exclude]

# Seçilen sütunlardaki metinleri küçük harfe çevirin ve Türkçe karakterleri İngilizce'ye dönüştürün
df[columns_to_process] = df[columns_to_process].applymap(lambda x: unidecode(str(x)).lower() if isinstance(x, str) else x)

# ₺ sembolünü geri getir (l harfini ₺ sembolüne dönüştür)
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].replace({'l':'₺'}, regex=True)

# "Universite Adi" sütunundaki "universitesi" kelimesini kaldır
df['Universite Adi'] = df['Universite Adi'].str.replace(' universitesi', '', case=False)

# "Bölüm" sütunundaki "bolumu" ve "alani" kelimelerini kaldır
df['Bölüm'] = df['Bölüm'].replace({
    ' bolumu': '',
    ' alani': ''
})

# Burs terimlerini içeren bir liste
burs_terimleri = [
    "kyk", "t3", "tübitak", "tev", "universite", "gsb", "vakfı", "tobb", "derneği", 
    "dernek", "özel", "meb", "vakıf", "yok", "tskev", "tanıdık", "nevader", 
    "bakanlıgı", "kulup", "yasar", "odtu", "depremzede", "geri odemeli", 
    "kredi", "ksv", "okul", "kök", "derece burs"
]

# Belirli terimleri içeren metinleri döndüren fonksiyon
def format_burs_terimleri(text, burs_terimleri):
    # "kredi ve yurtlar" terimini içeriyorsa "kyk" olarak ayarla
    if "kredi ve yurtlar" in text:
        return "kyk"
    results = [term for term in burs_terimleri if term in text]
    if results:
        return ', '.join(results)  # Terimleri cümle gibi döndür
    else:
        return text  # Hiçbir terim yoksa metni olduğu gibi bırak

# Burs terimlerini bul ve yeni bir sütun oluştur
df['Burs Aldigi Baska Kurum'] = df['Burs Aldigi Baska Kurum'].apply(lambda x: format_burs_terimleri(x, burs_terimleri))

# "Lise Adı" sütunundaki 'lisesi' ve 'koleji' kelimelerini kaldır
df['Lise Adi'] = df['Lise Adi'].str.replace(' lisesi', '', case=False)
df['Lise Adi'] = df['Lise Adi'].str.replace(' koleji', '', case=False)

# Türkçe stopwords setini tanımlayın
stop_words = set(stopwords.words('turkish'))

# Zemberek'i başlatın
ZEMBEREK_PATH = r"C:\Users\beyza\Downloads\zemberek-full (3).jar"
if not jpype.isJVMStarted():
    startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))

# Zemberek sınıflarını yükleyin
morphology = JClass('zemberek.morphology.TurkishMorphology').createWithDefaults()

# Metni lemmatize eden fonksiyon
def lemmatize_text(text):
    tokens = text.split()
    lemmatized_tokens = []
    for token in tokens:
        if token.lower() == 'unk':  # Eğer token 'UNK' ise olduğu gibi ekle
            lemmatized_tokens.append(token)
        else:
            # Her bir token'ı analiz et ve lemmatize et
            analysis = morphology.analyzeSentence(token)
            best_analysis = morphology.disambiguate(token, analysis).bestAnalysis()
            if best_analysis.size() > 0:
                lemma = str(best_analysis.get(0).getLemmas()[0])  # Lemmatize edilen hali al
                if lemma.lower() == 'unk':  # Eğer lemmatize edilen hali 'UNK' ise orijinal token'ı ekle
                    lemmatized_tokens.append(token)
                else:
                    lemmatized_tokens.append(lemma)
            else:
                lemmatized_tokens.append(token)  # Lemmatize edilemeyeni olduğu gibi ekle
    return ' '.join(lemmatized_tokens)

# STK ile ilgili sütunları temizleyen fonksiyon
def stk_temizleme(cumle):
    if pd.isna(cumle):
        return ''  # Eğer boş ise boş bir string döndür
    
    # Metni ön işlemden geçir
    cumle = str(cumle).lower()  # Küçük harfe çevir
    cumle = ' '.join([kelime for kelime in cumle.split() if kelime not in stop_words and not kelime.isdigit()])  # Stopwords ve sayıları kaldır
    cumle = cumle.translate(str.maketrans('', '', string.punctuation))  # Noktalama işaretlerini kaldır
    cumle = re.sub(r'[^\w\s]', ' ', cumle)  # Özel karakterleri kaldır
    cumle = cumle.replace('\n', ' ')
    cumle = re.sub(r'\s+', ' ', cumle).strip()  # Fazla boşlukları temizle
    
    # Lemmatizasyon uygula
    cumle = lemmatize_text(cumle)
    return cumle

# Uygulama: STK sütunlarını temizleyin
df["Hangi STK'nin Uyesisiniz?"] = df["Hangi STK'nin Uyesisiniz?"].apply(stk_temizleme)
df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"] = df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"].apply(stk_temizleme)

# Belirli sütunlarda '-' işaretlerini NaN ile değiştir
columns_to_fill = ['Burs Aldigi Baska Kurum', 'Baska Kurumdan Aldigi Burs Miktari', 'Anne Sektor', 'Baba Sektor', 'Spor Dalindaki Rolunuz Nedir?', 'Universite Not Ortalamasi']
df[columns_to_fill] = df[columns_to_fill].replace('-', np.nan)

# Sonuçları yeni bir CSV dosyasına kaydedin
df.to_csv(output_file_path, index=False)
