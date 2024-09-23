# DATATHON_2024
BTK Akademi, Google ve Girişimcilik Vakfı'nın düzenlediği Datathon 2024 yarışması dataBenders ekibinin proje kodları

Takım Adı: dataBenders
Üyeleri:
1)Beyza AYDOĞMUŞ - İzmir Katip Çelebi Üniversitesi Bilgisayar Mühendisliği 3.sınıf öğrencisi
2)Sena YÜKSEL - İzmir Katip Çelebi Üniversitesi Bilgisayar Mühendisliği 3.sınıf öğrencisi

Kodların Kullanılabilmesi İçin Uygulanması Gereken Aşamalar
-----------------------------------------------------------
Yüklenmesi Gerekenler:

```bash
pip install pandas
```
```bash
pip install numpy
```
```bash
pip install scikit-learn
```
```bash
pip install lightgbm
```
```bash
pip install nltk
```
```bash
pip install jpype1
```
Son olarak, Zemberek kütüphanesini indirin ve proje dizinine yerleştirin.
[Zemberek NLP GitHub Sayfası](https://github.com/ahmetaa/zemberek-nlp)

Bu linke girip "Jar Dağıtımları" kısmından Google Drive sayfasına tıklamalısınız. Burada farklı versiyonlar için jar dosyaları bulunmaktadır. Bu dosyaların içinden "distributions" dosyasını açmalı ve son sürüm zemberek-full.jar adlı dosyayı yüklemelisiniz.

Dosyaların Çalıştırılma Sırası:

1) temizleme.py dosyası ilk olarak kullanılmalıdır. Bu kod Girişimcilik Vafı'nın ilettiği train dosyasının temizlenmesi için kullanılmaktadır.
2) Bir diğer adımda test_temizleme.py dosyası çalıştırılacaktır. Bu kodda Girişimcilik Vafı'nın ilettiği test dosyası temizlenmektedir.
3) csvDosyasından_excelDosyasına_cevirme.py bir sonraki aşamada çalıştırılacaktır. Verilerin model eğitiminde kullanılabilmesi için excel dosyasına dönüştürmek için yazılmıştır.
4) training model.py kodu son aşamada çalıştırılmalıdır. Yarışma sürecinde eğitim kodunu Colab ortamında çalıştırdık. Eğitimde LightGBM framework'ü kullanılmıştır.

** Ek olarak, farklı_degerler.py dosyası ise test_temizleme.py ve temizleme.py dosyalarında özellik çıkarımı sırasında iyileştirmeye gidilmesi için kullanılmıştır.
