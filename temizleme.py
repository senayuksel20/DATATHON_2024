import numpy as np
import pandas as pd
import re
import string
import jpype
from nltk.corpus import stopwords
from jpype import JClass, startJVM, shutdownJVM, getDefaultJVMPath

# CSV dosyasını yükleme
train=r"train.csv" #train.csv dosyasını dosya yolunu buraya ekleyiniz
df = pd.read_csv(train)


# Türkiye'nin 81 ili (küçük harflerle)
iller = [
    'adana', 'adiyaman', 'afyonkarahisar', 'agri', 'amasya', 'ankara', 'antalya', 'artvin', 'aydin', 'balikesir', 
    'bilecik', 'bingol', 'bitlis', 'bolu', 'burdur', 'bursa', 'canakkale', 'cankiri', 'corum', 'denizli', 
    'diyarbakir', 'edirne', 'elazig', 'erzincan', 'erzurum', 'eskisehir', 'gaziantep', 'giresun', 'gumushane', 'hakkari', 
    'hatay', 'igdir', 'isparta', 'istanbul', 'izmir', 'kahramanmaras', 'karabuk', 'karaman', 'kars', 'kastamonu', 'kayseri', 
    'kirikkale', 'kirklareli', 'kirsehir', 'kilis', 'kocaeli', 'konya', 'kutahya', 'malatya', 'manisa', 'mardin', 
    'mersin', 'mugla', 'mus', 'nevsehir', 'nigde', 'ordu', 'osmaniye', 'rize', 'sakarya', 'samsun', 
    'siirt', 'sinop', 'sivas', 'sanliurfa', 'sirnak', 'tekirdag', 'tokat', 'trabzon', 'tunceli', 'usak', 
    'van', 'yalova', 'yozgat', 'zonguldak', 'aksaray', 'bayburt', 'karaman', 'kirikkale', 'batman', 'sirnak', 
    'bartin', 'ardahan', 'igdir', 'yalova', 'karabuk', 'kilis', 'osmaniye', 'duzce'
]

# İlçeler ve iller eşleştirmesi (örnek olarak sadece birkaç ilçe ve ili ekledim, daha fazlasını ekleyebilirsiniz)
ilceler = {
    'aladag': 'adana',
    'ceyhan': 'adana',
    'cukurova': 'adana',
    'feke': 'adana',
    'imamoglu': 'adana',
    'karaisali': 'adana',
    'karatas': 'adana',
    'kozan': 'adana',
    'pozanti': 'adana',
    'saimbeyli': 'adana',
    'saricam': 'adana',
    'seyhan': 'adana',
    'tufanbeyli': 'adana',
    'yumurtalik': 'adana',
    'yuregir': 'adana',
    'besni': 'adiyaman',
    'celikhan': 'adiyaman',
    'gerger': 'adiyaman',
    'golbasi': 'ankara',
    'kahta': 'adiyaman',
    'merkez': 'zonguldak',
    'samsat': 'adiyaman',
    'sincik': 'adiyaman',
    'tut': 'adiyaman',
    'basmakci': 'afyonkarahisar',
    'bayat': 'corum',
    'bolvadin': 'afyonkarahisar',
    'cay': 'afyonkarahisar',
    'cobanlar': 'afyonkarahisar',
    'dazkiri': 'afyonkarahisar',
    'dinar': 'afyonkarahisar',
    'emirdag': 'afyonkarahisar',
    'evciler': 'afyonkarahisar',
    'hocalar': 'afyonkarahisar',
    'ihsaniye': 'afyonkarahisar',
    'iscehisar': 'afyonkarahisar',
    'kiziloren': 'afyonkarahisar',
    'sandikli': 'afyonkarahisar',
    'sinanpasa': 'afyonkarahisar',
    'suhut': 'afyonkarahisar',
    'sultandagi': 'afyonkarahisar',
    'diyadin': 'agri',
    'dogubayazit': 'agri',
    'eleskirt': 'agri',
    'hamur': 'agri',
    'patnos': 'agri',
    'taslicay': 'agri',
    'tutak': 'agri',
    'agacoren': 'aksaray',
    'eskil': 'aksaray',
    'gulagac': 'aksaray',
    'guzelyurt': 'aksaray',
    'ortakoy': 'corum',
    'sariyahsi': 'aksaray',
    'sultanhani': 'aksaray',
    'goynucek': 'amasya',
    'gumushacikoy': 'amasya',
    'hamamozu': 'amasya',
    'merzifon': 'amasya',
    'suluova': 'amasya',
    'tasova': 'amasya',
    'akyurt': 'ankara',
    'altindag': 'ankara',
    'ayas': 'ankara',
    'bala': 'ankara',
    'beypazari': 'ankara',
    'camlidere': 'ankara',
    'cankaya': 'ankara',
    'cubuk': 'ankara',
    'elmadag': 'ankara',
    'etimesgut': 'ankara',
    'evren': 'ankara',
    'gudul': 'ankara',
    'haymana': 'ankara',
    'kahramankazan': 'ankara',
    'kalecik': 'ankara',
    'kecioren': 'ankara',
    'kizilcahamam': 'ankara',
    'mamak': 'ankara',
    'nallihan': 'ankara',
    'polatli': 'ankara',
    'pursaklar': 'ankara',
    'sereflikochisar': 'ankara',
    'sincan': 'ankara',
    'yenimahalle': 'ankara',
    'akseki': 'antalya',
    'aksu': 'isparta',
    'alanya': 'antalya',
    'demre': 'antalya',
    'dosemealti': 'antalya',
    'elmali': 'antalya',
    'finike': 'antalya',
    'gazipasa': 'antalya',
    'gundogmus': 'antalya',
    'ibradi': 'antalya',
    'kas': 'antalya',
    'kemer': 'burdur',
    'kepez': 'antalya',
    'konyaalti': 'antalya',
    'korkuteli': 'antalya',
    'kumluca': 'antalya',
    'manavgat': 'antalya',
    'muratpasa': 'antalya',
    'serik': 'antalya',
    'cildir': 'ardahan',
    'damal': 'ardahan',
    'gole': 'ardahan',
    'hanak': 'ardahan',
    'posof': 'ardahan',
    'ardanuc': 'artvin',
    'arhavi': 'artvin',
    'borcka': 'artvin',
    'hopa': 'artvin',
    'kemalpasa': 'izmir',
    'murgul': 'artvin',
    'savsat': 'artvin',
    'yusufeli': 'artvin',
    'bozdogan': 'aydin',
    'buharkent': 'aydin',
    'cine': 'aydin',
    'didim': 'aydin',
    'efeler': 'aydin',
    'germencik': 'aydin',
    'incirliova': 'aydin',
    'karacasu': 'aydin',
    'karpuzlu': 'aydin',
    'kocarli': 'aydin',
    'kosk': 'aydin',
    'kusadasi': 'aydin',
    'kuyucak': 'aydin',
    'nazilli': 'aydin',
    'soke': 'aydin',
    'sultanhisar': 'aydin',
    'yenipazar': 'bilecik',
    'altieylul': 'balikesir',
    'ayvalik': 'balikesir',
    'balya': 'balikesir',
    'bandirma': 'balikesir',
    'bigadic': 'balikesir',
    'burhaniye': 'balikesir',
    'dursunbey': 'balikesir',
    'edremit': 'van',
    'erdek': 'balikesir',
    'gomec': 'balikesir',
    'gonen': 'isparta',
    'havran': 'balikesir',
    'ivrindi': 'balikesir',
    'karesi': 'balikesir',
    'kepsut': 'balikesir',
    'manyas': 'balikesir',
    'marmara': 'balikesir',
    'savastepe': 'balikesir',
    'sindirgi': 'balikesir',
    'susurluk': 'balikesir',
    'amasra': 'bartin',
    'kurucasile': 'bartin',
    'ulus': 'bartin',
    'besiri': 'batman',
    'gercus': 'batman',
    'hasankeyf': 'batman',
    'kozluk': 'batman',
    'sason': 'batman',
    'aydintepe': 'bayburt',
    'demirozu': 'bayburt',
    'bozuyuk': 'bilecik',
    'golpazari': 'bilecik',
    'inhisar': 'bilecik',
    'osmaneli': 'bilecik',
    'pazaryeri': 'bilecik',
    'sogut': 'bilecik',
    'adakli': 'bingol',
    'genc': 'bingol',
    'karliova': 'bingol',
    'kigi': 'bingol',
    'solhan': 'bingol',
    'yayladere': 'bingol',
    'yedisu': 'bingol',
    'adilcevaz': 'bitlis',
    'ahlat': 'bitlis',
    'guroymak': 'bitlis',
    'hizan': 'bitlis',
    'mutki': 'bitlis',
    'tatvan': 'bitlis',
    'dortdivan': 'bolu',
    'gerede': 'bolu',
    'goynuk': 'bolu',
    'kibriscik': 'bolu',
    'mengen': 'bolu',
    'mudurnu': 'bolu',
    'seben': 'bolu',
    'yenicaga': 'bolu',
    'aglasun': 'burdur',
    'altinyayla': 'sivas',
    'bucak': 'burdur',
    'cavdir': 'burdur',
    'celtikci': 'burdur',
    'golhisar': 'burdur',
    'karamanli': 'burdur',
    'tefenni': 'burdur',
    'yesilova': 'burdur',
    'buyukorhan': 'bursa',
    'gemlik': 'bursa',
    'gursu': 'bursa',
    'harmancik': 'bursa',
    'inegol': 'bursa',
    'iznik': 'bursa',
    'karacabey': 'bursa',
    'keles': 'bursa',
    'kestel': 'bursa',
    'mudanya': 'bursa',
    'mustafakemalpasa': 'bursa',
    'nilufer': 'bursa',
    'orhaneli': 'bursa',
    'orhangazi': 'bursa',
    'osmangazi': 'bursa',
    'yenisehir': 'mersin',
    'yildirim': 'bursa',
    'ayvacik': 'samsun',
    'bayramic': 'canakkale',
    'biga': 'canakkale',
    'bozcaada': 'canakkale',
    'can': 'canakkale',
    'eceabat': 'canakkale',
    'ezine': 'canakkale',
    'gelibolu': 'canakkale',
    'gokceada': 'canakkale',
    'lapseki': 'canakkale',
    'yenice': 'karabuk',
    'atkaracalar': 'cankiri',
    'bayramoren': 'cankiri',
    'cerkes': 'cankiri',
    'eldivan': 'cankiri',
    'ilgaz': 'cankiri',
    'kizilirmak': 'cankiri',
    'korgun': 'cankiri',
    'kursunlu': 'cankiri',
    'orta': 'cankiri',
    'sabanozu': 'cankiri',
    'yaprakli': 'cankiri',
    'alaca': 'corum',
    'bogazkale': 'corum',
    'dodurga': 'corum',
    'iskilip': 'corum',
    'kargi': 'corum',
    'lacin': 'corum',
    'mecitozu': 'corum',
    'oguzlar': 'corum',
    'osmancik': 'corum',
    'sungurlu': 'corum',
    'ugurludag': 'corum',
    'acipayam': 'denizli',
    'babadag': 'denizli',
    'baklan': 'denizli',
    'bekilli': 'denizli',
    'beyagac': 'denizli',
    'bozkurt': 'kastamonu',
    'buldan': 'denizli',
    'cal': 'denizli',
    'cameli': 'denizli',
    'cardak': 'denizli',
    'civril': 'denizli',
    'guney': 'denizli',
    'honaz': 'denizli',
    'kale': 'malatya',
    'merkezefendi': 'denizli',
    'pamukkale': 'denizli',
    'saraykoy': 'denizli',
    'serinhisar': 'denizli',
    'tavas': 'denizli',
    'baglar': 'diyarbakir',
    'bismil': 'diyarbakir',
    'cermik': 'diyarbakir',
    'cinar': 'diyarbakir',
    'cungus': 'diyarbakir',
    'dicle': 'diyarbakir',
    'egil': 'diyarbakir',
    'ergani': 'diyarbakir',
    'hani': 'diyarbakir',
    'hazro': 'diyarbakir',
    'kayapinar': 'diyarbakir',
    'kocakoy': 'diyarbakir',
    'kulp': 'diyarbakir',
    'lice': 'diyarbakir',
    'silvan': 'diyarbakir',
    'sur': 'diyarbakir',
    'akcakoca': 'duzce',
    'cilimli': 'duzce',
    'cumayeri': 'duzce',
    'golyaka': 'duzce',
    'gumusova': 'duzce',
    'kaynasli': 'duzce',
    'yigilca': 'duzce',
    'enez': 'edirne',
    'havsa': 'edirne',
    'ipsala': 'edirne',
    'kesan': 'edirne',
    'lalapasa': 'edirne',
    'meric': 'edirne',
    'suloglu': 'edirne',
    'uzunkopru': 'edirne',
    'agin': 'elazig',
    'alacakaya': 'elazig',
    'aricak': 'elazig',
    'baskil': 'elazig',
    'karakocan': 'elazig',
    'keban': 'elazig',
    'kovancilar': 'elazig',
    'maden': 'elazig',
    'palu': 'elazig',
    'sivrice': 'elazig',
    'cayirli': 'erzincan',
    'ilic': 'erzincan',
    'kemah': 'erzincan',
    'kemaliye': 'erzincan',
    'otlukbeli': 'erzincan',
    'refahiye': 'erzincan',
    'tercan': 'erzincan',
    'uzumlu': 'erzincan',
    'askale': 'erzurum',
    'aziziye': 'erzurum',
    'cat': 'erzurum',
    'hinis': 'erzurum',
    'horasan': 'erzurum',
    'ispir': 'erzurum',
    'karacoban': 'erzurum',
    'karayazi': 'erzurum',
    'koprukoy': 'erzurum',
    'narman': 'erzurum',
    'oltu': 'erzurum',
    'olur': 'erzurum',
    'palandoken': 'erzurum',
    'pasinler': 'erzurum',
    'pazaryolu': 'erzurum',
    'senkaya': 'erzurum',
    'tekman': 'erzurum',
    'tortum': 'erzurum',
    'uzundere': 'erzurum',
    'yakutiye': 'erzurum',
    'alpu': 'eskisehir',
    'beylikova': 'eskisehir',
    'cifteler': 'eskisehir',
    'gunyuzu': 'eskisehir',
    'han': 'eskisehir',
    'inonu': 'eskisehir',
    'mahmudiye': 'eskisehir',
    'mihalgazi': 'eskisehir',
    'mihaliccik': 'eskisehir',
    'odunpazari': 'eskisehir',
    'saricakaya': 'eskisehir',
    'seyitgazi': 'eskisehir',
    'sivrihisar': 'eskisehir',
    'tepebasi': 'eskisehir',
    'araban': 'gaziantep',
    'islahiye': 'gaziantep',
    'karkamis': 'gaziantep',
    'nizip': 'gaziantep',
    'nurdagi': 'gaziantep',
    'oguzeli': 'gaziantep',
    'sahinbey': 'gaziantep',
    'sehitkamil': 'gaziantep',
    'yavuzeli': 'gaziantep',
    'alucra': 'giresun',
    'bulancak': 'giresun',
    'camoluk': 'giresun',
    'canakci': 'giresun',
    'dereli': 'giresun',
    'dogankent': 'giresun',
    'espiye': 'giresun',
    'eynesil': 'giresun',
    'gorele': 'giresun',
    'guce': 'giresun',
    'kesap': 'giresun',
    'piraziz': 'giresun',
    'sebinkarahisar': 'giresun',
    'tirebolu': 'giresun',
    'yaglidere': 'giresun',
    'kelkit': 'gumushane',
    'kose': 'gumushane',
    'kurtun': 'gumushane',
    'siran': 'gumushane',
    'torul': 'gumushane',
    'cukurca': 'hakkari',
    'derecik': 'hakkari',
    'semdinli': 'hakkari',
    'yuksekova': 'hakkari',
    'altinozu': 'hatay',
    'antakya': 'hatay',
    'arsuz': 'hatay',
    'belen': 'hatay',
    'defne': 'hatay',
    'dortyol': 'hatay',
    'erzin': 'hatay',
    'hassa': 'hatay',
    'iskenderun': 'hatay',
    'kirikhan': 'hatay',
    'kumlu': 'hatay',
    'payas': 'hatay',
    'reyhanli': 'hatay',
    'samandag': 'hatay',
    'yayladagi': 'hatay',
    'aralik': 'igdir',
    'karakoyunlu': 'igdir',
    'tuzluca': 'igdir',
    'atabey': 'isparta',
    'egirdir': 'isparta',
    'gelendost': 'isparta',
    'keciborlu': 'isparta',
    'sarkikaraagac': 'isparta',
    'senirkent': 'isparta',
    'sutculer': 'isparta',
    'uluborlu': 'isparta',
    'yalvac': 'isparta',
    'yenisarbademli': 'isparta',
    'adalar': 'istanbul',
    'arnavutkoy': 'istanbul',
    'atasehir': 'istanbul',
    'avcilar': 'istanbul',
    'bagcilar': 'istanbul',
    'bahcelievler': 'istanbul',
    'bakirkoy': 'istanbul',
    'basaksehir': 'istanbul',
    'bayrampasa': 'istanbul',
    'besiktas': 'istanbul',
    'beykoz': 'istanbul',
    'beylikduzu': 'istanbul',
    'beyoglu': 'istanbul',
    'buyukcekmece': 'istanbul',
    'catalca': 'istanbul',
    'cekmekoy': 'istanbul',
    'esenler': 'istanbul',
    'esenyurt': 'istanbul',
    'eyupsultan': 'istanbul',
    'fatih': 'istanbul',
    'gaziosmanpasa': 'istanbul',
    'gungoren': 'istanbul',
    'kadikoy': 'istanbul',
    'kagithane': 'istanbul',
    'kartal': 'istanbul',
    'kucukcekmece': 'istanbul',
    'maltepe': 'istanbul',
    'pendik': 'istanbul',
    'sancaktepe': 'istanbul',
    'sariyer': 'istanbul',
    'sile': 'istanbul',
    'silivri': 'istanbul',
    'sisli': 'istanbul',
    'sultanbeyli': 'istanbul',
    'sultangazi': 'istanbul',
    'tuzla': 'istanbul',
    'umraniye': 'istanbul',
    'uskudar': 'istanbul',
    'zeytinburnu': 'istanbul',
    'aliaga': 'izmir',
    'balcova': 'izmir',
    'bayindir': 'izmir',
    'bayrakli': 'izmir',
    'bergama': 'izmir',
    'beydag': 'izmir',
    'bornova': 'izmir',
    'buca': 'izmir',
    'cesme': 'izmir',
    'cigli': 'izmir',
    'dikili': 'izmir',
    'foca': 'izmir',
    'gaziemir': 'izmir',
    'guzelbahce': 'izmir',
    'karabaglar': 'izmir',
    'karaburun': 'izmir',
    'karsiyaka': 'izmir',
    'kinik': 'izmir',
    'kiraz': 'izmir',
    'konak': 'izmir',
    'menderes': 'izmir',
    'menemen': 'izmir',
    'narlidere': 'izmir',
    'odemis': 'izmir',
    'seferihisar': 'izmir',
    'selcuk': 'izmir',
    'tire': 'izmir',
    'torbali': 'izmir',
    'urla': 'izmir',
    'afsin': 'kahramanmaras',
    'andirin': 'kahramanmaras',
    'caglayancerit': 'kahramanmaras',
    'dulkadiroglu': 'kahramanmaras',
    'ekinozu': 'kahramanmaras',
    'elbistan': 'kahramanmaras',
    'goksun': 'kahramanmaras',
    'nurhak': 'kahramanmaras',
    'onikisubat': 'kahramanmaras',
    'pazarcik': 'kahramanmaras',
    'turkoglu': 'kahramanmaras',
    'eflani': 'karabuk',
    'eskipazar': 'karabuk',
    'ovacik': 'tunceli',
    'safranbolu': 'karabuk',
    'ayranci': 'karaman',
    'basyayla': 'karaman',
    'ermenek': 'karaman',
    'kazimkarabekir': 'karaman',
    'sariveliler': 'karaman',
    'akyaka': 'kars',
    'arpacay': 'kars',
    'digor': 'kars',
    'kagizman': 'kars',
    'sarikamis': 'kars',
    'selim': 'kars',
    'susuz': 'kars',
    'abana': 'kastamonu',
    'agli': 'kastamonu',
    'arac': 'kastamonu',
    'azdavay': 'kastamonu',
    'catalzeytin': 'kastamonu',
    'cide': 'kastamonu',
    'daday': 'kastamonu',
    'devrekani': 'kastamonu',
    'doganyurt': 'kastamonu',
    'hanonu': 'kastamonu',
    'ihsangazi': 'kastamonu',
    'inebolu': 'kastamonu',
    'kure': 'kastamonu',
    'pinarbasi': 'kayseri',
    'senpazar': 'kastamonu',
    'seydiler': 'kastamonu',
    'taskopru': 'kastamonu',
    'tosya': 'kastamonu',
    'akkisla': 'kayseri',
    'bunyan': 'kayseri',
    'develi': 'kayseri',
    'felahiye': 'kayseri',
    'hacilar': 'kayseri',
    'incesu': 'kayseri',
    'kocasinan': 'kayseri',
    'melikgazi': 'kayseri',
    'ozvatan': 'kayseri',
    'sarioglan': 'kayseri',
    'sariz': 'kayseri',
    'talas': 'kayseri',
    'tomarza': 'kayseri',
    'yahyali': 'kayseri',
    'yesilhisar': 'kayseri',
    'elbeyli': 'kilis',
    'musabeyli': 'kilis',
    'polateli': 'kilis',
    'bahsili': 'kirikkale',
    'baliseyh': 'kirikkale',
    'celebi': 'kirikkale',
    'delice': 'kirikkale',
    'karakecili': 'kirikkale',
    'keskin': 'kirikkale',
    'sulakyurt': 'kirikkale',
    'yahsihan': 'kirikkale',
    'babaeski': 'kirklareli',
    'demirkoy': 'kirklareli',
    'kofcaz': 'kirklareli',
    'luleburgaz': 'kirklareli',
    'pehlivankoy': 'kirklareli',
    'pinarhisar': 'kirklareli',
    'vize': 'kirklareli',
    'akcakent': 'kirsehir',
    'akpinar': 'kirsehir',
    'boztepe': 'kirsehir',
    'cicekdagi': 'kirsehir',
    'kaman': 'kirsehir',
    'mucur': 'kirsehir',
    'basiskele': 'kocaeli',
    'cayirova': 'kocaeli',
    'darica': 'kocaeli',
    'derince': 'kocaeli',
    'dilovasi': 'kocaeli',
    'gebze': 'kocaeli',
    'golcuk': 'kocaeli',
    'izmit': 'kocaeli',
    'kandira': 'kocaeli',
    'karamursel': 'kocaeli',
    'kartepe': 'kocaeli',
    'korfez': 'kocaeli',
    'ahirli': 'konya',
    'akoren': 'konya',
    'aksehir': 'konya',
    'altinekin': 'konya',
    'beysehir': 'konya',
    'bozkir': 'konya',
    'celtik': 'konya',
    'cihanbeyli': 'konya',
    'cumra': 'konya',
    'derbent': 'konya',
    'derebucak': 'konya',
    'doganhisar': 'konya',
    'emirgazi': 'konya',
    'eregli': 'zonguldak',
    'guneysinir': 'konya',
    'hadim': 'konya',
    'halkapinar': 'konya',
    'huyuk': 'konya',
    'ilgin': 'konya',
    'kadinhani': 'konya',
    'karapinar': 'konya',
    'karatay': 'konya',
    'kulu': 'konya',
    'meram': 'konya',
    'sarayonu': 'konya',
    'selcuklu': 'konya',
    'seydisehir': 'konya',
    'taskent': 'konya',
    'tuzlukcu': 'konya',
    'yalihuyuk': 'konya',
    'yunak': 'konya',
    'altintas': 'kutahya',
    'aslanapa': 'kutahya',
    'cavdarhisar': 'kutahya',
    'domanic': 'kutahya',
    'dumlupinar': 'kutahya',
    'emet': 'kutahya',
    'gediz': 'kutahya',
    'hisarcik': 'kutahya',
    'pazarlar': 'kutahya',
    'saphane': 'kutahya',
    'simav': 'kutahya',
    'tavsanli': 'kutahya',
    'akcadag': 'malatya',
    'arapgir': 'malatya',
    'arguvan': 'malatya',
    'battalgazi': 'malatya',
    'darende': 'malatya',
    'dogansehir': 'malatya',
    'doganyol': 'malatya',
    'hekimhan': 'malatya',
    'kuluncak': 'malatya',
    'puturge': 'malatya',
    'yazihan': 'malatya',
    'yesilyurt': 'tokat',
    'ahmetli': 'manisa',
    'akhisar': 'manisa',
    'alasehir': 'manisa',
    'demirci': 'manisa',
    'golmarmara': 'manisa',
    'gordes': 'manisa',
    'kirkagac': 'manisa',
    'koprubasi': 'trabzon',
    'kula': 'manisa',
    'salihli': 'manisa',
    'sarigol': 'manisa',
    'saruhanli': 'manisa',
    'sehzadeler': 'manisa',
    'selendi': 'manisa',
    'soma': 'manisa',
    'turgutlu': 'manisa',
    'yunusemre': 'manisa',
    'artuklu': 'mardin',
    'dargecit': 'mardin',
    'derik': 'mardin',
    'kiziltepe': 'mardin',
    'mazidagi': 'mardin',
    'midyat': 'mardin',
    'nusaybin': 'mardin',
    'omerli': 'mardin',
    'savur': 'mardin',
    'yesilli': 'mardin',
    'akdeniz': 'mersin',
    'anamur': 'mersin',
    'aydincik': 'yozgat',
    'bozyazi': 'mersin',
    'camliyayla': 'mersin',
    'erdemli': 'mersin',
    'gulnar': 'mersin',
    'mezitli': 'mersin',
    'mut': 'mersin',
    'silifke': 'mersin',
    'tarsus': 'mersin',
    'toroslar': 'mersin',
    'bodrum': 'mugla',
    'dalaman': 'mugla',
    'datca': 'mugla',
    'fethiye': 'mugla',
    'kavaklidere': 'mugla',
    'koycegiz': 'mugla',
    'marmaris': 'mugla',
    'mentese': 'mugla',
    'milas': 'mugla',
    'ortaca': 'mugla',
    'seydikemer': 'mugla',
    'ula': 'mugla',
    'yatagan': 'mugla',
    'bulanik': 'mus',
    'haskoy': 'mus',
    'korkut': 'mus',
    'malazgirt': 'mus',
    'varto': 'mus',
    'acigol': 'nevsehir',
    'avanos': 'nevsehir',
    'derinkuyu': 'nevsehir',
    'gulsehir': 'nevsehir',
    'hacibektas': 'nevsehir',
    'kozakli': 'nevsehir',
    'urgup': 'nevsehir',
    'altunhisar': 'nigde',
    'bor': 'nigde',
    'camardi': 'nigde',
    'ciftlik': 'nigde',
    'ulukisla': 'nigde',
    'akkus': 'ordu',
    'altinordu': 'ordu',
    'aybasti': 'ordu',
    'camas': 'ordu',
    'catalpinar': 'ordu',
    'caybasi': 'ordu',
    'fatsa': 'ordu',
    'golkoy': 'ordu',
    'gulyali': 'ordu',
    'gurgentepe': 'ordu',
    'ikizce': 'ordu',
    'kabaduz': 'ordu',
    'kabatas': 'ordu',
    'korgan': 'ordu',
    'kumru': 'ordu',
    'mesudiye': 'ordu',
    'persembe': 'ordu',
    'ulubey': 'usak',
    'unye': 'ordu',
    'bahce': 'osmaniye',
    'duzici': 'osmaniye',
    'hasanbeyli': 'osmaniye',
    'kadirli': 'osmaniye',
    'sumbas': 'osmaniye',
    'toprakkale': 'osmaniye',
    'ardesen': 'rize',
    'camlihemsin': 'rize',
    'cayeli': 'rize',
    'derepazari': 'rize',
    'findikli': 'rize',
    'guneysu': 'rize',
    'hemsin': 'rize',
    'ikizdere': 'rize',
    'iyidere': 'rize',
    'kalkandere': 'rize',
    'pazar': 'tokat',
    'adapazari': 'sakarya',
    'akyazi': 'sakarya',
    'arifiye': 'sakarya',
    'erenler': 'sakarya',
    'ferizli': 'sakarya',
    'geyve': 'sakarya',
    'hendek': 'sakarya',
    'karapurcek': 'sakarya',
    'karasu': 'sakarya',
    'kaynarca': 'sakarya',
    'kocaali': 'sakarya',
    'pamukova': 'sakarya',
    'sapanca': 'sakarya',
    'serdivan': 'sakarya',
    'sogutlu': 'sakarya',
    'tarakli': 'sakarya',
    '19 mayis': 'samsun',
    'alacam': 'samsun',
    'asarcik': 'samsun',
    'atakum': 'samsun',
    'bafra': 'samsun',
    'canik': 'samsun',
    'carsamba': 'samsun',
    'havza': 'samsun',
    'ilkadim': 'samsun',
    'kavak': 'samsun',
    'ladik': 'samsun',
    'salipazari': 'samsun',
    'tekkekoy': 'samsun',
    'terme': 'samsun',
    'vezirkopru': 'samsun',
    'yakakent': 'samsun',
    'akcakale': 'sanliurfa',
    'birecik': 'sanliurfa',
    'bozova': 'sanliurfa',
    'ceylanpinar': 'sanliurfa',
    'eyyubiye': 'sanliurfa',
    'halfeti': 'sanliurfa',
    'haliliye': 'sanliurfa',
    'harran': 'sanliurfa',
    'hilvan': 'sanliurfa',
    'karakopru': 'sanliurfa',
    'siverek': 'sanliurfa',
    'suruc': 'sanliurfa',
    'viransehir': 'sanliurfa',
    'baykan': 'siirt',
    'eruh': 'siirt',
    'kurtalan': 'siirt',
    'pervari': 'siirt',
    'sirvan': 'siirt',
    'tillo': 'siirt',
    'ayancik': 'sinop',
    'boyabat': 'sinop',
    'dikmen': 'sinop',
    'duragan': 'sinop',
    'erfelek': 'sinop',
    'gerze': 'sinop',
    'sarayduzu': 'sinop',
    'turkeli': 'sinop',
    'beytussebap': 'sirnak',
    'cizre': 'sirnak',
    'guclukonak': 'sirnak',
    'idil': 'sirnak',
    'silopi': 'sirnak',
    'uludere': 'sirnak',
    'akincilar': 'sivas',
    'divrigi': 'sivas',
    'dogansar': 'sivas',
    'gemerek': 'sivas',
    'golova': 'sivas',
    'gurun': 'sivas',
    'hafik': 'sivas',
    'imranli': 'sivas',
    'kangal': 'sivas',
    'koyulhisar': 'sivas',
    'sarkisla': 'sivas',
    'susehri': 'sivas',
    'ulas': 'sivas',
    'yildizeli': 'sivas',
    'zara': 'sivas',
    'cerkezkoy': 'tekirdag',
    'corlu': 'tekirdag',
    'ergene': 'tekirdag',
    'hayrabolu': 'tekirdag',
    'kapakli': 'tekirdag',
    'malkara': 'tekirdag',
    'marmaraereglisi': 'tekirdag',
    'muratli': 'tekirdag',
    'saray': 'van',
    'sarkoy': 'tekirdag',
    'suleymanpasa': 'tekirdag',
    'almus': 'tokat',
    'artova': 'tokat',
    'basciftlik': 'tokat',
    'erbaa': 'tokat',
    'niksar': 'tokat',
    'resadiye': 'tokat',
    'sulusaray': 'tokat',
    'turhal': 'tokat',
    'zile': 'tokat',
    'akcaabat': 'trabzon',
    'arakli': 'trabzon',
    'arsin': 'trabzon',
    'besikduzu': 'trabzon',
    'carsibasi': 'trabzon',
    'caykara': 'trabzon',
    'dernekpazari': 'trabzon',
    'duzkoy': 'trabzon',
    'hayrat': 'trabzon',
    'macka': 'trabzon',
    'of': 'trabzon',
    'ortahisar': 'trabzon',
    'salpazari': 'trabzon',
    'surmene': 'trabzon',
    'tonya': 'trabzon',
    'vakfikebir': 'trabzon',
    'yomra': 'trabzon',
    'cemisgezek': 'tunceli',
    'hozat': 'tunceli',
    'mazgirt': 'tunceli',
    'nazimiye': 'tunceli',
    'pertek': 'tunceli',
    'pulumur': 'tunceli',
    'banaz': 'usak',
    'esme': 'usak',
    'karahalli': 'usak',
    'sivasli': 'usak',
    'bahcesaray': 'van',
    'baskale': 'van',
    'caldiran': 'van',
    'catak': 'van',
    'ercis': 'van',
    'gevas': 'van',
    'gurpinar': 'van',
    'ipekyolu': 'van',
    'muradiye': 'van',
    'ozalp': 'van',
    'tusba': 'van',
    'altinova': 'yalova',
    'armutlu': 'yalova',
    'ciftlikkoy': 'yalova',
    'cinarcik': 'yalova',
    'termal': 'yalova',
    'akdagmadeni': 'yozgat',
    'bogazliyan': 'yozgat',
    'candir': 'yozgat',
    'cayiralan': 'yozgat',
    'cekerek': 'yozgat',
    'kadisehri': 'yozgat',
    'saraykent': 'yozgat',
    'sarikaya': 'yozgat',
    'sefaatli': 'yozgat',
    'sorgun': 'yozgat',
    'yenifakili': 'yozgat',
    'yerkoy': 'yozgat',
    'alapli': 'zonguldak',
    'caycuma': 'zonguldak',
    'devrek': 'zonguldak',
    'gokcebey': 'zonguldak',
    'kilimli': 'zonguldak',
    'kozlu': 'zonguldak',
    'merkez' : 'zonguldak'
}


# Dönüştürme işlemi yapılacak sütunlar
columns_to_transform = [
    "Cinsiyet",
    "Dogum Tarihi",
    "Dogum Yeri",
    "Ikametgah Sehri",
    "Universite Adi",
    "Universite Turu",
    "Burs Aliyor mu?", 
    "Bölüm",
    "Daha Once Baska Bir Universiteden Mezun Olmus", 
    "Lise Adi",
    "Lise Adi Diger",
    "Lise Sehir",
    "Lise Turu",
    "Lise Bolumu",
    "Lise Bolum Diger",
    "Baska Bir Kurumdan Burs Aliyor mu?", 
    "Burs Aldigi Baska Kurum",
    "Anne Egitim Durumu",
    "Anne Calisma Durumu",
    "Anne Sektor",
    "Baba Egitim Durumu",
    "Baba Calisma Durumu",
    "Baba Sektor",
    "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?", 
    "Uye Oldugunuz Kulubun Ismi",
    "Profesyonel Bir Spor Daliyla Mesgul musunuz?", 
    "Spor Dalindaki Rolunuz Nedir?",
    "Aktif olarak bir STK üyesi misiniz?", 
    "Hangi STK'nin Uyesisiniz?",
    "Stk Projesine Katildiniz Mi?", 
    "Girisimcilikle Ilgili Deneyiminiz Var Mi?",
    "Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?",
    "Ingilizce Biliyor musunuz?",
    "Ingilizce Seviyeniz?",
    "Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite",
    "Universite Kacinci Sinif"
]

# Doğum Tarihi sütununu düzenleme fonksiyonu
def temizle_dogum_tarihi(tarih, basvuru_yili):
    # Ay isimlerini numaraya çevir
    aylar = {
        'ocak': '01', 'subat': '02', 'mart': '03', 'nisan': '04', 'mayis': '05', 'haziran': '06',
        'temmuz': '07', 'agustos': '08', 'eylul': '09', 'ekim': '10', 'kasim': '11', 'aralik': '12',
        'january': '01', 'jan': '01', 'february': '02', 'feb': '02', 'march': '03', 'mar': '03', 
        'april': '04', 'apr': '04', 'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07', 
        'august': '08', 'aug': '08', 'september': '09', 'sep': '09', 'october': '10', 'oct': '10', 
        'november': '11', 'nov': '11', 'december': '12', 'dec': '12'
    }

    # Tarih string'inin sonundaki saat bilgisini çıkar ve kaydet
    saat_bilgisi = re.search(r'\d{1,2}(:|.)\d{2}', tarih)
    saat_bilgisi = saat_bilgisi.group() if saat_bilgisi else ''

    # Tarih string'inin sonundaki saat bilgisini kaldır
    if basvuru_yili in [2020, 2021, 2022]:
        tarih = re.sub(r'\s*\d{1,2}(:|.)\d{2}\s*$', '', tarih).strip()
    else:
        tarih = re.sub(r'\s*\d{1,2}(:)\d{2}\s*$', '', tarih).strip()

    # Ay isimlerini numaraya çevir
    for ay, numara in aylar.items():
        tarih = re.sub(r'\b' + ay + r'\b', numara, tarih)

    tarih = re.sub(r'[ /-]', '.', tarih)
    tarih = str(tarih)  # Tarihi string'e çevir
    tarih_seperate = tarih.split('.')
    
    # Tarih kısmı yeterli eleman içeriyor mu kontrol et
    if len(tarih_seperate) == 3:
        # Tarih formatı: DD.MM.YYYY
            try:
                gun = int(tarih_seperate[0])
                ay = int(tarih_seperate[1])
                yil = int(tarih_seperate[2])
                
                if ay > 12 or ay == 0:
                    tarih_seperate[0], tarih_seperate[1] = tarih_seperate[1], tarih_seperate[0]
                    gun = int(tarih_seperate[0])
                    ay = int(tarih_seperate[1])
                
                if gun>31:
                    tarih_seperate[0], tarih_seperate[2] = tarih_seperate[2], tarih_seperate[0]
                    gun = int(tarih_seperate[0])
                    yil = int(tarih_seperate[2])
                
                # Yıl iki basamaklıysa tamamla
                if len(str(yil)) == 2:
                    if yil > 50:
                        yil = 1900 + yil
                    else:
                        yil = 2000 + yil

                if len(str(yil))==1:
                    yil = 2000 + yil    

                ay = str(ay).zfill(2)
                gun = str(gun).zfill(2)
                yil = str(yil)

                tarih = f'{gun}.{ay}.{yil}'
            except ValueError:
                return '-'

    # Saat bilgisini ekle
    tarih = tarih + ' 00:00'

    return tarih

# Kardes Sayisi sütunundaki sayısal olmayan verileri atma ve sayısal değerleri int yapma
def temizle_kardes_sayisi(deger):
    try:
        # Sayısal bir değeri int'e çevir
        return int(float(deger))
    except:
        # Eğer sayısal değilse None olarak döndür
        return '0'

# Egitim durumu normalizasyonu
def egitim_durumu(deger):
    deger = str(deger).lower()  # Küçük harfe çevirme
    if "ortaokul mezunu" in deger or "ortaokul" in deger:
        return "ortaokul"
    elif "ilkokul mezunu" in deger or "ilkokul" in deger:
        return "ilkokul"
    elif "lise mezunu" in deger or "lise" in deger:
        return "lise"
    elif "universite mezunu" in deger or "universite" in deger:
        return "universite"
    elif "yuksek lisans doktora" in deger or "doktora" in deger:
        return "doktora"
    elif deger == "-" or deger == "0" or "egitimi yok" in deger or "egitim yok" in deger:
        return "egitimi yok"
    elif "yuksek lisans" in deger or "yuksek lisans" in deger:
        return "yuksek lisans"
    else:
        return deger

# Çalışma durumu normalizasyonu
def calisma_durumu(deger):
    deger = str(deger).lower()  # Küçük harfe çevirme
    if deger == "-" or deger == "emekli" or "hayir" in deger or "hayir" in deger:
        return "hayir"
    elif "evet" in deger or "evet" in deger:
        return "evet"
    else:
        return deger
    
# Noktalama işaretlerini kaldıran fonksiyon
def noktalama_temizle(metin):
    if pd.isna(metin):  # Boş hücreleri kontrol et
        return metin
    return re.sub(r'[^\w\s]', ' ', metin)  # Noktalama işaretlerini boşluk ile değiştir

# Her sütun için dönüştürme işlemini uygulayın
for column in columns_to_transform:

    df[column] = df[column].apply(noktalama_temizle)  # Noktalama işaretlerini kaldır
    df[column] = df[column].str.lower()  # Küçük harfe çevir
    df[column] = df[column].str.translate(str.maketrans("çğıöşü", "cgiosu"))  # Türkçe karakterleri değiştir
    df[column] = df[column].fillna('-')  # Boş hücrelere "-" koy
    df[column] = df[column].str.replace('i̇', 'i')

df["Burslu ise Burs Yuzdesi"] = df["Burslu ise Burs Yuzdesi"].fillna('-')

# Sütundaki her hücre için işlemi uygula

df['Anne Sektor'] = df['Anne Sektor'].replace(['0', 0, 0.0], '-')
df['Baba Sektor'] = df['Baba Sektor'].replace(['0', 0, 0.0], '-')
df['Kardes Sayisi'] = df['Kardes Sayisi'].apply(temizle_kardes_sayisi)
df['Anne Egitim Durumu'] = df['Anne Egitim Durumu'].apply(egitim_durumu)
df['Baba Egitim Durumu'] = df['Baba Egitim Durumu'].apply(egitim_durumu)
df['Anne Calisma Durumu'] = df['Anne Calisma Durumu'].apply(calisma_durumu)
df['Baba Calisma Durumu'] = df['Baba Calisma Durumu'].apply(calisma_durumu)
# "universitesi" kelimesini kaldır
df['Universite Adi'] = df['Universite Adi'].str.replace(' universitesi', '', case=False)
# "Universite Kacinci Sinif" sütunundaki değerleri dönüştürme
df['Universite Kacinci Sinif'] = df['Universite Kacinci Sinif'].replace({
    'tez': 'yuksek lisans',
    '0': 'hazirlik'
})
# bolumu ve alani kelimesini kaldır
df['Bölüm'] = df['Bölüm'].replace({
    ' bolumu': '',
    ' alani': ''
})
df['Spor Dalindaki Rolunuz Nedir?'] = df['Spor Dalindaki Rolunuz Nedir?'].replace(['0', 0, 0.0], '-')
# Doğum Tarihi sütununu temizle
df['Dogum Tarihi'] = df.apply(lambda row: temizle_dogum_tarihi(row['Dogum Tarihi'], int(row['Basvuru Yili'])), axis=1)
def get_il(dogum_yeri):
    # Önce ili kontrol et
    for il in iller:
        if il in dogum_yeri.lower():
            return il
    # İlçe bilgilerini kontrol et
    for ilce, il in ilceler.items():
        if ilce in dogum_yeri.lower():
            return il
    # Ne il ne de ilçe bulunursa None döner
    return "-"

# "Dogum Yeri" sütununu güncelleme
df['Dogum Yeri'] = df['Dogum Yeri'].apply(get_il)
df['Ikametgah Sehri'] = df['Ikametgah Sehri'].apply(get_il)
df['Lise Sehir'] = df['Lise Sehir'].apply(get_il)

def notlar_limit(num):
    """Belirli sınırları düzenleme işlevi"""
    if num == 2.50:
        return 2.49
    elif num == 3.00:
        return 2.99
    elif num == 3.50:
        return 3.49
    else:
        return num

def format_ortalama(value):
    """Veriyi belirtilen formatta dönüştürme"""
    if pd.isna(value):  # Eğer değer NaN ise '-' döndür
        return "-"
    
    # Değerin string olduğundan emin ol ve küçük harfe çevir
    value = str(value).lower().strip()
    
    # "ortalama bulunmuyor" veya "not ortalaması yok" varsa
    if value == "ortalama bulunmuyor" or value == "not ortalaması yok":
        return "-"
    
    # "ve altı" ifadesini "0 - X" formatına dönüştürme
    if "ve altı" in value:
        number_match = re.match(r'(\d+(\.\d+)?)', value)
        if number_match:
            num = float(number_match.group(1))
            return f"0.00 - {num:.2f}"
    
    # Sayıları ayırmak ve işlemler yapmak için regex
    match = re.match(r'(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)$', value)
    if match:
        # Sayıları al
        num1 = float(match.group(1))
        num2 = float(match.group(3))
        
        # Sayıları küçükten büyüğe sıralama
        if num1 > num2:
            num1, num2 = num2, num1
        
        # Büyük değeri ayarlama
        num2 = notlar_limit(num2)
        
        # Formatlama ve iki basamağa yuvarlama
        return f"{num1:.2f} - {num2:.2f}"
    
    else:
        # Hatalı formatları işlemek için
        return value

# 'Universite Not Ortalamasi' sütununu formatlama
df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].apply(format_ortalama)

def is_in_100_scale(value):
    # Değerin 100'lük sistemde olup olmadığını kontrol eden fonksiyon.
    # Eğer değer 100'lük sisteme ait bir aralık veya sayı ise True döner.
    if isinstance(value, str):
        return bool(re.match(r'(\d{2,3}(\.\d+)?)\s*(-\s*\d{2,3}(\.\d+)?)?$', value))
    return False

def convert_to_100_scale(value):
    # 4'lük sistemi 100'lük sisteme çevirme ve tam sayıya yuvarlama
    try:
        value = float(value)
        return int((value / 4.0) * 100)  # 4'lük sistemi 100'lük sisteme çevir ve tam sayıya yuvarla
    except ValueError:
        return None  # Eğer float değilse, None döndür

def format_ortalama_lise(value):
    # Veriyi belirtilen formatta dönüştürme
    if pd.isna(value):  # Eğer değer NaN ise '-' döndür
        return "-"
    
    # Değerin string olduğundan emin ol ve küçük harfe çevir
    value = str(value).lower().strip()
    
    # "ortalama bulunmuyor" veya "not ortalaması yok" varsa
    if value == "ortalama bulunmuyor" or value == "not ortalaması yok":
        return "-"
    
    # Eğer değer 100'lük sistemdeyse, olduğu gibi bırak
    if is_in_100_scale(value):
        return value
    
    # "ve altı" ifadesini "0 - X" formatına dönüştürme
    if "ve altı" in value:
        number_match = re.match(r'(\d+(\.\d+)?)', value)
        if number_match:
            num = float(number_match.group(1))
            return f"0 - {int(num * 25)}"  # 4'lük sisteme göre 25 ile çarparak çeviriyoruz
    
    # Sayıları ayırmak ve işlemler yapmak için regex
    match = re.match(r'(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)$', value)
    if match:
        # Sayıları al
        num1 = float(match.group(1))
        num2 = float(match.group(3))
        
        # Sayıları küçükten büyüğe sıralama
        if num1 > num2:
            num1, num2 = num2, num1
        
        # 100'lük sisteme çevir
        num1_100 = convert_to_100_scale(num1)
        num2_100 = convert_to_100_scale(num2)
        
        # Tam sayıları aynen bırak
        num1_str = f"{num1_100}"
        num2_str = f"{num2_100}"
        
        # Aralık formatını döndür, her iki tarafında birer boşluk olacak şekilde
        return f"{num1_str} - {num2_str}"
    
    else:
        try:
            # Tek bir not ortalaması varsa onu 100'lük sisteme çevir ve tam sayıya yuvarla
            num = float(value)
            num_100 = convert_to_100_scale(num)
            
            # Eğer değer 0 ise aynen bırak
            if num_100 == 0:
                return "0"
            return f"{num_100}"
        except ValueError:
            return value
# Belirtilen sütunu formatlama
df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].apply(format_ortalama_lise)

# 'Lise Turu' sütunundaki verileri kontrol eden ve güncelleyen fonksiyon
def update_lise_turu(text):
    text = text.lower()  # Küçük harfe çeviriyoruz
    if "ozel lisesi" in text or "ozel lise" in text:
        return "ozel"
    elif "duz lise" in text:
        return "devlet"
    return text  # Eğer koşullardan biri karşılanmazsa, mevcut değer kalır
    
# 'Lise Turu' sütunundaki 'lisesi' kelimesini çıkaran fonksiyon
def remove_lisesi(text):
    return text.replace("lisesi", "").strip()

# 'Lise Turu' sütununu güncelle
df['Lise Turu'] = df['Lise Turu'].apply(remove_lisesi)
df['Lise Turu'] = df['Lise Turu'].apply(update_lise_turu)

# Burs terimleri listesi
burs_terimleri = [
    "kyk", "t3", "tübitak", "tev", "universite", "gsb", "vakfı", "tobb", "derneği", 
    "dernek", "özel", "meb", "vakıf", "yok", "tskev", "tanıdık", "nevader", 
    "bakanlıgı", "kulup", "yasar", "odtu", "depremzede", "geri odemeli", 
    "kredi", "ksv", "okul", "kök", "derece burs"
]

# Belirli terimleri içerenleri cümle şeklinde döndür
def format_burs_terimleri(text, burs_terimleri):

    # "kredi" ve "yurtlar" terimlerini içeriyorsa "kyk" olarak ayarla
    if "kredi ve yurtlar" in text:
        return "kyk"
    results = [term for term in burs_terimleri if term in text]
    if results:
        return ', '.join(results)  # Terimleri cümle gibi yaz
    else:
        return text  # Hiçbir terim içermiyorsa metni olduğu gibi bırak

    
# Burs terimlerini bul ve yeni bir sütun oluştur
df['Burs Aldigi Baska Kurum'] = df['Burs Aldigi Baska Kurum'].apply(lambda x: format_burs_terimleri(x, burs_terimleri))

# Noktalama işaretlerini sil (- * % dışında)
def remove_punctuation(text):
    if pd.isna(text):  # NaN değerini kontrol et
        return "-"
# "tl" ifadesini sayılardan bağımsız olarak kaldır
    text = re.sub(r'tl', '', text, flags=re.IGNORECASE)
    text = re.sub(r'lira', '', text, flags=re.IGNORECASE)
    text = re.sub(r'turk lirasi', '', text, flags=re.IGNORECASE)
    return re.sub(r'[^\w\s\-\*%]', '', str(text))

def process_numbers(text):
    if pd.isna(text):  # Eğer değer NaN ise, NaN olarak geri dön
        return text
    
    if isinstance(text, float):  # Eğer veri float ise, olduğu gibi geri dön
        return text
    
    # "burs almıyorum" varsa, "0" yaz
    if "almiyorum" in text.lower():
        return "0"
    
    # "bin turk lirasi" varsa, "1000" yaz
    if "bin" in text.lower():
        return "1000"
    
    # "yillik belirlenen kapsamli burs kadar" varsa, "-" koy
    if "yillik belirlenen kapsamli burs kadar" in text.lower():
        return "-"
    
    # "darussafaka lisesi 375 kyk 400" varsa, "775" yaz
    if "darussafaka lisesi 375 kyk 400" in text.lower():
        return "775"
        # "%" varsa "-" koy
    if '%' in text:
        return "-"
        
# "*" var ise, öncesi ve sonrasındaki sayıları çarp
    if '*' in text:
        parts = re.split(r'\*', text)
        if len(parts) == 2:
            left_num = int(re.findall(r'\d+', parts[0])[-1]) if re.findall(r'\d+', parts[0]) else 0
            right_num = int(re.findall(r'\d+', parts[1])[0]) if re.findall(r'\d+', parts[1]) else 0
            return str(left_num * right_num)
    
    # "ve" veya "+" varsa sayıları topla
    if "ve" in text or "+" in text:
        numbers = re.findall(r'\d+', text)
        numbers = [int(n) for n in numbers if int(n) >= 12]
        return str(sum(numbers)) if numbers else text
    
    # Birleşik yazılmış sayılar (örneğin "300kadar") durumunu ele al
    combined_numbers = re.findall(r'\d+', text)
    combined_numbers = [int(n) for n in combined_numbers if int(n) >= 12]

    if combined_numbers:
        if len(combined_numbers) > 1:
            return str(max(combined_numbers))
        return str(combined_numbers[0])
    
    # Eğer birden fazla sayı varsa en büyüğünü döndür
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers if int(n) >= 12]
    if len(numbers) > 1:
        return str(max(numbers))
    
    # Eğer sadece bir sayı varsa, o sayıyı döndür
    return str(numbers[0]) if numbers else text


# Noktalama işaretlerini kaldır ve diğer işlemleri uygula
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].apply(remove_punctuation)
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].str.lower()
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].str.translate(str.maketrans("çğıöşü", "cgiosu"))
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].fillna('-')
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].str.replace('i̇', 'i')


def categorize_amount(amount):
    if pd.isna(amount):  # Eğer değer NaN ise, NaN olarak geri dön
        return amount
    
    # Temizlenmiş sayıyı al
    clean_amount = process_numbers(amount)
    
    try:
        # Sayıya çevir
        value = int(clean_amount)
    except ValueError:
        return "-"  # Sayıya çevrilemeyen değerler için
    
    # Sayıyı aralıklara göre sınıflandır
    if value >= 1000:
        return "1000₺ ve üstü"
    elif 500 <= value <= 999:
        return "500₺ - 999₺"
    elif 0 <= value <= 499:
        return "0 - 499₺"
    else:
        return "-"

# Örnek kullanım
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].apply(categorize_amount)

def categorize_sport_role(role):
    # "lider" veya "kaptan" varsa "Lider/Kaptan" yaz
    if "lider" in role or "kaptan" in role:
        return "lider/kaptan"
    
    # "bireysel" varsa "bireysel spor" yaz
    if "bireysel" in role:
        return "bireysel spor"
    
    # Diğer durumlar için orijinal rolü döndür
    return role

# Örnek kullanım
df['Spor Dalindaki Rolunuz Nedir?'] = df['Spor Dalindaki Rolunuz Nedir?'].apply(categorize_sport_role)

    # "Lise Adı" sütunundaki 'lisesi' ve 'koleji' kelimelerini kaldır 
df['Lise Adi'] = df['Lise Adi'].str.replace(' lisesi', '', case=False)
df['Lise Adi'] = df['Lise Adi'].str.replace(' koleji', '', case=False)

# Sözlüğü tanımla
categories = {
    "sayisal": ["mf", "sayisal", "fen sayisal", "sayisal2 yil", "sayisal matematik", "sayisal fen", "sayisal bolumu", "sayisal fen bilimleri", "sayisal fen bolumu", "sayisal 2 yil", "sayisal bol", "sayisal 1 yil", "sayisal alan", "sayisal bolumu", "sayisal matematik fen bilimleri", "sayisas", "m f", "sasyisal", "mf", "mf sayisal", "mf matematik fen", "mf fen bilimleri", "mf agirlik", "mf agirlikli", "mf sayisal", "mf fen", "mf bolum", "mf mezuniyet notu 87", "mf fen matematik", "mf sayisal", "mf matematik fen bilimleri", "mf matematik fen", "sayilsal", "sayisak", "kimya", "sayiasal", "fizik matematik", "fm", "fen", "fen bilimleri", "fen matematik", "fen bilimleri sayisal", "fen ve matematik", "fen bolumu", "fen matematik bolumu", "fen bilimleri bolumu", "fen bilimleri alani", "fen bilimleri sayisal", "fen bilimleri sayisal bolumu", "fen bilimleri", "fen bilimler", "fen bilimlerisayisal", "fen ve matematik bolumu"],
    "esit agirlik": ["tm", "tm sayisal", "tm sosyal bilimler", "tm sosyal", "tm dil", "tm bolum yok", "tm matematik", "tm esit agirlik", "tm 2 yil", "tm alan ayrimi yok", "tm ve matematik", "tm sosyal bilimler", "tm dil ve edebiyat", "tm social", "tm mf","turkce matematik bolumu", "esit agialik", "t m", "turkce matematitj", "matematik turkce", "turkce matematik alani", "turkce mat", "turkce matematik ve uluslararasi bakalorya", "turkce  matematik", "turkce matematil", "turkce matematik", "turkce   matematik", "turkce ve matematik", "esit agirlik", "ea", "ea esit agirlik", "ea tm", "ea bolumu", "ea 1 yil", "ea esit agirlik bolumu", "ea ve tm", "ea sosyal bilimler", "ea turkce matematik", "ea esit agirlik", "turkce  matematik esitagirlik", "esit agirlikli programlar", "matematik  turkce", "t m", "esit agitlik", "esit agrlik", "esit agirlik bolumu", "esit agilik", "esit agirlik", "esit agirluj", "turkce matemetik", "esit agirlak", "esit agirlikturkce matematik", "esitagirlil", "es it agirlik", "esit agirsik", "e a", "turkce  mathematik", "esit agarlik", "turkce_matematik","esit agirlik almanca", "alan turkce matematik", "turkce matematik esit agirlik", "turkce matematim", "esit agirlil", "turce matematik", "esir agirlik", "trkce matematik", "tutkce matematik", "esit agirlik turkce matematik", "esit agirlik turkce matematik", "esit agirlik turkce matematik alani", "esitagirlik"],
    "sozel": ["ts", "ts 1 2", "ts 10 sinif", "ts sosyal bilimler", "ts 2 yil", "ts sozel", "ts bolum yok", "ts 1 yil", "ts sosyal", "ts dil", "t s","soz", "edebiyat","ozel", "sozle", "soyal bilimler",  "sozel", "sozel bolum", "sozel sosyal", "sozel bilimler", "sozel turkce", "sozel bolum", "sozel edebiyat", "sozel ve sosyal bilimler","sosyal bilimler", "sosyal bilimler ve edebiyat", "sosyal bilimler bolumu","sosyal bilgiler", "sosyal bilimler ve edebiyat", "sosyal bilimler", "sosyal bilimler","sosyal bilimler ve edebiyat bolumu", "sosyal bilimler ve sosyal", "sosyal bilimler","tarih"],
    "diger": ["teknik harita tapu kadastro", "otomotiv teknolojileri", "mekatronik", "elektrik elektronik", "elektrik elektronik teknolojisi", "endustriyel otomasyon teknolojileri", "endustriyel otomasyon", "otomasyon sistemleri", "elektrik elektronik teknikerligi","elektrik elektronik teknolojisi alani", "elektrik tesisat ve pano", "kontrol ve enstrumantasyon teknolojisi", "automasyon", "otonomatik kumanda", "askeri lise", "askeri", "radyo ve televizyon", "radyo tv sinema", "radyo televizyon","radyoloji", "radyoloji teknisyenligi", "maliye", "arkeoloji", "iklimlendirme ve tesisat tenolojisi", "gida teknolojisi","bilimsel", "endustriyel otomasyon teknolojileri", "bulum ayrimi yok", "bilisimteknolojileri","buro yonetimi ve yonetici asistanligi", "buro yonetimi ve yonetici asistanligi", "sosyoloji", "genel kultur", "buro yonetimi", "bilgisayar ve yazilim", "konaklama ve seyahat", "konaklama ve seyahat hizmetleri", "konaklama ve seyahat", "diger","bilgisayar", "bilgisayar programciligi", "bilgisayar destekli makina ressamligi", "bilgisayar programciligi web tasarim", "bilgisayar teknolojileri","bilgisayarli muhasebe", "bilgisayarli makina imalati", "bilgisayar donanimi","bilgisayar web programciligi", "bilgisayar teknolojileri ve bilisim sistemleri","veri tabani programciligi", "web tasarim", "web programciligi","muhasebe", "muhasebe ve finansman", "muhasebe finans", "muhasebe ve bilgisayarli muhasebe", "muhasebe ve finansman dis ticaret", "muhasebe finansman dis ticaret", "muhasebe bolumu",         "muhasebe ve finans", "muhasebe finans", "muhasebe ve finansman dis ticaret","halkla iliskiler ve organizasyon", "halkla iliskiler ve organizasyon hizmetleri","konaklama ve seyahat hizmetleri", "konaklama ve seyahat", "konaklama hizmetleri","turizm ve otelcilik", "turizm", "yiyecek icecek hizmetleri", "yiyecek ve icecek","yiyecek icecek mutfak","imam hatip", "imam hatip lisesi", "anadolu imam hatip", "duz imam hatip programi","mesleki egitim veren bolum", "meslek", "onburo", "tip sekreterligi","meslek", "buro yonetimi ve sekreterlik", "buro yonetimi", "giyim uretim tekstil teknolojisi","ahsap teknolojisi mobilya imalati", "grafik ve fotograf", "grafik ve fotografcilik", "grafik tasarim", "moda tasarimi","giyim uretim teknolojisi kadin giyim modelistligi", "hazir giyim","metal teknolojisi", "tekstil teknolojisi", ],
    "dil": ["yabanci diil ingilizce", "ingilizce agirlikli", "ydl", "ingilizce egitim", "ingilizce","dil", "yabanci dil", "yabanci dil ingilizce", "yabanci dil ing", "yabanci dil","yabanci dil almanca","yabanci dilingilizce", "dil bolumu", "yabanci dilingilizce", "dil alani", ]
}

# 'Lise Bolumu' sütunundaki her değeri kontrol et
def categorize_value(value):
    value = value.lower()
    for category, keywords in categories.items():
        if any(keyword in value for keyword in keywords):
            return category
    return "diger"

df['Lise Bolumu'] = df['Lise Bolumu'].apply(categorize_value)
# '-' karakterlerini NaN ile değiştirme
df.replace('-', np.nan, inplace=True)

df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].replace({
    '0 - 499₺': '0 - 499 ₺',
    '1000₺ ve üstü': '1000₺ ve ustu'
})


stop_words = set(stopwords.words('turkish'))
# Zemberek'i tanımlama
ZEMBEREK_PATH = r"zemberek-full.jar" #zemberek jar dosyasınının dosya yolunu ekleyiniz
if not jpype.isJVMStarted():
    startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))


# Zemberek sınıflarını yükleme
morphology = JClass('zemberek.morphology.TurkishMorphology').createWithDefaults()
Paths = JClass('java.nio.file.Paths')

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
                lemma = str(best_analysis.get(0).getLemmas()[0])  # Lemmatize edilen hali
                if lemma.lower() == 'unk':  # Eğer lemmatize hali 'UNK' ise orijinal token'ı ekle
                    lemmatized_tokens.append(token)
                else:
                    lemmatized_tokens.append(lemma)
            else:
                lemmatized_tokens.append(token)  # Lemmatize edilemeyen kelimeyi olduğu gibi ekle
    return ' '.join(lemmatized_tokens)

def stk_temizleme(cumle):
    if pd.isna(cumle):
        return ''  # Eğer cümle NaN veya None ise boş string döndür
    
    # Metni ön işleme
    cumle = str(cumle).lower()  # String'e çevir ve küçük harfe dönüştür
    cumle = ' '.join([kelime for kelime in cumle.split() if kelime not in stop_words and not kelime.isdigit()])  # Stopword'leri ve sayıları kaldır
    cumle = cumle.translate(str.maketrans('', '', string.punctuation))  # Noktalama işaretlerini kaldır
    cumle = re.sub(r'[^\w\s]', ' ', cumle)  # Özel karakterleri boşlukla değiştir
    cumle = cumle.replace('\n', ' ')  # Yeni satır karakterlerini boşlukla değiştir
    cumle = re.sub(r'\s+', ' ', cumle).strip()  # Fazla boşlukları tek boşluğa indir ve kenar boşluklarını temizle
    
    # Lemmatization uygula
    cumle = lemmatize_text(cumle)
    return cumle

# Uygulama
df["Hangi STK'nin Uyesisiniz?"] = df["Hangi STK'nin Uyesisiniz?"].apply(stk_temizleme)
df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"] = df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"].apply(stk_temizleme)

df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].replace({
    'hazırlığım': 'hazirligim',
    '1.00 - 2.49': '0 - 1.79',
    '0.00 - 2.50': '0 - 1.79',
    '0.00 - 1.79':'0 - 1.79',
    '2.00 - 2.49': '1.80 - 2.49',
    '3.00 - 4.00': '3.00 - 3.49'
})
df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].replace({
    '0 - 62': '25 - 49',
    '0 - 1100': '25 - 49',
    '25 - 50': '25 - 49',
    '0 - 625':'25 - 49',
    '0 - 600': '25 - 49',
    '50 - 75': '50 - 74',
    '54-45': '50 - 74',
    '69-55': '50 - 74',
    '62 - 75':'50 - 74',
    '75 - 87': '75 - 100',
    '87 - 100': '75 - 100',
    '100-85': '75 - 100',
    '84-70': '75 - 100',

})

df.to_csv(r"temizlenmis_veri.csv", index=False) #temizlenen temizlenmis_veri.csv dosyasın kaydedilmesini istediğiniz dosya yolunu ekleyiniz

