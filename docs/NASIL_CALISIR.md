# Proje nasıl çalışıyor? (Mülakat hazırlık notu)

Bu not, projeyi mülakatta kendinden emin anlatabilmen için yazıldı. Önce bütün resmi,
sonra her dosyayı, en sonda da muhtemel soruları ve cevap taslaklarını bulacaksın.

## 1. Tek cümlede proje

Banka şikâyetlerini LLM ile 7 ürün kategorisine ayırıp doğru ekibe yönlendiren, aciliyetini
belirleyen ve bunu klasik bir makine öğrenmesi modeliyle aynı veride karşılaştıran bir sistem.

**Neden önemli?** Bankalara günde binlerce şikâyet geliyor. Yanlış ekibe giden şikâyet
günlerce bekliyor, müşteri BDDK'ya gidiyor. Otomatik yönlendirme hem süreyi hem maliyeti düşürür.

## 2. Akış (sırasıyla çalıştırdığın komutlar)

| Adım | Dosya | Ne yapıyor |
| --- | --- | --- |
| 1 | `src/prepare_data.py` | CFPB dosyasını parça parça okur, metni olmayan ve kapsam dışı ürünleri atar, her kategoriden eşit sayıda şikâyet seçer, %80 eğitim / %20 test diye ayırır |
| 2 | `src/baseline.py` | TF-IDF + Logistic Regression modelini eğitim verisiyle eğitir, test verisinde ölçer |
| 3 | `src/evaluate_llm.py` | Test setinden her kategoriden 30 şikâyet seçer, hem LLM'e hem baseline'a aynı şikâyetleri sınıflandırtır |
| 4 | `src/report.py` | Sonuç tablolarını ve grafikleri `results/RESULTS.md` dosyasına yazar |
| 5 | `app.py` | Streamlit arayüzü: tek şikâyet, toplu dosya yükleme, model karşılaştırma |

Yardımcı dosyalar: `config.py` (kategoriler, ayarlar), `metrics.py` (iki modeli aynı şekilde
ölçen fonksiyon), `llm_classifier.py` (LLM'le konuşan kısım), `plots.py` (grafikler).

## 3. Bilmen gereken kavramlar (sade anlatım)

**TF-IDF:** Metni sayılara çevirir. Bir kelime bu şikâyette çok geçiyor ama diğer
şikâyetlerde az geçiyorsa yüksek puan alır. "escrow" kelimesi konut kredisi şikâyetlerinde
yüksek, "the" her yerde geçtiği için sıfıra yakın puan alır. `ngram_range=(1, 2)` ile
"late fee" gibi iki kelimelik ifadeleri de yakalarız.

**Logistic Regression:** TF-IDF sayılarından her kategori için bir olasılık hesaplar, en
yükseğini seçer. Hızlı, açıklanabilir, metin sınıflandırmada güçlü bir başlangıç modeli.
`class_weight="balanced"` sınıflar dengesizse azınlığa daha çok ağırlık verir.

**Zero-shot:** LLM'e hiç etiketli örnek göstermeden sadece kategorilerin tanımını verip
sınıflandırtmak. Baseline ise binlerce örnekle eğitildi. Karşılaştırmanın asıl ilginç yanı bu.

**Neden dengeli örneklem (stratified)?** Ham veride kredi sicili şikâyetleri çok baskın.
Dengelemezsek her şeye "kredi sicili" diyen bir model bile yüksek doğruluk alırdı.

**Accuracy vs. macro F1:** Accuracy, toplam doğru oranı. Macro F1 her kategorinin F1'ini
ayrı hesaplayıp ortalamasını alır, yani küçük kategorilerdeki başarısızlığı gizlemez.
F1 = precision ile recall'ın harmonik ortalaması.
- Precision: "Konut kredisi" dediklerimin kaçı gerçekten konut kredisi?
- Recall: Gerçek konut kredisi şikâyetlerinin kaçını yakaladım?

**Confusion matrix:** Satırlar gerçek kategori, sütunlar tahmin. Köşegen dışındaki sayılar
hangi kategorilerin birbirine karıştığını gösterir (ör. kart şikâyeti ↔ borç tahsilatı).

**Structured output + enum:** LLM'den serbest metin değil, belirli bir şemaya uyan JSON
istiyoruz (`ComplaintLabel` sınıfı). Kategori alanı sadece 7 değerden birini alabiliyor, yani
model olmayan bir kategori uyduramıyor ve cevabı parse etmeye gerek kalmıyor.

**Temperature 0:** Modelin rastgeleliğini en aza indirir. Aynı şikâyete her seferinde aynı
cevabı verir, ölçüm tekrarlanabilir olur.

**Önbellek (cache):** Her cevap `results/llm_cache.jsonl` dosyasına yazılıyor. Aynı şikâyet
tekrar sorulursa API çağrılmıyor, hem kota hem zaman kazanıyoruz. Prompt değişirse
`PROMPT_VERSION` artırılır, eski cevaplar kullanılmaz.

**Rate limit + exponential backoff:** Ücretsiz Gemini kotası dakikada birkaç istek. Kod
istekleri aralıklarla gönderiyor. 429 (çok fazla istek) veya 5xx hatası gelirse 5, 10, 20, 40
saniye bekleyip tekrar deniyor. 400 gibi bizim hatamız olan durumlarda boşuna tekrar denemiyor.

**Hatalar neden "yanlış" sayılıyor?** Başarısız çağrıyı listeden atsaydık skor yapay olarak
yükselirdi. Gerçek hayatta da sınıflandırılamayan şikâyet bir sorundur.

## 4. Kendi bilgisayarında çalıştırma (Windows)

```powershell
git clone https://github.com/aygunduzumutberke/<repo-adi>.git
cd <repo-adi>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` dosyasını aç, `GEMINI_API_KEY=` satırına https://aistudio.google.com/apikey
adresinden aldığın anahtarı yapıştır.

Önce kurulumun çalıştığını gör (2 dakika, internet ve anahtar gerekmez):

```powershell
python scripts\make_demo_data.py
python -m src.prepare_data --demo
python -m src.baseline
python -m src.evaluate_llm --per-class 5 --provider mock
streamlit run app.py
```

Sonra gerçek veri: README'deki "Full run on real data" adımları. LLM değerlendirmesi
yaklaşık 30 dakika sürer, bilgisayar açık kalsın. Yarıda kesilirse tekrar çalıştır,
önbellek sayesinde kaldığı yerden devam eder.

Demo veriyle çıkan %100 skorlar anlamsızdır (şablondan üretilmiş yapay metin). GitHub'a
sadece gerçek veriyle ürettiğin `results/` dosyalarını yükle.

## 5. Mülakat soruları ve cevap taslakları

**"Projeyi anlat."**
Bankalara gelen şikâyetlerin doğru ekibe yönlendirilmesini otomatikleştirmek istedim.
ABD'nin kamuya açık CFPB veritabanından 7 kategoride dengeli bir örneklem aldım. Klasik
bir TF-IDF + Logistic Regression modeli eğittim, Gemini'yi ise hiç örnek göstermeden
kullandım. İkisini aynı test şikâyetlerinde ölçtüm. Sonuç: [rakamlarını buraya koy].
Streamlit arayüzünde Türkçe şikâyetleri de sınıflandırabiliyor.

**"LLM mi daha iyi, klasik model mi?"**
Kendi rakamına göre cevapla. Genel çerçeve: Eğitim verisi bolsa klasik model çok ucuz, çok
hızlı ve doğrulukta yakın olabilir. LLM'in avantajı etiketli veri istememesi, başka dillerde
de çalışması ve özet ile aciliyet gibi ek bilgiler üretmesi. Gerçek bir bankada ikisini
birlikte kullanırdım. Klasik model ilk filtre olur, emin olmadığı durumlarda LLM'e sorulur.

**"Neden CFPB, neden Türk verisi değil?"**
Türkiye'de kamuya açık, etiketli ve lisansı net bir şikâyet veri seti bulamadım. Şikâyet
sitelerinden izinsiz veri toplamak istemedim. CFPB resmi, ücretsiz ve gerçek ürün
etiketleri var, bu da doğruluğu ölçmeyi mümkün kılıyor.

**"Modelin en çok neyi karıştırıyor?"**
Confusion matrix'e bak ve en büyük köşegen dışı hücreyi anlat. Ör. kart kaynaklı borcun
tahsilat şikâyeti hem "kart" hem "borç tahsilatı" olabiliyor. Bu bir model hatası olmaktan
çok, kategorilerin doğal örtüşmesi.

**"Prompt'u nasıl tasarladın?"**
Her kategorinin kısa tanımını verdim ve karışan durumlar için açık kurallar ekledim: kredi
raporundaki hata her zaman "credit reporting", borç tahsilat aramaları her zaman "debt
collection". Çıktıyı enum'lu bir şemayla sınırladım.

**"LLM'in aciliyet tahmini doğru mu?"**
Ölçemedim çünkü veride aciliyet etiketi yok. Bunu README'de sınırlama olarak yazdım.
Gerçek hayatta bir uzmanın birkaç yüz şikâyeti etiketlemesiyle ölçülebilir.

**"Maliyet ve hız?"**
Baseline şikâyet başına 1 milisaniyenin altında çalışıyor. LLM yüzlerce milisaniye sürüyor
ve ücretli katmanda token başı ücret var. Önbellek ve rate limit bunun için var.

**"Kişisel veriler?"**
CFPB yayınlamadan önce kişisel bilgileri XXXX ile maskeliyor. Gerçek bir bankada müşteri
verisini dış bir API'ye göndermeden önce maskelemek veya şirket içinde barındırılan bir model
kullanmak gerekir (KVKK).

**"Neyi geliştirirdin?"**
Etiketli Türkçe veri toplayıp Türkçe performansını ölçerdim. Klasik model + LLM hibrit
yapısını kurardım. Birkaç yüz şikâyeti elle etiketleyip aciliyet tahminini ölçerdim.

## 6. CV'ye nasıl yazılır

Gerçek sonuçları aldıktan sonra rakamları yerleştir:

> **Bank Complaint Classifier (LLM vs. Classical ML)** — Python, Gemini API, scikit-learn, Streamlit
> - Built a zero-shot LLM classifier with structured JSON output that routes banking complaints to 7 product teams and rates urgency; reached __% accuracy (macro F1 __) on CFPB complaints vs. __% for a TF-IDF + Logistic Regression baseline trained on __k examples.
> - Added caching, rate limiting with exponential back-off and a Streamlit app for single and batch (Excel) classification, including Turkish complaints.
