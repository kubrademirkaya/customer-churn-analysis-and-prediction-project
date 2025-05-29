# Telco Customer Churn Prediction and Exploratory Data Analysis (EDA)

## Proje Hakkında

Bu proje, bir telekomünikasyon şirketinin müşteri kaybını (churn) tahmin etmeye yönelik kapsamlı bir analitik çalışmanın başlangıç aşamasını temsil etmektedir. Müşteri kaybı, birçok sektörde olduğu gibi telekomünikasyon sektöründe de önemli bir iş sorunudur. Bu projenin amacı, müşterilerin neden hizmetlerini sonlandırdığını anlamak ve erken müdahale için potansiyel risk faktörlerini belirlemektir.

Projenin bu ilk aşaması, ham veriden değerli içgörüler elde etmeye odaklanan Keşifsel Veri Analizi (EDA) üzerine kuruludur. Gelecekte makine öğrenimi modelleri geliştirilerek müşteri kaybı tahmini yapılacaktır.

## Veri Seti

Bu projede kullanılan veri seti, Kaggle'dan alınmıştır. "Telco Customer Churn" veri seti, telekomünikasyon şirketi müşterilerine ait çeşitli bilgileri içermektedir. Veri seti, demografik bilgiler (yaş, cinsiyet, bağımlılıklar), abone olunan hizmetler (telefon, internet, güvenlik, yedekleme vb.), faturalandırma detayları (aylık fatura, toplam fatura, kontrat tipi) ve müşteri kaybı (`Churn Label`) gibi çeşitli özellikleri barındırmaktadır.

Veri Seti: https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/

## Gerçekleştirilen Analiz ve Keşifler (EDA)

Projenin bu aşamasında aşağıdaki temel analizler yapılmıştır:

### 1. Veri Temizleme ve Hazırlık
* Başlangıç veri seti incelenerek eksik değerler ve veri tipleri hakkında genel bir fikir edinilmiştir.

### 2. Tek Değişkenli Analiz (Univariate Analysis)
* **Kategorik Değişkenler:** Her kategorik özelliğin dağılımı (örn. cinsiyet, kontrat tipi, internet hizmeti) ve bunların `Churn` ile ilişkisi (churn eden ve etmeyen gruplar arasındaki dağılım farklılıkları) bar grafikleri (`countplot`) kullanılarak görselleştirilmiştir.
    * **Önemli Bulgular:** Aydan aya (Month-to-Month) sözleşmesi olan müşterilerin, Fiber Optik internet hizmeti kullananların ve kağıtsız fatura (Paperless Billing) kullananların churn oranlarının daha yüksek olduğu gözlemlenmiştir. Ayrıca, bazı şehirlerdeki müşteri kaybı oranlarının dikkat çekici derecede farklı olduğu tespit edilmiştir.
* **Sayısal Değişkenler:** Her sayısal özelliğin dağılımı histogramlar (`histplot`) ve kutu grafikleri (`boxplot`) ile incelenmiştir.
    * **Önemli Bulgular:** `Total Charges`, `Total Revenue`, `Number of Referrals`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges` gibi birçok değişkenin sağa çarpık (right-skewed) dağılımlara sahip olduğu ve "0" değerinde yoğunlaştığı (zero-inflated) belirlenmiştir. Bu durum, ileride veri dönüşümü (örn. logaritmik dönüşüm) ve özellik mühendisliği (örn. ikili özellik oluşturma) ihtiyacını ortaya koymuştur.

### 3. Çok Değişkenli Analiz (Multivariate Analysis)
* **Sayısal Değişkenlerin Churn ile İlişkisi:** Belirlenen kritik sayısal özelliklerin (`Tenure in Months`, `Monthly Charge`, `Total Charges`, `Total Revenue` vb.) `Churn` durumuna göre dağılımları yoğunluk grafikleri (KDE plots) ile karşılaştırılmıştır.
    * **Önemli Bulgular:**
        * **Daha kısa abonelik süresine (`Tenure in Months`) sahip müşterilerin churn etme olasılığı belirgin şekilde daha yüksektir.**
        * **Aylık faturaları (`Monthly Charge`) daha yüksek olan müşterilerin churn etme eğilimi daha fazladır.**
        * **Toplam faturaları (`Total Charges`) ve toplam gelirleri (`Total Revenue`) düşük olan müşterilerin churn etme olasılığı daha yüksektir.**
        * Referans vermeyen, iade almayan veya ekstra veri kullanmayan müşterilerin churn etme eğiliminin daha yüksek olduğu gözlemlenmiştir.
* **Korelasyon Analizi:** Sayısal değişkenler arasındaki doğrusal ilişkileri anlamak için bir korelasyon matrisi (heatmap) oluşturulmuştur.
    * **Önemli Bulgular:** `Total Charges` ve `Total Revenue` arasında **0.97** gibi aşırı yüksek bir pozitif korelasyon tespit edilmiştir. Benzer şekilde `Tenure in Months` ile `Total Charges` (0.87) ve `Total Revenue` (0.85) arasında da güçlü korelasyonlar belirlenmiştir. Bu durum, veri ön işleme aşamasında çoklu doğrusallık yönetimi (örn. bir değişkeni çıkarma) ihtiyacını göstermektedir.

## Kullanılan Teknolojiler

* **Python**
* **Pandas** (Veri manipülasyonu ve analizi için)
* **NumPy** (Sayısal işlemler ve diziler için)
* **Matplotlib** (Statik veri görselleştirmeleri için)
* **Seaborn** (İstatistiksel veri görselleştirmeleri için)

## Sonraki Adımlar

Projenin bir sonraki aşamasında, elde edilen içgörüler doğrultusunda veri ön işleme (eksik değer yönetimi, sayısal dönüşümler, ölçeklendirme), kategorik değişken kodlaması (One-Hot Encoding) ve özellik mühendisliği adımları gerçekleştirilecektir. Ardından, makine öğrenimi modelleri (Lojistik Regresyon, Karar Ağaçları, Random Forest, XGBoost vb.) eğitilip değerlendirilerek müşteri kaybı tahmini yapılacaktır.
