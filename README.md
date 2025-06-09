# Telco Customer Churn Prediction and Analysis with Gradio Interface

## Proje Hakkında

Bu proje, bir telekomünikasyon şirketinin müşteri kaybını (churn) tahmin etmeye yönelik kapsamlı bir analitik çalışmayı ve bu tahmini etkileşimli bir web arayüzü aracılığıyla sunmayı amaçlamaktadır. Müşteri kaybı, telekomünikasyon sektöründe önemli bir iş sorunudur ve erken müdahale için potansiyel risk faktörlerinin belirlenmesi kritik öneme sahiptir.

Projenin temel hedefleri:

- Ham müşteri verilerinden değerli içgörüler elde etmek için **Keşifsel Veri Analizi (EDA)** yapmak.
- Makine öğrenimi modellerini eğitmek için veriyi ön işlemek.
- **XGBoost** gibi güçlü bir sınıflandırma modeli kullanarak müşteri kaybını tahmin etmek.
- Eğitilmiş modeli ve ön işleme adımlarını kaydederek yeniden kullanılabilir hale getirmek.
- Kullanıcıların yeni müşteri verilerini girerek gerçek zamanlı churn tahmini yapabilmeleri için **Gradio** ile basit ve etkileşimli bir web arayüzü geliştirmek.

## Veri Seti

Bu projede kullanılan veri seti, Kaggle'dan alınmıştır. "[Telco Customer Churn](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/)" veri seti, telekomünikasyon şirketi müşterilerine ait çeşitli bilgileri içermektedir. Veri seti, demografik bilgiler (yaş, cinsiyet, bağımlılıklar), abone olunan hizmetler (telefon, internet, güvenlik, yedekleme vb.), faturalandırma detayları (aylık fatura, toplam fatura, kontrat tipi) ve müşteri kaybı ('Churn Label') gibi çeşitli özellikleri barındırmaktadır.

## Gerçekleştirilen Analiz ve Modeller

Proje, veri analizi ve makine öğrenimi modellemesi olmak üzere iki ana aşamada gerçekleştirilmiştir:

### 1. Keşifsel Veri Analizi (EDA)

Bu aşamada veri setinin yapısı, özellikleri ve müşteri kaybı ile ilişkileri derinlemesine incelenmiştir:

- **Veri Temizleme ve Hazırlık:** Başlangıç veri seti incelenerek eksik değerler ve veri tipleri hakkında genel bir fikir edinilmiştir.
- **Tek Değişkenli Analiz (Univariate Analysis):**
    - **Kategorik Değişkenler:** Her kategorik özelliğin dağılımı (örn. cinsiyet, kontrat tipi, internet hizmeti) ve bunların `Churn` ile ilişkisi (churn eden ve etmeyen gruplar arasındaki dağılım farklılıkları) bar grafikleri (countplot) kullanılarak görselleştirilmiştir.
        - **Önemli Bulgular:** Aydan aya (Month-to-Month) sözleşmesi olan müşterilerin, Fiber Optik internet hizmeti kullananların ve kağıtsız fatura (Paperless Billing) kullananların churn oranlarının daha yüksek olduğu gözlemlenmiştir. Ayrıca, bazı şehirlerdeki müşteri kaybı oranlarının dikkat çekici derecede farklı olduğu tespit edilmiştir.
    - **Sayısal Değişkenler:** Her sayısal özelliğin dağılımı histogramlar (histplot) ve kutu grafikleri (boxplot) ile incelenmiştir.
        - **Önemli Bulgular:** `Total Charges`, `Total Revenue`, `Number of Referrals`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges` gibi birçok değişkenin sağa çarpık (right-skewed) dağılımlara sahip olduğu ve "0" değerinde yoğunlaştığı (zero-inflated) belirlenmiştir. Bu durum, ileride veri dönüşümü (örn. logaritmik dönüşüm) ve özellik mühendisliği (örn. ikili özellik oluşturma) ihtiyacını ortaya koymuştur.
- **Çok Değişkenli Analiz (Multivariate Analysis):**
    - **Sayısal Değişkenlerin Churn ile İlişkisi:** Belirlenen kritik sayısal özelliklerin (`Tenure in Months`, `Monthly Charge`, `Total Charges`, `Total Revenue` vb.) `Churn` durumuna göre dağılımları yoğunluk grafikleri (KDE plots) ile karşılaştırılmıştır.
        - **Önemli Bulgular:**
            - Daha kısa abonelik süresine (`Tenure in Months`) sahip müşterilerin churn etme olasılığı belirgin şekilde daha yüksektir.
            - Aylık faturaları (`Monthly Charge`) daha yüksek olan müşterilerin churn etme eğilimi daha fazladır.
            - Toplam faturaları (`Total Charges`) ve toplam gelirleri (`Total Revenue`) düşük olan müşterilerin churn etme olasılığı daha yüksektir.
            - Referans vermeyen, iade almayan veya ekstra veri kullanmayan müşterilerin churn etme eğiliminin daha yüksek olduğu gözlemlenmiştir.
    - **Korelasyon Analizi:** Sayısal değişkenler arasındaki doğrusal ilişkileri anlamak için bir korelasyon matrisi (heatmap) oluşturulmuştur.
        - **Önemli Bulgular:** `Total Charges` ve `Total Revenue` arasında `0.97` gibi aşırı yüksek bir pozitif korelasyon tespit edilmiştir. Benzer şekilde `Tenure in Months` ile `Total Charges` (`0.87`) ve `Total Revenue` (`0.85`) arasında da güçlü korelasyonlar belirlenmiştir. Bu durum, veri ön işleme aşamasında çoklu doğrusallık yönetimi (örn. bir değişkeni çıkarma) ihtiyacını göstermiştir.
          
### 2. Veri Ön İşleme ve Model Eğitimi

EDA'dan elde edilen içgörüler doğrultusunda, model eğitimi için aşağıdaki veri ön işleme adımları uygulanmıştır:

- **Kategorik Değişken Kodlaması:**
    - İkili ve ordinal kategorik sütunlar için **Label Encoding** uygulanmıştır.
    - Nominal kategorik sütunlar için **One-Hot Encoding** (`pd.get_dummies` ile `drop_first=True` parametresi kullanılarak) uygulanmıştır. `drop_first=True` kullanımı, çoklu doğrusallık sorununu azaltmaya yardımcı olmuştur.
- **Sayısal Dönüşümler:** Sağa çarpık dağılıma sahip sayısal değişkenler (`Population`, `Avg Monthly GB Download`, `Total Long Distance Charges`, `Total Revenue`) için **logaritmik dönüşüm** (`np.log1p`) uygulanmıştır.
- **Özellik Ölçeklendirme:** Tüm sayısal özellikler (yaş, nüfus, abonelik süresi, faturalar vb.) **StandardScaler** kullanılarak ölçeklendirilmiştir.
- **Model Seçimi ve Eğitimi:** Müşteri churn tahmin problemi için güçlü performansı ve hızından dolayı **XGBoost Sınıflandırıcısı** seçilerek eğitilmiştir.

### 3. Model ve Ön İşleme Objelerinin Kaydedilmesi

Canlı tahmin arayüzünde kullanılmak üzere, eğitim aşamasında oluşturulan tüm kritik objeler `joblib` kütüphanesi ile kaydedilmiştir:

- **`xgboost_churn_model.joblib`**: Eğitilmiş XGBoost modeli.
- **`scaler.joblib`**: StandardScalar nesnesi.
- **`label_encoders.joblib`**: Tüm Label Encoder'ları içeren sözlük.
- **`model_features.joblib`**: Modelin beklediği özelliklerin (sütunların) sıralı listesi. Bu, OHE sonrası oluşan tüm yeni sütun isimlerini de içerir.
- **`nominal_categories.joblib`**: OHE uygulanmış nominal sütunların orijinal kategorik değerlerini içeren sözlük (Gradio Dropdown'ları için kullanılır).

## Gradio Etkileşimli Arayüzü

Projenin son aşamasında, eğitilmiş modelin ve kaydedilmiş ön işleme adımlarının kullanılabildiği basit ve etkileşimli bir web arayüzü **Gradio** ile geliştirilmiştir. Bu arayüz, kullanıcıların çeşitli müşteri özelliklerini girerek churn etme olasılığını anında tahmin etmelerini sağlar.

Arayüzün temel özellikleri:

- Kullanıcı dostu form alanları (radio butonları, checkbox, dropdown menüler, slider'lar).
- Girilen verilere, modelin eğitimde gördüğü aynı ön işleme adımlarının (Label Encoding, One-Hot Encoding, Log dönüşümü, Ölçeklendirme) uygulanması.
- İşlenmiş verinin XGBoost modeline gönderilerek churn olasılığının ve sınıflandırmasının (`Müşteri Churn Edecek` veya `Müşteri Churn Etmeyecek`) gösterilmesi.

## Kullanılan Teknolojiler

- **Python**
- **Pandas**: Veri manipülasyonu ve analizi için.
- **NumPy**: Sayısal işlemler ve diziler için.
- **Matplotlib**: Statik veri görselleştirmeleri için.
- **Seaborn**: İstatistiksel veri görselleştirmeleri için.
- **Scikit-learn**: Ön işleme (LabelEncoder, StandardScaler) ve modelleme (XGBoost ile uyumlu) için.
- **XGBoost**: Güçlü ve verimli bir gradient boosting kütüphanesi.
- **Gradio**: Hızlı ve kolay makine öğrenimi demoları oluşturmak için.
- **Joblib**: Python objelerini hızlıca diskten kaydetmek ve yüklemek için.
- **Git LFS**: Büyük dosyaların (model dosyaları gibi) Git depolarında verimli bir şekilde yönetilmesi için.

---

## Kurulum ve Çalıştırma (Yerel)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Depoyu Klonlayın:**Bash
    
    `git clone https://github.com/kubrademirkaya/customer-churn-analysis-and-prediction-project.git
    cd telco-customer-churn-prediction`
    
2. **Sanal Ortam Oluşturun (Önerilir):**Bash
    
    `python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # veya
    .\venv\Scripts\activate  # Windows`
    
3. **Gerekli Kütüphaneleri Yükleyin:** Projenin bağımlılıklarını `requirements.txt` dosyasından yükleyin:Bash
    
    `pip install -r requirements.txt`
    
    (`requirements.txt` dosyasının içeriği: `gradio`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib` olmalıdır.)
    
4. **Veri Setini İndirin:** [Kaggle Telco Customer Churn veri setini](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/) indirin ve `Telco_Customer_Churn.csv` adlı dosyayı projenizin ana dizinine yerleştirin.
5. **Modeli Eğitin ve Objeleri Kaydedin:** Projenin kök dizininde bulunan **veri ön işleme ve model eğitimini içeren Jupyter Notebook dosyasını (`EDA_and_Modeling.ipynb` gibi)** çalıştırın. Bu notebook, gerekli `.joblib` model ve ön işleme objelerini oluşturacaktır.
6. **Gradio Uygulamasını Başlatın:**
Klonladığınız dizinin kökünde bulunan `app.py` dosyasını çalıştırın:Bash
    
    `python app.py`
    
    Bu komut, Gradio arayüzünü otomatik olarak varsayılan bir tarayıcı penceresinde açacaktır.
    

---

## Canlı Demoyu Görüntüle (Hugging Face Spaces)

Bu projenin çalışan Gradio arayüzünü doğrudan bir web tarayıcısı üzerinden görüntülemek için aşağıdaki bağlantıyı kullanabilirsiniz:

[**Telco Churn Prediction Gradio Uygulaması**](https://huggingface.co/spaces/kubrademirkaya/telco-churn-prediction)

Uygulama, Hugging Face Spaces üzerinde barındırılmaktadır ve her yeni güncellemede otomatik olarak dağıtılır.

---

## Sonraki Adımlar (Gelecek Planları)

- Daha gelişmiş özellik mühendisliği tekniklerini keşfetmek.
- Farklı makine öğrenimi modellerini (örn. LightGBM, CatBoost) denemek ve performanslarını karşılaştırmak.
- Model optimizasyonu için hiperparametre ayarlama tekniklerini (örn. GridSearchCV, Optuna) uygulamak.
- Modelin açıklanabilirliğini artırmak için SHAP veya LIME gibi araçları kullanmak.

---
