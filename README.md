
Fine Tuning in Transfer Learning : Large-Language-Model (LLM) OLMo-1B Supaya Bisa Bahasa Indonesia
==================================================================================================

![captionless image](https://miro.medium.com/v2/resize:fit:1280/format:webp/0*s1TDsuyiUgQcWIsi.jpg)

[Reference](https://medium.com/@abdan.hafidz/fine-tuning-in-transfer-learning-large-language-model-llm-olmo-1b-supaya-bisa-bahasa-indonesia-97f4c4297f4e)

by [Abdan Hafidz](https://medium.com/@abdan.hafidz?source=post_page---byline--97f4c4297f4e---------------------------------------)




Bayangkan kamu punya temen pintar banget — lulusan luar negeri, fasih ngomong Inggris, ngerti segala topik mulai dari filsafat Yunani sampai teori kuantum. Tapi begitu diajak ngobrol pakai Bahasa Indonesia, dia cuma bisa jawab, “I’m sorry, I don’t understand.” Nah, kurang lebih itulah kondisi banyak **Large Language Model (LLM)** hari ini. Pintar, tapi nggak _merakyat_. Maka dari itu, di tutorial ini, kita akan “memaksa” (dengan penuh kasih sayang tentunya) model bahasa bernama **OLMo-1B** untuk belajar Bahasa Indonesia, biar bisa jadi lebih lokal, lebih nyambung, dan kalau bisa — sedikit lebih _ngapak_.

Kita bakal pakai dataset bernama **Cendol Indonesia Conversational Dataset** — karena apalagi yang lebih Indonesia daripada cendol dan obrolan warung kopi? Supaya modelnya nggak kaget dengan budaya lokal, kita fine-tune-nya pelan-pelan pakai metode **LoRA Adapter** dan teknik **Parameter-Efficient Fine-Tuning (PEFT)** — biar nggak berat di dompet maupun GPU. Plus, kita saring data pakai **TF-IDF + K-Means Clustering**, biar modelnya nggak cuma belajar dari data yang itu-itu aja, tapi juga dari ocehan netizen yang beragam, absurd, dan kadang bikin mikir ulang soal hidup.

Jadi… Apa Itu Transfer Learning dan Fine-Tuning?
================================================

Oke, sebelum kita nyemplung lebih dalam ke teknis, yuk kita kenalan dulu sama dua istilah keramat ini: **Transfer Learning** dan **Fine-Tuning**. Tapi tenang, ini bukan mata kuliah semester akhir kok. Santai aja.

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*HYGtMhx_lYcrUT9I)

**Transfer Learning** itu ibarat kamu punya temen yang udah jago di satu bidang — misalnya dia jago banget baca buku filsafat, ngerti konsep logika, dan bisa ngelucu pake kalimat Shakespeare. Nah, sekarang kamu mau ngajarin dia ngobrol kayak anak tongkrongan. Daripada ngajarin dari nol (yang makan waktu dan listrik), kamu cukup _“transfer”_ ilmu dasarnya, terus tinggal **tuning** sedikit biar bisa paham konteks lokal, kayak “gaskeun,” “gak relate,” atau “mager parah.”

Nah, proses _ngetuning-nya_ itu yang disebut **Fine-Tuning**.

Secara teknis:

*   **Transfer Learning** adalah pendekatan di mana kita ambil model yang udah dilatih sebelumnya (pretrained model), lalu kita pakai lagi untuk tugas baru yang mirip.
*   **Fine-Tuning** adalah proses melanjutkan pelatihan model tersebut, tapi dengan dataset baru (dalam hal ini, Bahasa Indonesia). Jadi kita nggak mulai dari nol, tapi dari “udah bisa ngomong, tinggal diajarin logat lokalnya.”

Kalau pakai analogi anak kost:
Daripada masak mie instan dari tepung, lebih baik beli mie yang udah setengah jadi, tinggal kasih bumbu Cendol dan cabai rawit, jadi deh mie rasa lokal!

Langsung aja kita mulai, gimana caranya?

1.  **Dataset Preparation & Processing**

Pertama tama, kita siapin dulu datasetnya. Kita ambil dari [https://huggingface.co/datasets/indonlp/cendol_collection_v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2)

Terlihat kalau isinya banyak sekali ada 1_2 juta lebih_ baris, untuk itu kita akan lakukan teknik sampling supaya pada saat kita melatih model dalam proses fine-tune waktu yang dibutuhkan tidak terlalu lama.

Teknik Sampling yang kita gunakan adalah dengan **_Diversity Sampling_** dan akan mengoptimalisasi konsumsi dataset sebanyak _10 ribu_ baris.

> **Diversity sampling** adalah teknik pemilihan data yang tujuannya bukan cuma “ambil yang banyak,” tapi “ambil yang beragam.” Jadi, daripada model dikasih data yang isinya seragam kayak chat GPT ngajarin basa basi, kita pastikan model belajar dari berbagai jenis gaya bahasa, topik, dan cara ngomong — biar nggak jadi robot yang cuma ngerti satu tipe percakapan doang. Ibarat ngajarin orang ngobrol, kita nggak mau dia cuma ngerti bahasa formal rapat Zoom, tapi juga ngerti gaya ngobrol anak Twitter, ibu-ibu WhatsApp, sampe netizen +62 di kolom komentar. Nah, _diversity sampling_ bikin data pelatihan lebih kaya dan representatif, sehingga model jadi lebih fleksibel dan nyambung ke banyak konteks.

Sampling kali ini dilakukan dengan bantuan dari Mas **Juang Maulana & Rafael,** mereka melakukan beberapa langkah di bawah ini dalam hal persiapan dataset :

Pertama kita persiapkan libraries yang dibutuhkan :

```
# Import necessary libraries
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings('ignore')
```

Kemudian konfigurasi parameter yang diperlukan :

```
# Configuration parameters
NUM_ROWS_SAMPLED = 10000  # Number of rows to sample
CUSTOM_K = None  # Set to a number to override the elbow method
MAX_FEATURES = 100  # Max features for TF-IDF
RANDOM_STATE = 42  # For reproducibility
SAVE_FIGURES = True  # Whether to save figurespy
```

1.  1. Load Dataset

Pada bagian ini, kita mencoba **memuat dataset** dari file lokal `sample_subset_indo_saja.parquet` untuk mempercepat proses eksperimen. Tapi kalau file-nya belum ada (atau hilang entah ke mana), kode ini akan secara otomatis **mengambil ulang dataset dari sumber aslinya**, yaitu Hugging Face Datasets. Dari kumpulan besar tersebut, kita **filter subset khusus** yang hanya berbahasa Indonesia—kayak `indo_puisi`, `wikihow`, `wikipedia_id`, sampai data dari `dolly` dan prompt seputar `safety` dan `identity`. Nah, supaya data yang kita ambil nggak itu-itu aja, kita gunakan teknik **stratified sampling berdasarkan** `**template_name**`. Ini bertujuan biar distribusi jenis data tetap terjaga. Tapi karena kadang hidup tak semulus teori—stratifikasi bisa gagal kalau datanya terlalu banyak kelas unik atau tidak cukup proporsional—maka kita siapin **fallback ke random sampling**. Setelah data di-_sampling_, hasilnya langsung disimpan dalam format `.parquet`, jadi kalau mau eksperimen ulang tinggal load aja, nggak perlu unduh dan proses dari awal lagi. Praktis, cepat, dan hemat internet kampus 😄

```
# Load the dataset (with error handling)
try:
    data = pd.read_parquet('sample_subset_indo_saja.parquet')
    print("Successfully loaded the dataset from parquet file.")
except:
    print("Could not find the parquet file. Loading from original source...")
    # Original dataset loading code
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    ds = load_dataset("indonlp/cendol_collection_v2")
    # Define the specific subsets to include
    subset_indo_saja = ds["train"].filter(lambda example: example["subset_name"] in {
        'indo_puisi', 'wikihow', 'wikipedia_id', 'safety_prompt', 'identity_prompt', 'dolly'
    })
    # Check if we have enough data for the requested sample size
    if len(subset_indo_saja) <= NUM_ROWS_SAMPLED:
        # If we have fewer rows than requested, use all available data
        sample_subset_indo_saja = subset_indo_saja
        print(f"Using all available data: {len(sample_subset_indo_saja)} rows")
    else:
        # Stratified sampling based on template_name to maintain distribution
        template_names = subset_indo_saja['template_name']
        try:
            # Try stratified sampling first
            _, sample_indices = train_test_split(
                range(len(subset_indo_saja)),
                test_size=NUM_ROWS_SAMPLED/len(subset_indo_saja),
                stratify=template_names,
                random_state=RANDOM_STATE
            )
        except ValueError:
            # Fallback to random sampling if stratify fails (e.g., with too many classes)
            print("Stratified sampling failed, using random sampling instead")
            np.random.seed(RANDOM_STATE)
            sample_indices = np.random.choice(
                range(len(subset_indo_saja)),
                size=NUM_ROWS_SAMPLED,
                replace=False
            )
        sample_subset_indo_saja = subset_indo_saja.select(sample_indices)
        print(f"Sampled {len(sample_subset_indo_saja)} rows out of {len(subset_indo_saja)}")
    # Convert to pandas DataFrame
    data = pd.DataFrame(sample_subset_indo_saja)
    data.to_parquet('sample_subset_indo_saja.parquet')
    print("Dataset loaded from source and saved to parquet file.")
```

Setelah menjalankan langkah di atas kita bisa melihat hasil dataset yang di-load salah satu hasilnya sebagai berikut :

```
Dataset shape: (10000, 7)
Dataset columns: ['dataset_name', 'subset_name', 'prompt_id', 'template_name', 'dataset_key', 'input', 'output']
Sample of the data:
  dataset_name   subset_name                             prompt_id  \
0  nusa_t2t_v2  wikipedia_id  64a2ab72-18e4-4e66-bb04-71f861e3ca06   
1  nusa_t2t_v2  wikipedia_id  200186d6-de5f-499f-bfc3-a49bfa6cccfd   
            template_name dataset_key  \
0     wikipedia_section_2       train   
1  wikipedia_subsection_0       train   
                                               input  \
0                        Stasiun Jūkujō itu apa sih?   
1  Tolong jelaskan mengenai Rumus Naegle dari Keh...   
                                              output  
0  adalah sebuah stasiun kereta api di kota Mizuh...  
1  Jika tahun tetap adalah  dari haid terakhir.\n...  
Distribution of subset_name:
subset_name
wikipedia_id                                 4758
wikipedia_ms                                 1624
wikipedia_min                                1123
wikipedia_su                                  362
wikipedia_jv                                  289
nusaparagraph_rhetoric_jav_nusantara_text     136
nusaparagraph_rhetoric_sun_nusantara_text     133
indo_puisi                                    121
nusaparagraph_rhetoric_bew_nusantara_text     119
nusaparagraph_rhetoric_min_nusantara_text     108
wikihow                                        77
wikipedia_ban                                  71
safety_prompt                                  71
nusaparagraph_rhetoric_mad_nusantara_text      67
identity_prompt                                67
nusaparagraph_rhetoric_mak_nusantara_text      67
dolly                                          54
wikipedia_gor                                  53
nusaparagraph_emot_jav_nusantara_text          47
nusaparagraph_topic_sun_nusantara_text         46
nusaparagraph_emot_sun_nusantara_text          44
wikipedia_map-bms                              43
nusaparagraph_topic_bew_nusantara_text         42
nusaparagraph_rhetoric_btk_nusantara_text      40
wikipedia_bjn                                  39
nusaparagraph_emot_mak_nusantara_text          34
nusaparagraph_topic_min_nusantara_text         34
nusaparagraph_topic_mak_nusantara_text         31
nusaparagraph_emot_bew_nusantara_text          31
nusaparagraph_topic_mad_nusantara_text         30
nusaparagraph_topic_jav_nusantara_text         30
nusaparagraph_emot_min_nusantara_text          27
nusaparagraph_rhetoric_bug_nusantara_text      23
nusaparagraph_topic_btk_nusantara_text         22
nusaparagraph_emot_btk_nusantara_text          21
nusaparagraph_emot_mad_nusantara_text          19
nusaparagraph_rhetoric_mui_nusantara_text      17
nusaparagraph_rhetoric_rej_nusantara_text      17
wikipedia_ace                                  13
nusaparagraph_emot_rej_nusantara_text          10
nusaparagraph_topic_rej_nusantara_text          8
wikipedia_tet                                   7
nusaparagraph_emot_mui_nusantara_text           7
wikipedia_nia                                   6
nusaparagraph_topic_mui_nusantara_text          4
nusaparagraph_emot_bug_nusantara_text           4
nusaparagraph_topic_bug_nusantara_text          4
Name: count, dtype: int64
Distribution of template_name:
template_name
wikipedia_section_0           464
wikipedia_subsection_4        459
wikipedia_section_daerah_0    455
wikipedia_subsection_0        448
wikipedia_section_2           446
wikipedia_section_4           441
wikipedia_section_3           439
wikipedia_section_daerah_3    437
wikipedia_section_daerah_1    428
wikipedia_section_1           423
Name: count, dtype: int64
```

1.  2. Embedding / Vectorization

```
# Prepare the data for clustering
# 1. Text feature extraction using TF-IDF
text_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,  # Limit features to avoid dimensionality issues
    stop_words='english',  # Remove English stop words (might need custom Indonesian stop words)
    ngram_range=(1, 2)  # Include unigrams and bigrams
)
# Check if 'text' column exists and find the appropriate text column
if 'text' in data.columns:
    text_column = 'text'
elif 'prompt' in data.columns:
    text_column = 'prompt'
else:
    # Find potential text columns
    potential_text_columns = []
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                if isinstance(data[col].iloc[0], str) and len(data[col].iloc[0].split()) > 5:
                    potential_text_columns.append(col)
            except:
                continue
    if potential_text_columns:
        text_column = potential_text_columns[0]
    else:
        raise ValueError("Could not find a suitable text column for analysis")
print(f"\nUsing '{text_column}' column for text vectorization")
# Fill NaN values in text column if any
data[text_column] = data[text_column].fillna("")
# Apply TF-IDF vectorization
try:
    text_features = text_vectorizer.fit_transform(data[text_column])
    print(f"Text features shape after TF-IDF: {text_features.shape}")
except Exception as e:
    print(f"Error during text vectorization: {e}")
    # Fallback to just using a small amount of text data if there's an issue
    text_features = text_vectorizer.fit_transform(data[text_column].astype(str).str[:1000])
    print(f"Using truncated text. Text features shape after TF-IDF: {text_features.shape}")
```

mempersiapkan data teks untuk proses **clustering**, dimulai dengan ekstraksi fitur berbasis **TF-IDF (Term Frequency–Inverse Document Frequency)**. Teknik ini digunakan untuk mengubah representasi teks mentah menjadi fitur numerik yang dapat diproses oleh algoritma machine learning. Proses dimulai dengan inisialisasi `TfidfVectorizer` dari Scikit-learn, dengan konfigurasi sebagai berikut:

*   `max_features=MAX_FEATURES`: membatasi jumlah fitur maksimum untuk menghindari permasalahan dimensi tinggi.
*   `stop_words='english'`: menghapus stopword bahasa Inggris, meskipun untuk kasus bahasa Indonesia sebaiknya diganti atau dikustomisasi dengan daftar stopword lokal.
*   `ngram_range=(1, 2)`: mempertimbangkan unigram dan bigram sebagai unit fitur untuk menangkap konteks kata yang lebih baik.

Selanjutnya, kode melakukan pengecekan terhadap kolom teks yang relevan. Secara default, akan digunakan kolom `text` jika tersedia; jika tidak, akan dicari `prompt`. Jika keduanya tidak tersedia, maka dilakukan proses heuristik untuk menemukan kolom bertipe string dengan konten teks yang cukup panjang (lebih dari lima kata). Ini penting untuk memastikan bahwa vektorisasi TF-IDF dilakukan terhadap kolom yang benar-benar berisi teks.

Setelah kolom teks berhasil diidentifikasi, seluruh nilai `NaN` pada kolom tersebut akan diisi dengan string kosong untuk mencegah error saat pemrosesan. Kemudian, vektorisasi TF-IDF dijalankan menggunakan `fit_transform()`. Jika terjadi error (misalnya karena karakter non-standar atau teks tidak bersih), maka akan digunakan fallback berupa **pemotongan teks hingga 1000 karakter pertama**, sebagai bentuk pertolongan pertama untuk memastikan proses ekstraksi fitur tetap dapat berjalan.

Hasil akhir dari proses ini adalah matriks sparse `text_features` yang merepresentasikan setiap dokumen dalam bentuk vektor berdimensi `max_features`, siap digunakan dalam tahap clustering berikutnya.

Why TF-IDF?
===========

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YMGWkymYnXlv6kTpZCzBlw.png)

Penggunaan **TF-IDF sebelum K-Means Clustering** adalah langkah krusial dalam pipeline unsupervised learning berbasis teks. Hal ini karena algoritma **K-Means** bekerja pada ruang fitur numerik dan menggunakan **jarak Euclidean** atau varian lain untuk mengelompokkan data. Sementara itu, teks dalam bentuk mentah (string) tidak bisa langsung digunakan dalam perhitungan matematis. Oleh karena itu, diperlukan proses **vektorisasi** yang mengubah dokumen teks menjadi representasi numerik yang bermakna.

Di sinilah **TF-IDF (Term Frequency–Inverse Document Frequency)** berperan. TF-IDF tidak hanya menghitung frekuensi kemunculan kata dalam dokumen (seperti bag-of-words), tetapi juga memberikan bobot yang lebih tinggi untuk kata-kata yang spesifik terhadap dokumen tertentu dan lebih rendah untuk kata-kata yang terlalu umum di seluruh korpus. Hal ini membuat TF-IDF unggul dalam menjaga konteks penting dan karakteristik unik setiap teks dibandingkan metode sederhana seperti count vectorization.

Setelah teks diubah menjadi representasi TF-IDF, setiap dokumen kini menjadi vektor berdimensi tetap yang menggambarkan “makna statistik” dari teks tersebut. Vektor-vektor ini menjadi input ideal untuk algoritma **K-Means**, yang kemudian dapat menemukan pola-pola kedekatan antar dokumen dan mengelompokkannya berdasarkan kemiripan konteks atau tema.

Singkatnya, **TF-IDF menjembatani dunia teks dan dunia matematis** dengan mengubah teks menjadi angka, sehingga K-Means bisa melakukan tugasnya: membentuk cluster dokumen yang ‘mirip-mirip’ berdasarkan bobot kata yang relevan.

1.  2. Feature Engineering

```
# 2. Feature engineering - add categorical features as one-hot encoding
categorical_features = pd.get_dummies(data[['subset_name', 'template_name']], drop_first=True)
print(f"Categorical features shape: {categorical_features.shape}")
# Convert sparse matrix to dense for concatenation
text_features_dense = text_features.toarray()
# Combine features (text features and categorical features)
combined_features = np.hstack((text_features_dense, categorical_features.values))
print(f"Combined features shape: {combined_features.shape}")
# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)
# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
print(f"Reduced features shape: {reduced_features.shape}")
```

Setelah berhasil mengekstrak makna dari teks lewat TF-IDF, sekarang saatnya kita bilang, “Eh teks, kenalan dulu dong sama temen-temen kategorikal!” Yup, di tahap ini kita melakukan **feature engineering** — alias nambahin fitur-fitur tambahan yang bisa bantu model mengenali pola yang lebih kaya.

Kita mulai dengan mengubah kolom kategorikal seperti `subset_name` dan `template_name` menjadi representasi numerik lewat **One-Hot Encoding**. Tujuannya? Supaya model bisa tahu: "Oh, data dari Wikipedia dan data dari puisi itu beda vibes-nya!"

Setelah fitur kategorikal siap, kita gabungkan mereka dengan fitur teks (hasil TF-IDF tadi) jadi satu paket lengkap. Tapi karena TF-IDF ngasih hasil dalam bentuk **sparse matrix**, kita perlu ubah dulu ke format **dense array** biar bisa digabung bareng.

Kemudian, supaya semua fitur punya skala yang seimbang (biar gak ada yang dominan kayak anak teknik pas rebutan mic di presentasi), kita normalisasi semuanya dengan **StandardScaler**. Nah, setelah semua fitur udah _on the same level_, kita kurangi dimensinya pakai **PCA** (Principal Component Analysis) jadi dua dimensi — biar bisa kita plot dan lihat klasternya nanti.

📉 Intinya: dari teks → tambah konteks kategorikal → normalisasi → reduksi dimensi → siap visualisasi!
Dengan ini, kita nggak cuma tahu isi teksnya, tapi juga dari “keluarga” subset mana dia berasal.

**Standar Scaling**

Standard Scaling digunakan untuk menyamakan skala antar fitur dengan cara mengubah distribusi data menjadi memiliki mean = 0 dan standar deviasi = 1. Ini penting karena banyak algoritma ML (termasuk PCA dan K-Means) sensitif terhadap skala data.

![captionless image](https://miro.medium.com/v2/resize:fit:492/format:webp/0*NBf0OVQU20bJym0u.png)

**Principal Component Analysis (PCA)**

PCA adalah teknik **reduksi dimensi** yang bertujuan mencari kombinasi linear dari fitur asli yang bisa **menjelaskan varian data terbesar**. Artinya, kita ubah data berdimensi tinggi (ratusan fitur TF-IDF dan OHE) ke dimensi lebih rendah (biasanya 2D atau 3D), tapi tanpa terlalu banyak kehilangan informasi.

Secara matematis, PCA mencari **eigenvector dan eigenvalue** dari _covariance matrix_ data, lalu memproyeksikan data ke arah vektor-vektor utama (principal components) tersebut.

Jika dilakukan PCA di atas maka didapatkan hasil sebagai berikut :

![captionless image](https://miro.medium.com/v2/resize:fit:750/format:webp/1*30lJgBgAlm-uamOIII69mw.png)

1.  3. Klustering

**Elbow Method**

Setelah fitur kita siap, langkah selanjutnya adalah menentukan **berapa banyak klaster** yang paling masuk akal untuk membagi data kita. Di sinilah metode **Elbow Method** masuk.

Dalam proses clustering menggunakan algoritma seperti KMeans, salah satu tantangan utama adalah menentukan jumlah klaster (k) yang optimal. Elbow Method adalah pendekatan visual yang umum digunakan untuk memilih nilai k yang tepat dengan membandingkan nilai _inertia_ (dikenal juga sebagai _within-cluster sum of squares_ atau WCSS) terhadap berbagai nilai k.

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*WhTam5c_IzrdzDS-qZBO8A.png)

Saat nilai k bertambah, WCSS akan cenderung menurun karena setiap klaster menjadi lebih kecil dan lebih homogen. Namun, penurunan ini akan mulai melambat setelah titik tertentu. Titik di mana penurunan WCSS mulai tidak signifikan lagi inilah yang disebut _elbow_ — dan biasanya menjadi indikasi jumlah klaster optimal.

```
# Finding the optimal number of clusters using the Elbow Method
if CUSTOM_K is None:
    try:
        plt.figure(figsize=(10, 6))
        elbow = KElbowVisualizer(KMeans(random_state=RANDOM_STATE), k=(2, 15))
        elbow.fit(scaled_features)
        if SAVE_FIGURES:
            plt.savefig('elbow_method.png')
        plt.show()
        optimal_k = elbow.elbow_value_ if elbow.elbow_value_ is not None else 5
    except:
        # Fallback if visualization fails
        optimal_k = 5
        print("Could not visualize elbow method, using default k=5")
else:
    optimal_k = CUSTOM_K
    print(f"Using custom k value: {optimal_k}")
print(f"Optimal number of clusters: {optimal_k}")
```![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vDWZeMvEsEaG4V14kEbhxg.png)

Sebagai contoh kita mendapatkan salah satu jumlah kluster yang optimal yaitu **K = 10**.

Dalam konteks fine-tuning LLM dengan pendekatan diversity sampling, clustering digunakan untuk memilih subset data yang representatif berdasarkan distribusi semantik dan kategori. Menentukan jumlah klaster yang terlalu sedikit akan membuat data menjadi terlalu umum, sementara terlalu banyak klaster dapat menyebabkan model overfit terhadap noise atau variasi kecil yang tidak signifikan.

Dengan Elbow Method, kita dapat memperoleh keseimbangan antara kompleksitas model dan keanekaragaman data dengan cara yang relatif sederhana dan efisien.

**K-Means Clustering**

Setelah fitur teks dan kategorikal diekstraksi dan diskalakan, langkah selanjutnya adalah mengelompokkan data ke dalam beberapa klaster menggunakan algoritma K-Means. Ini penting dalam proses **diversity sampling**, agar kita bisa memilih data yang mewakili setiap karakteristik atau pola berbeda dari keseluruhan dataset.

```
# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)
# Add cluster labels to the original dataframe
data['cluster'] = cluster_labels
```

K-Means adalah algoritma unsupervised learning yang bertujuan membagi data ke dalam _k_ kelompok (clusters), di mana setiap data point dimasukkan ke dalam klaster dengan **centroid** terdekat. Proses ini dilakukan secara iteratif untuk meminimalkan jarak internal (dalam klaster) dan meningkatkan heterogenitas antar klaster.

Secara matematis, tujuan utama K-Means adalah meminimalkan fungsi objektif berikut:

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JbesV2iBsvVKOY5_JopEHg.png)

Visualisasi K — Means dapat dimuat ke dalam plot dua dimensi untuk mempermudah pemahaman :

```
# Visualize the clusters using PCA-reduced features
try:
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                         c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('K-means Clustering of Indonesian NLP Dataset')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('kmeans_clusters_visualization.png')
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")
    print("Could not create visualization. Continuing with analysis.")
```![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*_BaxvUJ5jzsIuwVJ1HcMHQ.png)

Dengan nilai K = 10 klusterisasi dilakukan dan semua data akan terkelompok ke dalam 10 kluster

```
# Analyze the clusters
print("\nCluster distribution:")
print(data['cluster'].value_counts())
# Let's see what characterizes each cluster
for cluster_id in range(optimal_k):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} samples):")
    # Top subset_names in this cluster
    print("Top subset_names:")
    print(cluster_data['subset_name'].value_counts().head(3))
    # Top template_names in this cluster
    print("Top template_names:")
    print(cluster_data['template_name'].value_counts().head(3))
    # Sample texts from this cluster
    print("Sample text:")
    try:
        sample_text = cluster_data[text_column].iloc[0]
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
    except:
        print("No text sample available")
    print("-" * 50)
```

Sebagai contoh hasil dari klustering akan menunjukkan sajian distribusi sebagai berikut :

```
Cluster distribution:
cluster
9    2128
0    2064
4    1822
7    1298
6    1030
1     643
8     556
5     279
3     109
2      71
Name: count, dtype: int64
Cluster 0 (2064 samples):
Top subset_names:
subset_name
wikipedia_id    2023
wikipedia_ms      20
dolly             13
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_section_2    249
wikipedia_section_0    245
wikipedia_section_4    239
Name: count, dtype: int64
Sample text:
Jika tahun tetap adalah  dari haid terakhir.
Haid terakhir tanggal 16-1-2016, maka Hari taksiran persalinan:
DD = 16 + 7 = 23
MM = 1 + 9 = 10
Jadi, HPL = 23 Oktober 2016
Jika tahun bertambah satu...
--------------------------------------------------
Cluster 1 (643 samples):
Top subset_names:
subset_name
wikipedia_id     339
wikipedia_ms     192
wikipedia_bjn     38
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_section_daerah_0    68
wikipedia_section_daerah_4    50
wikipedia_section_daerah_2    43
Name: count, dtype: int64
Sample text:
adalah sebuah stasiun kereta api di kota Mizuho, Prefektur Gifu, Jepang, yang dioperasikan oleh operator kereta api swasta Tarumi Railway.
--------------------------------------------------
Cluster 2 (71 samples):
Top subset_names:
subset_name
safety_prompt    71
Name: count, dtype: int64
Top template_names:
template_name
safety_prompt    71
Name: count, dtype: int64
Sample text:
Maaf, Cendol tidak bisa membuat teks yang berisikan hal-hal yang mdapat enyakiti orang lain. Cendol adalah AI bermartabat
--------------------------------------------------
Cluster 3 (109 samples):
Top subset_names:
subset_name
nusaparagraph_rhetoric_mak_nusantara_text    67
nusaparagraph_rhetoric_sun_nusantara_text    12
nusaparagraph_rhetoric_jav_nusantara_text     7
Name: count, dtype: int64
Top template_names:
template_name
generate_par_rhetoric_1    49
classification              6
retorika_qa_format          6
Name: count, dtype: int64
Sample text:
Awak dhewe kudu ngusahe supoyo ujian nasional kui ora diadake. Soale, akeh anak sing stres mergo ujian kui. Malah ono anak sing tekan depresi mergo wedi ora lulus utowo entuk biji sing elek neng ujian...
--------------------------------------------------
Cluster 4 (1822 samples):
Top subset_names:
subset_name
wikipedia_id    1612
indo_puisi        94
wikihow           74
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_subsection_0    279
wikipedia_subsection_3    240
wikipedia_subsection_4    232
Name: count, dtype: int64
Sample text:
Dua fakta tentang latar belakang keluarga Ælfgifu dinyatakan jelas oleh sumber. Pertama-tama, ibunya memiliki nama Æthelgifu, seorang wanita dari kelahiran tinggi (natione præcelsa). Kedua, ia berhubu...
--------------------------------------------------
Cluster 5 (279 samples):
Top subset_names:
subset_name
wikipedia_jv                                 264
nusaparagraph_rhetoric_jav_nusantara_text      9
wikipedia_map-bms                              3
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_section_daerah_0    36
wikipedia_section_daerah_1    34
wikipedia_section_daerah_3    33
Name: count, dtype: int64
Sample text:
Asil Tandhing Djarum ISL 2010 pada situs web resmi PT Liga Indonésia
Ulah raga ing taun 2010
Ulah raga ing taun 2011
--------------------------------------------------
Cluster 6 (1030 samples):
Top subset_names:
subset_name
wikipedia_min    671
wikipedia_id     359
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_section_daerah_1    141
wikipedia_section_daerah_4    137
wikipedia_section_daerah_0    127
Name: count, dtype: int64
Sample text:
Transvaalobrium sudrei adalah spesies kumbang tanduk panjang yang tergolong famili Cerambycidae. Spesies ini juga merupakan bagian dari genus Transvaalobrium, ordo Coleoptera, kelas Insecta, filum Art...
--------------------------------------------------
Cluster 7 (1298 samples):
Top subset_names:
subset_name
wikipedia_ms     1294
wikipedia_jv        2
wikipedia_min       1
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_subsection_daerah_1    144
wikipedia_subsection_daerah_3    142
wikipedia_subsection_daerah_0    138
Name: count, dtype: int64
Sample text:
Huawei Technologies Co. Ltd. (; ) merupakan syarikat multinasional China yang menghasilkan alat-alat rangkaian telekomunikasi yang berpusat di Shenzhen, Guangdong. Syarikat tersebut merupakan pembuat ...
--------------------------------------------------
Cluster 8 (556 samples):
Top subset_names:
subset_name
wikipedia_id       297
identity_prompt     67
wikipedia_ms        60
Name: count, dtype: int64
Top template_names:
template_name
identity_prompt        67
wikipedia_section_0    56
wikipedia_section_3    54
Name: count, dtype: int64
Sample text:
-  Sebelah Utara             : Desa Masukih (Kec. Miri Manasa )
-  Sebelah Timur             : Desa Lawang Tamang (Kec. M. Talawang)
-  Sebelah Selatan           : Desa Mampai Jaya (Kec. Kapuas hulu...
--------------------------------------------------
Cluster 9 (2128 samples):
Top subset_names:
subset_name
wikipedia_min    422
wikipedia_su     335
wikipedia_id     126
Name: count, dtype: int64
Top template_names:
template_name
wikipedia_section_daerah_3    106
wikipedia_section_daerah_2     95
classification                 90
Name: count, dtype: int64
Sample text:
tidak
--------------------------------------------------
```

Serta bisa di-analisis fitur kata/phrase per klusternya :

```
cluster_centers = kmeans.cluster_centers_
# Get the most important features (words) for each cluster
try:
    if hasattr(text_vectorizer, 'get_feature_names_out'):
        feature_names = text_vectorizer.get_feature_names_out()
    else:
        feature_names = text_vectorizer.get_feature_names()
    # Extract text features importance from cluster centers
    text_feature_importance = cluster_centers[:, :len(feature_names)]
    # Get top 5 important features for each cluster
    top_features_per_cluster = {}
    for i in range(optimal_k):
        top_indices = text_feature_importance[i].argsort()[-5:][::-1]
        top_features_per_cluster[i] = [feature_names[idx] for idx in top_indices]
    print("\nTop features (words/phrases) per cluster:")
    for cluster, features in top_features_per_cluster.items():
        print(f"Cluster {cluster}: {', '.join(features)}")
except Exception as e:
    print(f"Error during feature importance analysis: {e}")
    print("Could not extract top features per cluster.")
```

Hasil analisis ini bisa memberikan inferensi untuk kita membuat hipotesis pengelompokkan data, misal didapatkan hasil dari langkah di atas sebagai berikut :

```
Top features (words/phrases) per cluster:
Cluster 0: adalah, pada, tahun, film, seorang
Cluster 1: kota, daerah, wilayah, sekolah, timur
Cluster 2: bisa, ke, itu, yang, tidak
Cluster 3: bisa, tidak, anak, dua, daerah
Cluster 4: yang, untuk, dengan, bahwa, karena
Cluster 5: ing, lan, bisa, tanggal, para
Cluster 6: ordo, famili, bagian dari, kingdom, spesies
Cluster 7: daripada, beliau, kepada, telah, dan
Cluster 8: kecamatan, kabupaten, desa, indonesia, timur
Cluster 9: asteroid, tidak, ko, nan, salah
```

Maka kita bisa membuat hipotesis kategorisasi data

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*5ia-wljMkhMgoxsCkF6etA.png)

Kita juga bisa melakukan analisis mendetail

```
# Generate cluster-specific information
print("\nDetailed Cluster Analysis:")
for cluster_id in range(optimal_k):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"\n== CLUSTER {cluster_id} SUMMARY ==")
    print(f"Number of samples: {len(cluster_data)}")
    # Distribution by subset
    print("\nSubset distribution:")
    subset_dist = cluster_data['subset_name'].value_counts()
    for subset, count in subset_dist.items():
        percentage = (count / len(cluster_data)) * 100
        print(f"- {subset}: {count} samples ({percentage:.1f}%)")
    # Try to characterize the cluster
    if 'indo_puisi' in subset_dist and subset_dist['indo_puisi'] / len(cluster_data) > 0.5:
        print("This appears to be primarily a poetry cluster.")
    elif 'wikipedia_id' in subset_dist and subset_dist['wikipedia_id'] / len(cluster_data) > 0.5:
        print("This appears to be primarily a Wikipedia content cluster.")
    elif 'wikihow' in subset_dist and subset_dist['wikihow'] / len(cluster_data) > 0.5:
        print("This appears to be primarily a how-to/instructional content cluster.")
    # Print separator
    print("-" * 50)
``````
Detailed Cluster Analysis:
== CLUSTER 0 SUMMARY ==
Number of samples: 2064
Subset distribution:
- wikipedia_id: 2023 samples (98.0%)
- wikipedia_ms: 20 samples (1.0%)
- dolly: 13 samples (0.6%)
- wikihow: 3 samples (0.1%)
- wikipedia_min: 2 samples (0.1%)
- wikipedia_gor: 1 samples (0.0%)
- wikipedia_tet: 1 samples (0.0%)
- wikipedia_map-bms: 1 samples (0.0%)
This appears to be primarily a Wikipedia content cluster.
--------------------------------------------------
== CLUSTER 1 SUMMARY ==
Number of samples: 643
Subset distribution:
- wikipedia_id: 339 samples (52.7%)
- wikipedia_ms: 192 samples (29.9%)
- wikipedia_bjn: 38 samples (5.9%)
- wikipedia_su: 24 samples (3.7%)
- wikipedia_ban: 22 samples (3.4%)
- wikipedia_min: 12 samples (1.9%)
- dolly: 4 samples (0.6%)
- wikipedia_jv: 2 samples (0.3%)
- nusaparagraph_topic_bew_nusantara_text: 1 samples (0.2%)
- wikipedia_ace: 1 samples (0.2%)
- nusaparagraph_topic_btk_nusantara_text: 1 samples (0.2%)
- nusaparagraph_topic_rej_nusantara_text: 1 samples (0.2%)
- wikipedia_gor: 1 samples (0.2%)
- nusaparagraph_rhetoric_jav_nusantara_text: 1 samples (0.2%)
- nusaparagraph_topic_mak_nusantara_text: 1 samples (0.2%)
- nusaparagraph_topic_mad_nusantara_text: 1 samples (0.2%)
- nusaparagraph_topic_mui_nusantara_text: 1 samples (0.2%)
- nusaparagraph_topic_jav_nusantara_text: 1 samples (0.2%)
This appears to be primarily a Wikipedia content cluster.
--------------------------------------------------
== CLUSTER 2 SUMMARY ==
Number of samples: 71
Subset distribution:
- safety_prompt: 71 samples (100.0%)
--------------------------------------------------
== CLUSTER 3 SUMMARY ==
Number of samples: 109
Subset distribution:
- nusaparagraph_rhetoric_mak_nusantara_text: 67 samples (61.5%)
- nusaparagraph_rhetoric_sun_nusantara_text: 12 samples (11.0%)
- nusaparagraph_rhetoric_jav_nusantara_text: 7 samples (6.4%)
- nusaparagraph_rhetoric_bew_nusantara_text: 7 samples (6.4%)
- nusaparagraph_rhetoric_min_nusantara_text: 5 samples (4.6%)
- nusaparagraph_rhetoric_btk_nusantara_text: 3 samples (2.8%)
- nusaparagraph_rhetoric_rej_nusantara_text: 2 samples (1.8%)
- nusaparagraph_rhetoric_bug_nusantara_text: 2 samples (1.8%)
- nusaparagraph_rhetoric_mad_nusantara_text: 2 samples (1.8%)
- nusaparagraph_rhetoric_mui_nusantara_text: 2 samples (1.8%)
--------------------------------------------------
== CLUSTER 4 SUMMARY ==
Number of samples: 1822
Subset distribution:
- wikipedia_id: 1612 samples (88.5%)
- indo_puisi: 94 samples (5.2%)
- wikihow: 74 samples (4.1%)
- dolly: 25 samples (1.4%)
- nusaparagraph_rhetoric_bew_nusantara_text: 5 samples (0.3%)
- nusaparagraph_emot_bew_nusantara_text: 4 samples (0.2%)
- wikipedia_ms: 2 samples (0.1%)
- nusaparagraph_rhetoric_mui_nusantara_text: 2 samples (0.1%)
- wikipedia_su: 2 samples (0.1%)
- wikipedia_map-bms: 1 samples (0.1%)
- nusaparagraph_topic_bew_nusantara_text: 1 samples (0.1%)
This appears to be primarily a Wikipedia content cluster.
--------------------------------------------------
== CLUSTER 5 SUMMARY ==
Number of samples: 279
Subset distribution:
- wikipedia_jv: 264 samples (94.6%)
- nusaparagraph_rhetoric_jav_nusantara_text: 9 samples (3.2%)
- wikipedia_map-bms: 3 samples (1.1%)
- nusaparagraph_topic_jav_nusantara_text: 2 samples (0.7%)
- wikipedia_id: 1 samples (0.4%)
--------------------------------------------------
== CLUSTER 6 SUMMARY ==
Number of samples: 1030
Subset distribution:
- wikipedia_min: 671 samples (65.1%)
- wikipedia_id: 359 samples (34.9%)
--------------------------------------------------
== CLUSTER 7 SUMMARY ==
Number of samples: 1298
Subset distribution:
- wikipedia_ms: 1294 samples (99.7%)
- wikipedia_jv: 2 samples (0.2%)
- wikipedia_min: 1 samples (0.1%)
- wikipedia_id: 1 samples (0.1%)
--------------------------------------------------
== CLUSTER 8 SUMMARY ==
Number of samples: 556
Subset distribution:
- wikipedia_id: 297 samples (53.4%)
- identity_prompt: 67 samples (12.1%)
- wikipedia_ms: 60 samples (10.8%)
- wikipedia_gor: 46 samples (8.3%)
- wikipedia_map-bms: 37 samples (6.7%)
- wikipedia_ban: 24 samples (4.3%)
- wikipedia_min: 15 samples (2.7%)
- wikipedia_ace: 6 samples (1.1%)
- wikipedia_nia: 1 samples (0.2%)
- wikipedia_bjn: 1 samples (0.2%)
- wikipedia_su: 1 samples (0.2%)
- nusaparagraph_emot_btk_nusantara_text: 1 samples (0.2%)
This appears to be primarily a Wikipedia content cluster.
--------------------------------------------------
== CLUSTER 9 SUMMARY ==
Number of samples: 2128
Subset distribution:
- wikipedia_min: 422 samples (19.8%)
- wikipedia_su: 335 samples (15.7%)
- wikipedia_id: 126 samples (5.9%)
- nusaparagraph_rhetoric_sun_nusantara_text: 121 samples (5.7%)
- nusaparagraph_rhetoric_jav_nusantara_text: 119 samples (5.6%)
- nusaparagraph_rhetoric_bew_nusantara_text: 107 samples (5.0%)
- nusaparagraph_rhetoric_min_nusantara_text: 103 samples (4.8%)
- nusaparagraph_rhetoric_mad_nusantara_text: 65 samples (3.1%)
- wikipedia_ms: 56 samples (2.6%)
- nusaparagraph_emot_jav_nusantara_text: 47 samples (2.2%)
- nusaparagraph_topic_sun_nusantara_text: 46 samples (2.2%)
- nusaparagraph_emot_sun_nusantara_text: 44 samples (2.1%)
- nusaparagraph_topic_bew_nusantara_text: 40 samples (1.9%)
- nusaparagraph_rhetoric_btk_nusantara_text: 37 samples (1.7%)
- nusaparagraph_emot_mak_nusantara_text: 34 samples (1.6%)
- nusaparagraph_topic_min_nusantara_text: 34 samples (1.6%)
- nusaparagraph_topic_mak_nusantara_text: 30 samples (1.4%)
- nusaparagraph_topic_mad_nusantara_text: 29 samples (1.4%)
- nusaparagraph_emot_min_nusantara_text: 27 samples (1.3%)
- indo_puisi: 27 samples (1.3%)
- nusaparagraph_topic_jav_nusantara_text: 27 samples (1.3%)
- nusaparagraph_emot_bew_nusantara_text: 27 samples (1.3%)
- wikipedia_ban: 25 samples (1.2%)
- wikipedia_jv: 21 samples (1.0%)
- nusaparagraph_topic_btk_nusantara_text: 21 samples (1.0%)
- nusaparagraph_rhetoric_bug_nusantara_text: 21 samples (1.0%)
- nusaparagraph_emot_btk_nusantara_text: 20 samples (0.9%)
- nusaparagraph_emot_mad_nusantara_text: 19 samples (0.9%)
- nusaparagraph_rhetoric_rej_nusantara_text: 15 samples (0.7%)
- nusaparagraph_rhetoric_mui_nusantara_text: 13 samples (0.6%)
- dolly: 12 samples (0.6%)
- nusaparagraph_emot_rej_nusantara_text: 10 samples (0.5%)
- nusaparagraph_topic_rej_nusantara_text: 7 samples (0.3%)
- nusaparagraph_emot_mui_nusantara_text: 7 samples (0.3%)
- wikipedia_ace: 6 samples (0.3%)
- wikipedia_tet: 6 samples (0.3%)
- wikipedia_gor: 5 samples (0.2%)
- wikipedia_nia: 5 samples (0.2%)
- nusaparagraph_topic_bug_nusantara_text: 4 samples (0.2%)
- nusaparagraph_emot_bug_nusantara_text: 4 samples (0.2%)
- nusaparagraph_topic_mui_nusantara_text: 3 samples (0.1%)
- wikipedia_map-bms: 1 samples (0.0%)
--------------------------------------------------
```

1.  4. Save Dataset

Sampel yang sudah diperoleh dari hasil klustering akan kita simpan

```
# Save the clustered data as parquet
try:
    data.to_parquet('kmeans_clustered_data.parquet', index=False)
    print("\nClustered data saved to 'kmeans_clustered_data.parquet'")
except Exception as e:
    print(f"Error saving Parquet file: {e}")
```

Anda bisa mengunduh salah satu dataset (.**parquet**) yang sudah selesai di pra-proses dan disimpan ke open source cloud based file storage, _catbox :_ [_https://files.catbox.moe/s33mcj.parquet_](https://files.catbox.moe/s33mcj.parquet)

**2. YA YA YA INI DIA BIANG SETRESSNYA — Model Training**

![captionless image](https://miro.medium.com/v2/resize:fit:1124/format:webp/1*4PzA1lbnzJe4k06H7V154A.png)

Nah, di tahap ini kita akan melatih model menggunakan **LoRA (Low-Rank Adaptation)**, yang merupakan metode **Parameter-Efficient Fine-Tuning (PEFT)**. Ini tuh kayak cara pintar buat melatih model besar tanpa makan banyak sumber daya. LoRA memungkinkan kita untuk hanya melakukan fine-tuning pada sebagian kecil parameter dari model dasar (base model), jadi hemat banget, gak perlu ngabisin banyak memori atau waktu komputasi.

Kita bakal menggunakan kombinasi antara **transformers** dari Hugging Face dan **peft** buat ngejalanin fine-tuning ini. Model yang kita pakai nanti bakal lebih ringan, tapi tetap powerful.

Transformer (bukan Robot)
=========================

![captionless image](https://miro.medium.com/v2/resize:fit:980/format:webp/1*F5pP-ZEBdHY1rzzeFVIQ0A.png)

**Transformer** adalah arsitektur yang pertama kali diperkenalkan oleh Vaswani et al. dalam paper legendaris _“Attention is All You Need”_ (2017). Arsitektur ini menggantikan mekanisme sekuensial pada RNN dengan _self-attention_, memungkinkan model untuk menangkap dependensi jarak jauh dalam data secara lebih efisien.

> _📚 Referensi:
> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need._ Advances in neural information processing systems_, 30._

PEFT (dengan LoRA)
==================

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*K10qWoCvtw8aYVof.png)

PEFT, singkatan dari **Parameter-Efficient Fine-Tuning**, adalah pendekatan untuk melakukan fine-tuning model besar (seperti LLMs) tanpa harus melakukan update terhadap seluruh parameter model. Ini sangat relevan dalam era model bahasa besar (LLMs) seperti GPT, LLaMA, atau OLMo, yang terdiri dari miliaran parameter dan memerlukan sumber daya komputasi yang besar untuk dilatih ulang secara penuh.

Bayangkan kamu punya model sebesar **1.3 miliar parameter**, dan kamu hanya ingin melatihnya untuk tugas tertentu — misalnya, membuat model chatbot Bahasa Indonesia. Tanpa PEFT, kamu harus update semua parameter model, memakan waktu, memori, dan risiko _overfitting_. PEFT memungkinkan kita melakukan fine-tuning hanya pada bagian-bagian kecil dari model, tapi tetap mendapatkan performa yang sangat kompetitif.

Salah satu teknik PEFT yang populer adalah **LoRA (Low-Rank Adaptation)**. LoRA menambahkan _adapter layer_ ke dalam model pre-trained yang sudah ada, dan selama proses fine-tuning, hanya adapter-nya yang dilatih. Struktur asli model tetap utuh, sehingga:

*   **Parameter yang dilatih jauh lebih sedikit**
*   **Hasil fine-tuning lebih stabil dan efisien**
*   **Model tetap bisa digunakan ulang di task lain**

LoRA (Low-Rank Adaptation)
==========================

![captionless image](https://miro.medium.com/v2/resize:fit:1200/format:webp/0*tZZe2r6W93V_WIch.png)

**LoRA** adalah metode fine-tuning parameter yang diperkenalkan oleh Hu et al. (2021) sebagai solusi terhadap masalah besar: model besar seperti GPT atau BERT memiliki milyaran parameter, dan fine-tuning semua parameter untuk task baru memakan sumber daya besar.

Gagasan Utama LoRA:
===================

Daripada meng-update semua bobot, LoRA menambahkan _low-rank matrices_ yang dilatih, sementara parameter asli dibekukan. Matematika sederhananya:

Jika `W` adalah bobot asli, maka LoRA mengganti:

```
W x → (W + ΔW) x  dengan ΔW = A @ B
```

Di mana `A ∈ ℝ^(d×r)`, `B ∈ ℝ^(r×k)`, dan `r` adalah _rank_ yang jauh lebih kecil (misalnya 8). Ini membuat jumlah parameter yang perlu dilatih sangat kecil.

> _📚 Referensi:
> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021)._ LoRA: Low-Rank Adaptation of Large Language Models_. arXiv preprint arXiv:2106.09685._

Wes jangan terlalu pusing bahas Arsitektur, sebelum mulai training kita lakukan beberapa persiapan dulu

2. 1. Training Preparation

Download dan load dataset hasil pre-processing

```
!wget https://files.catbox.moe/s33mcj.parquet -o kmeans.parquet
``````
import pandas as pd
from datasets import load_dataset
dataset = load_dataset('parquet',
                       data_files='kmeans.parquet',
                       split = 'train')
```![captionless image](https://miro.medium.com/v2/resize:fit:912/format:webp/1*D_X9LqTfs4E-IutT-vXDYg.png)

Kemudian lakukan conversational formatting agar dataset bisa dimuat ke dalam arsitektur transformer :

```
def pre_process_conversation(examples):
    conversations = [        [            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        for input_text, output_text in zip(examples["input"], examples["output"])
    ]
    return {"conversations": conversations}
def check_valid_conversations(examples):
    return [all(msg["content"] is not None for msg in conv)
            for conv in examples["conversations"]]
dataset = dataset.map(
    pre_process_conversation,
    batched=True,
    batch_size=1000,
    num_proc=10,
    remove_columns=dataset.column_names
)
```

Langkah di atas perlu dilakukan untuk mempersiapkan dataset percakapan agar sesuai dengan format yang dibutuhkan untuk model training. Fungsi pertama, `**pre_process_conversation**`, mengubah data yang memiliki kolom `input` dan `output` menjadi format yang lebih terstruktur, dengan setiap percakapan dibagi menjadi dua bagian: satu untuk peran **user** dan satu lagi untuk peran **assistant**.

Setiap pasangan `input_text` dan `output_text` dari dataset diubah menjadi dua entri dalam percakapan, masing-masing berisi informasi tentang peran dan isi pesan. Fungsi kedua, `**check_valid_conversations**`, berfungsi untuk memverifikasi bahwa setiap percakapan yang telah diproses memiliki konten yang valid (yakni tidak ada nilai `None` pada kolom `content`).

Terakhir, `**dataset.map()**` digunakan untuk memetakan fungsi `**pre_process_conversation**` ke seluruh dataset secara batch, dengan pemrosesan paralel menggunakan 10 proses untuk efisiensi. Setelah pemrosesan, kolom `input` dan `output` yang lama dihapus, menyisakan format data yang lebih siap untuk digunakan dalam pelatihan model.

Formatting percakapan sangat penting dalam konteks pelatihan model bahasa, terutama untuk model berbasis dialog seperti GPT atau model lainnya yang mengandalkan struktur percakapan untuk memahami konteks. Dalam model seperti ini, penting untuk memastikan bahwa data percakapan dikemas dengan cara yang dapat dipahami oleh model, dengan memisahkan peran dalam dialog seperti **user** dan **assistant**. Tanpa format yang tepat, model mungkin akan kesulitan dalam membedakan peran dan konteks dalam percakapan, yang dapat mengurangi kemampuan model dalam memberikan respons yang relevan dan tepat.

Dengan memformat percakapan ke dalam struktur

`{role: user, content: input_text}`

`{role: assistant, content: output_text}`

kita memberikan konteks yang jelas mengenai siapa yang berbicara dan apa yang mereka katakan. Ini memungkinkan model untuk belajar dalam konteks dialog yang lebih realistis dan sesuai dengan interaksi manusia. Selain itu, format yang jelas dan konsisten juga memudahkan model dalam memahami pola percakapan dan merespons berdasarkan peran yang berbeda (seperti pengguna atau asisten). Oleh karena itu, proses formatting ini adalah langkah krusial dalam persiapan data agar model dapat belajar dengan baik dan menghasilkan output yang lebih akurat dalam tugas percakapan.

Sekarang kita sudah siap melakukan training dengan dataset tadi

![captionless image](https://miro.medium.com/v2/resize:fit:652/format:webp/1*MXR3T51yaxPabA0n-n3fwQ.png)

2.2. Config Parameters

```
class Config:
  def __init__(self):
    self.LORA_R = 8
    self.LORA_ALPHA = 32
    self.LORA_DROPOUT = 0.05
    self.LEARNING_RATE = 2e-4
    self.BATCH_SIZE = 8
    self.EPOCHS = 3
    self.CUTOFF_LEN = 512
    self.OUTPUT_DIR = "olmo-lora"
# Kenapa saya pakai nilai parameternya segitu? karena saya suka angka 
# angka 8, 32, 0.05 dan 2e-4,8, 3 dan 512 :)
# intinya LORA_ALPHA DAN LEARNING RATE mutually koheren, 
      # jadi keduany sama - sama mempengaruhi kualitas hasil training, 
      # semakin kecil semakin spesifik trainingnya dan prosesnya semakin lama
      # saya pilih yang ideal menurut rekomendasi teman saya @FaizOnichan 
      # berdasarkan pengalaman eksperimen pada final project mata kuliah ML sem lalu     
```

Kelas `**Config**` berfungsi untuk menyimpan konfigurasi atau pengaturan yang digunakan selama pelatihan model. Setiap atribut dalam kelas ini menentukan parameter-parameter yang penting dalam pelatihan model, seperti pengaturan hyperparameter dan konfigurasi output. Berikut penjelasan dari masing-masing atribut:

1.  `**LORA_R**`: Nilai ini mengatur dimensi dari rank dalam teknik LoRA (Low-Rank Adaptation). LoRA digunakan untuk fine-tuning model dengan lebih efisien dan ringan, dan `LORA_R` mengontrol kompleksitas rank tersebut.
2.  `**LORA_ALPHA**`: Merupakan faktor penskalaan yang digunakan dalam LoRA untuk menyesuaikan bobot selama fine-tuning. Ini mempengaruhi seberapa besar pengaruh adaptasi LoRA terhadap model.
3.  `**LORA_DROPOUT**`: Mengatur tingkat dropout yang digunakan selama pelatihan untuk mencegah overfitting. Dropout adalah teknik untuk "menghilangkan" beberapa neuron sementara selama pelatihan.
4.  `**LEARNING_RATE**`: Menentukan kecepatan model dalam memperbarui bobotnya selama proses pelatihan. Nilai `2e-4` berarti model akan melakukan pembaruan dengan laju yang relatif lambat, yang cocok untuk fine-tuning model besar.
5.  `**BATCH_SIZE**`: Menentukan jumlah data yang diproses dalam satu kali iterasi selama pelatihan. Dalam hal ini, `8` berarti setiap batch terdiri dari 8 contoh data.
6.  `**EPOCHS**`: Mengindikasikan berapa kali seluruh dataset akan diproses oleh model selama pelatihan. `3` berarti model akan melalui seluruh dataset tiga kali.
7.  `**CUTOFF_LEN**`: Menentukan panjang maksimum dari input yang diproses oleh model. Input yang lebih panjang dari nilai ini akan dipotong menjadi 512 token. Ini membantu untuk memastikan bahwa input model tidak terlalu besar dan lebih mudah diproses.
8.  `**OUTPUT_DIR**`: Menentukan direktori tempat model yang telah dilatih akan disimpan setelah proses pelatihan selesai. Dalam hal ini, model akan disimpan di folder bernama `olmo-lora`.

Secara keseluruhan, kelas `**Config**` ini mengatur semua parameter penting yang digunakan dalam proses fine-tuning dan pelatihan model, sehingga memudahkan dalam pengelolaan dan reproduksi eksperimen.

Selanjutnya kita atur untuk konfigurasi LORA

```
config_params = Config()
config = LoraConfig(
    r=config_params.LORA_R,
    lora_alpha=config_params.LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Adjust these for OLMo
    lora_dropout=config_params.LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

Di bagian pertama, `**config_params = Config()**`, sebuah objek `**Config**` dibuat berdasarkan kelas `**Config**` yang telah dijelaskan sebelumnya. Objek `**config_params**` ini akan digunakan untuk mengambil berbagai nilai pengaturan yang telah ditentukan dalam kelas `**Config**` untuk diintegrasikan dengan konfigurasi LoRA selanjutnya.

Kemudian, `**config = LoraConfig(...)**` digunakan untuk mengonfigurasi parameter-parameter terkait dengan Low-Rank Adaptation (LoRA), yang merupakan teknik untuk fine-tuning model dengan cara yang lebih efisien dan mengurangi beban komputasi. Parameter-parameter yang ditentukan dalam `**LoraConfig**` adalah sebagai berikut:

1.  `**r=config_params.LORA_R**`: Nilai `**LORA_R**` yang diambil dari objek `**config_params**` digunakan untuk menentukan dimensi rank dalam LoRA. Rank ini mengontrol seberapa besar ukuran matrix adaptasi yang akan ditambahkan pada model selama fine-tuning.
2.  `**lora_alpha=config_params.LORA_ALPHA**`: Nilai `**LORA_ALPHA**` mengontrol penskalaan pengaruh LoRA terhadap bobot model yang sudah ada. Semakin besar nilai alpha, semakin besar pengaruh LoRA terhadap hasil akhirnya.
3.  `**target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]**`: Ini adalah daftar modul target dalam model yang akan menggunakan LoRA. Dalam konteks model transformer (seperti GPT atau model sejenis), modul-modul ini merujuk pada bagian-bagian yang terlibat dalam perhitungan query (q), key (k), value (v), dan output (o) yang menjadi bagian dari perhatian (attention). Dengan memilih modul-modul ini, LoRA hanya akan diterapkan pada bagian-bagian yang relevan dari model.
4.  `**lora_dropout=config_params.LORA_DROPOUT**`: Nilai `**LORA_DROPOUT**` menentukan tingkat dropout yang diterapkan pada parameter LoRA untuk mencegah overfitting. Dropout akan membuat beberapa parameter LoRA dihilangkan secara acak selama pelatihan.
5.  `**bias="none"**`: Menentukan bahwa tidak akan ada bias yang digunakan dalam penyesuaian LoRA. Bias biasanya digunakan untuk menyesuaikan output model, tetapi dalam LoRA, bias ini bisa ditiadakan untuk meningkatkan efisiensi.
6.  `**task_type=TaskType.CAUSAL_LM**`: Menetapkan tipe tugas untuk model yang dilatih. Dalam hal ini, `**TaskType.CAUSAL_LM**` menunjukkan bahwa model diharapkan untuk melakukan tugas _causal language modeling_, di mana model memprediksi kata berikutnya dalam urutan berdasarkan konteks sebelumnya, sesuai dengan cara kerja model-model seperti GPT.

Secara keseluruhan, kode ini mengonfigurasi dan menyiapkan parameter-parameter LoRA untuk fine-tuning model transformer besar, dengan penyesuaian di beberapa bagian penting dari model untuk menghemat sumber daya komputasi tanpa mengurangi performa model secara signifikan.

2. 3. Load LLM & Tokenizer

```
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
tokenizer.pad_token = tokenizer.eos_token
olmo_model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
```![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bqf_vEIHcksvbd5Wejt-xg.png)```
OlmoForCausalLM(
  (model): OlmoModel(
    (embed_tokens): Embedding(50304, 2048, padding_idx=1)
    (layers): ModuleList(
      (0-15): 16 x OlmoDecoderLayer(
        (self_attn): OlmoAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): OlmoMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): OlmoLayerNorm()
        (post_attention_layernorm): OlmoLayerNorm()
      )
    )
    (norm): OlmoLayerNorm()
    (rotary_emb): OlmoRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=50304, bias=False)
```

Sedikit funfact kenapa dipilih **OLMo-1B** untuk tutorial kali ini, karena model ini mempunyai aksesibilitas untuk merubah banyak bagian dari arsitektur dan hyper parameternya.

Jangan lupa untuk gunakan cuda selama proses training dengan menjalankan sintaks `**olmo_model.to("cuda")**`

Sekarang kita coba untuk menjalankan model yang sudah di-load

```
message = ["What's a Bird?", "What a bird can does?"]
inputs = tokenizer(message,
                   return_tensors='pt',
                   return_token_type_ids=False,
                   padding=True)
inputs = {k: v.to('cuda') for k,v in inputs.items()}
response = olmo_model.generate(**inputs,
                               max_new_tokens=100,
                               do_sample=True,
                               top_k=50,
                               top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Dan hasilnya pun di luar nalar

```
What's a Bird?Miniature Pinscher Miniature Pinscher. When you hear the name Miniature Pinscher, you expect an adorable and furry little creature with a soft face and large ears. What you may not know is that they are actually very smart dogs and have a lot to offer owners. The Miniature Pinscher is the smallest of the working dogs (i.e. Mastiffs and German Shepherds are the largest) and is known to be one of the friend
```

Ya, mungkin bisa kita maklumi karena parameter yang sedikit, model ini terkategori sebagai _Toy Model_ yang pemanfaatannya tidak terlalu luas.

2. 4. Conversation Formatting

Dataset akan diformat ulang, bertujuan untuk mengubah struktur data percakapan (conversation) yang berbentuk list of dictionary menjadi sebuah teks naratif utuh yang dapat digunakan sebagai input model bahasa atau keperluan pelatihan.

```
def format_conversation(example):
    conversation = example['conversations']
    formatted_text = ""
    for message in conversation:
        if message['role'] == 'user':
            formatted_text += f"User: {message['content']}\n"
        else:
            formatted_text += f"Assistant: {message['content']}\n"
    return formatted_text.strip()
``````
train_dataset = dataset.map(
        lambda x: {'formatted_text': format_conversation(x)},
        remove_columns=dataset.column_names
)
```

2. 5. Training Arguments & Metric Evaluator

`TrainingArguments` digunakan untuk menyetel berbagai parameter pelatihan.

```
from transformers import Trainer, TrainingArguments
from transformers import Trainer, TrainingArguments
import evaluate
# Load metrik dari evaluate
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
# compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Untuk BLEU, bentuknya: list of tokens
    bleu_preds = [pred.strip().split() for pred in decoded_preds]
    bleu_labels = [[label.strip().split()] for label in decoded_labels]
    bleu = bleu_metric.compute(predictions=bleu_preds, references=bleu_labels)
    # Untuk ROUGE, langsung pakai string
    rouge = rouge_metric.compute(
        predictions=[pred.strip() for pred in decoded_preds],
        references=[label.strip() for label in decoded_labels],
        use_stemmer=True
    )
    return {
        "bleu": bleu["bleu"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rougeLsum": rouge["rougeLsum"],
    }
# TrainingArguments tetap
training_args = TrainingArguments(
    output_dir=config_params.OUTPUT_DIR,
    num_train_epochs=config_params.EPOCHS,
    per_device_train_batch_size=config_params.BATCH_SIZE,
    save_steps=50,
    logging_steps=10,
    eval_steps=100,
    learning_rate=config_params.LEARNING_RATE,
    fp16=False,
    optim="adamw_torch",
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    dataloader_num_workers=11,
    report_to="wandb",
)

```

Pada tahap ini, kita mulai mempersiapkan proses pelatihan model bahasa berbasis _transformer_ dengan pendekatan yang terstruktur dan terukur. Dua komponen utama yang digunakan adalah `Trainer` dan `TrainingArguments` dari library `transformers` milik HuggingFace. Keduanya berfungsi sebagai fondasi dalam manajemen eksperimen pelatihan model.

Untuk mengukur performa model selama pelatihan, digunakan dua metrik utama dari library `evaluate`, yaitu:

*   **BLEU (Bilingual Evaluation Understudy)**: Cocok digunakan untuk mengevaluasi keluaran model yang bersifat _translation-like_ atau dalam format sekuensial yang pendek. BLEU membandingkan kesamaan n-gram antara prediksi model dan referensi.
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Digunakan untuk mengukur kualitas ringkasan (summary) atau teks yang lebih panjang. ROUGE mengukur tumpang tindih antara prediksi dan referensi dari segi kata atau frasa.

Fungsi `compute_metrics` di atas melakukan decoding hasil prediksi dan label yang masih berupa token ID menjadi teks asli, lalu menghitung metrik berdasarkan teks tersebut. Untuk BLEU, teks diubah ke bentuk token-list, sementara ROUGE bisa langsung menggunakan string.

Namun dalam eksperimen kali ini, belum berfokus kepada robustness metric sehingga belum melakukan evaluasi metrik ROUGE dan BLEU.

2. 6. Running The Training

Sebelum kita mulai training jangan lupa untuk melakukan tokenisasi dataset

```
# Preprocess dataset (dengan tokenizer)
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["formatted_text"],
        truncation=True,
        max_length=config_params.CUTOFF_LEN,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # ← tambahkan ini
    return tokenized
tokenized_dataset = train_dataset.map(preprocess_function, remove_columns=["formatted_text"])
```

Agar model dapat belajar dengan benar, input perlu dipersiapkan dalam format yang dapat diproses. Di sinilah fungsi `preprocess_function` berperan. Fungsi ini akan melakukan tokenisasi terhadap teks hasil format percakapan, memotong (truncation) hingga panjang maksimum (`max_length`), dan menambahkan padding. Kemudian, `labels` disalin dari `input_ids`, karena model bersifat _causal language modeling_ (memprediksi token berikutnya berdasarkan token sebelumnya).

Dengan demikian, proses pelatihan menjadi sepenuhnya terotomatisasi dan terintegrasi, dimulai dari preprocessing, training, sampai evaluasi berbasis metrik yang relevan. Seluruh proses ini memastikan bahwa fine-tuning dapat dilakukan dengan cara yang terstruktur dan dapat direplikasi.

Siapkan trainer

```
trainer = Trainer(
    model=train_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
``````
TrainingArguments(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
average_tokens_across_devices=False,
batch_eval_metrics=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=11,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
dispatch_batches=None,
do_eval=False,
do_predict=False,
do_train=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_steps=100,
eval_strategy=IntervalStrategy.NO,
eval_use_gather_object=False,
evaluation_strategy=None,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=True,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=0.0002,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=olmo-lora/runs/Apr14_13-06-25_c06d475bfaea,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=10,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_kwargs={},
lr_scheduler_type=SchedulerType.COSINE,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=3,
optim=OptimizerNames.ADAMW_TORCH,
optim_args=None,
optim_target_modules=None,
output_dir=olmo-lora,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=8,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['wandb'],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=olmo-lora,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=50,
save_strategy=SaveStrategy.STEPS,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
split_batches=None,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=None,
torchdynamo=None,
tp_size=0,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
warmup_ratio=0.03,
warmup_steps=0,
weight_decay=0.0,
)
```

Kita juga dapat melihat trainable-params dari trainer

```
train_model.print_trainable_parameters()
```![captionless image](https://miro.medium.com/v2/resize:fit:1182/format:webp/1*yM6AnwZfDI9SigxqKSW8Vw.png)

3 … 2 … 1 … Mulailah Melatih

Pada saat proses pelatihan berjalan, kita bisa melihat nilai los setiap stepnya

```

trainer.train()
``````
Step Training Loss
10 6.971000
20 6.339900
30 2.763700
40 1.541000
50 1.481300
60 1.535600
70 1.451200
80 1.273200
90 1.264800
100 1.433900
110 1.192800
120 1.150800
130 1.494400
140 1.514200
150 1.361900
160 1.392100
170 1.344300
180 1.138800
190 1.330400
200 1.350400
210 1.443100
220 1.023400
230 1.084300
240 1.270000
250 1.323800
260 1.316200
270 1.087700
280 1.202300
290 1.085200
300 1.265600
310 1.112000
320 1.159000
330 1.159900
340 1.213300
350 1.074500
360 1.135000
370 1.211500
380 1.133200
390 1.279000
400 1.234300
410 1.087300
420 0.999400
430 1.086500
440 1.034400
450 1.067900
460 1.164000
470 1.143000
480 1.173400
490 1.209600
500 1.129200
510 1.381100
520 1.216100
530 1.000800
540 1.159800
550 1.216400
560 1.268400
570 1.146900
580 1.086700
590 1.113200
600 1.172500
610 1.181500
620 1.100600
630 1.247600
640 1.203200
650 1.144400
660 1.104600
670 1.100900
680 1.185500
690 1.201000
700 1.122100
710 1.369800
720 1.099600
730 1.299800
740 1.137800
750 1.121500
760 1.046200
770 1.262000
780 1.040700
790 1.018900
800 0.986100
810 1.276500
820 1.189700
830 1.078800
840 1.059500
850 1.155100
860 0.996200
870 0.960400
880 1.097400
890 1.145700
900 1.236700
910 1.221100
920 0.965000
930 1.051100
940 1.224200
950 1.091700
960 1.086400
970 1.094700
980 1.014800
990 1.296000
1000 1.289400
1010 1.234500
1020 1.193100
1030 1.245200
1040 1.101300
1050 1.214800
1060 1.231200
1070 1.271400
1080 1.532300
1090 1.133100
1100 1.122700
1110 0.942600
1120 1.202500
1130 1.126400
1140 1.448700
1150 1.282300
1160 1.066700
1170 1.074000
1180 1.045400
1190 1.009000
1200 1.183300
1210 1.085600
1220 1.163400
1230 1.271300
1240 1.360500
1250 1.129700
1260 1.157300
1270 1.029500
1280 1.142900
1290 1.180500
1300 1.060500
1310 1.082500
1320 1.053200
1330 1.315500
1340 1.251800
1350 0.851900
1360 1.115900
1370 0.907500
1380 1.091700
1390 1.215000
1400 1.144100
1410 1.112800
1420 1.080400
1430 1.008500
1440 1.184100
1450 1.150300
1460 1.148300
1470 0.921700
1480 1.134600
1490 1.268200
1500 1.121500
1510 0.986900
1520 1.120200
1530 1.198700
1540 1.007800
1550 1.074600
1560 1.067400
1570 1.176500
1580 1.102700
1590 1.012600
1600 0.991200
1610 1.237800
1620 1.196100
1630 1.000400
1640 1.126200
1650 1.072600
1660 1.123800
1670 1.211900
1680 0.874000
1690 1.298900
1700 1.089100
1710 1.156400
1720 1.031300
1730 1.046900
1740 1.139000
1750 1.159500
1760 1.168300
1770 1.115900
1780 1.338000
1790 1.021100
1800 1.130600
1810 1.198500
1820 1.198900
1830 1.173000
1840 1.199800
1850 1.041500
1860 1.022300
1870 1.250000
1880 1.185900
1890 1.145000
1900 0.901700
1910 1.172600
1920 1.147500
1930 1.097800
1940 1.144300
1950 1.084900
1960 1.037400
1970 1.095800
1980 1.272900
1990 0.893700
2000 1.087000
2010 0.844400
2020 0.952700
2030 1.141100
2040 1.012300
2050 1.008100
2060 1.108300
2070 1.095500
2080 1.062100
2090 0.913000
2100 1.011700
2110 1.143500
2120 1.107800
2130 1.129600
2140 1.105300
2150 1.310000
2160 1.036100
2170 1.136600
2180 1.100400
2190 1.305500
2200 0.982900
2210 1.064300
2220 1.083900
2230 1.167400
2240 1.105300
2250 1.027800
2260 1.163000
```

2. 7. Save Model

Jika Training sudah selesai, simpan model yang sudah dihasilkan

```
trainer.save_model(f'{config_params.OUTPUT_DIR}-final-model')
train_model.save_pretrained(f'{config_params.OUTPUT_DIR}-final-model')
```

Kita juga memanfaatkan catbox untuk menyimpan model secara online

```
import os
import zipfile
import requests
def upload_to_catbox(file_path):
    url = 'https://catbox.moe/user/api.php'
    with open(file_path, 'rb') as f:
        response = requests.post(url, data={'reqtype': 'fileupload'}, files={'fileToUpload': f})
    if response.status_code == 200:
        return response.text.strip()
    else:
        raise Exception(f"Gagal upload {file_path}. Status: {response.status_code}, Respon: {response.text}")
def zip_directory(source_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=source_dir)
                zipf.write(full_path, arcname)
def upload_zipped_model(base_output_dir):
    final_model_dir = f"{base_output_dir}-final-model"
    zip_filename = f"{final_model_dir}.zip"
    if not os.path.exists(final_model_dir):
        raise FileNotFoundError(f"Direktori {final_model_dir} tidak ditemukan.")
    print(f"📦 Membuat zip dari direktori: {final_model_dir} -> {zip_filename}")
    zip_directory(final_model_dir, zip_filename)
    print(f"☁️ Mengupload zip ke Catbox: {zip_filename}")
    try:
        url = upload_to_catbox(zip_filename)
        print(f"✅ Upload berhasil! Link: {url}")
        return zip_filename, url
    except Exception as e:
        print(f"❌ Upload gagal: {e}")
        return zip_filename, None
# Jalankan
zip_name, link = upload_zipped_model(f'{config_params.OUTPUT_DIR}')
# Print hasil akhir
print("\n=== Hasil Upload ===")
if link:
    print(f"{zip_name}: {link}")
else:
    print("Tidak ada file yang berhasil diupload.")
```

Anda dapat membuka contoh model yang sudah jadi di tautan berikut :

[https://files.catbox.moe/eda06l.zip](https://files.catbox.moe/eda06l.zip)

2. 8. Inference Testing

```
import torch
from transformers import AutoTokenizer
import time
import sys
class OLMoPEFTInference:
    def __init__(self, model, tokenizer_name="allenai/OLMo-1B-hf"):
        """
        Inisialisasi untuk inference dengan model hasil get_peft_model.
        """
        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Jaga-jaga biar gak error padding
        # Model hasil training (get_peft_model)
        self.model = model
        self.model.eval()  # Mode eval wajib
        self.device = next(self.model.parameters()).device
    def _prepare_input(self, user_text, history=None):
        """
        Format input jadi prompt. History bisa ditambahkan untuk chat mode.
        """
        prompt = ""
        if history:
            for turn in history:
                role, content = turn["role"], turn["content"]
                prompt += f"{role.capitalize()}: {content}\n"
        prompt += f"User: {user_text}\nAssistant:"
        return prompt
    def generate(self, user_input, history=None, stream_output=True):
        """
        Generate output dari prompt atau percakapan.
        """
        formatted = self._prepare_input(user_input, history)
        # Tokenisasi prompt
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        # Generate dengan model
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=300,
                # temperature=0.3,
                # top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # Decode hasil
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.split("Assistant:")[-1].strip()
        if stream_output:
            self._stream_response(response)
        else:
            print(response)
        return response
    def _stream_response(self, response):
        """
        Print response secara streaming seperti chatbot beneran.
        """
        for char in response:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.015)
        print("\n")
# =====================================
# ✅ Contoh penggunaan
# =====================================
if __name__ == "__main__":
    from peft import get_peft_model, LoraConfig
    from transformers import AutoModelForCausalLM
    # Load base dan adapter model
    base = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-1B-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Misal kita sudah punya config LoRA-nya (bisa dari PeftConfig atau manual)
    config = LoraConfig.from_pretrained("olmo-lora/checkpoint-1550")  # Ganti path sesuai model kamu
    # Inisialisasi inferencer
    infer = OLMoPEFTInference(train_model)
    # Jalankan inference (mode chat)
    chat = [        {"role": "user", "content": "Halo siapa kamu?"},
        {"role": "assistant", "content": "Saya Cendol, model bahasa yang cerdas."}
    ]
    infer.generate("Apa itu sayuran?")#, dauistory=chat)
```

Kita akan menguji coba hasil model dengan mencoba untuk melakukan percakapan dan memberikan promp sederhana _“apa itu sayuran?”_

Dan diperoleh response dari model sudah bisa menggunakan bahasa Indonesia

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*tMpogtckItyNG4WH_hmPnA.png)

Walaupun mungkin model ini belum Robust karena merupakan fine tune menggunakan _Toy Model._

Metrik evaluasi yang dihasilkan dari pelatihan tersaji melalui reporting third party yaitu _wandb_

```
from google.colab import userdata
import wandb
wandb.login(key=userdata.get("WANDB_KEY"))
wandb.init(project="fp-kcv", name=f"fp-kcv-wak-report")
```

Sebelumnya telah dilakukan banyak uji coba dalam pelatihan model dan diperoleh sajian hasil sebagai berikut

**Training Report**

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*RUN_XQlQdJfNXdlLIfDaqA.png)

**Resources Report (Big RAM GPU A100 + T4 via Google Colab)**

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9yCZTLWmD_JyLE3J8X_FFQ.png)
