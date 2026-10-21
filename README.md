<div align="center">

# 🍩 The Simpsons Character Classifier

**End-to-End Image Classification Pipeline with Deep Learning, Transfer Learning, and Multi-Platform Deployment (SavedModel, TFLite, & TF.js)**

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Architecture](https://img.shields.io/badge/Backbone-MobileNetV2-blue?logo=google&logoColor=white)](https://arxiv.org/abs/1801.04381)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.40%25-brightgreen)](#-hasil-evaluasi--performa)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br/>

*Proyek Machine Learning klasifikasi gambar karakter kartun The Simpsons menggunakan arsitektur MobileNetV2 dengan teknik Transfer Learning & Fine-Tuning, dilengkapi ekspor model ke format SavedModel, TensorFlow Lite (TFLite), dan TensorFlow.js (TFJS).*

---

[📖 Ringkasan Proyek](#-ringkasan-proyek) •
[✨ Fitur Utama](#-fitur-utama) •
[📊 Dataset](#-informasi-dataset) •
[🧠 Arsitektur Model](#-arsitektur-dan-pelatihan-model) •
[📈 Hasil Evaluasi](#-hasil-evaluasi--performa) •
[📁 Struktur Direktori](#-struktur-direktori) •
[🚀 Cara Menjalankan](#-panduan-instalasi--penggunaan) •
[📦 Format Model](#-deployment--format-model) •
[👨‍💻 Profil Pembuat](#-author)

</div>

---

## 📖 Ringkasan Proyek

Proyek ini bertujuan untuk mengidentifikasi dan mengklasifikasikan 10 karakter ikonik dari serial animasi legendaris **The Simpsons** berdasarkan input gambar. 

Dengan memanfaatkan metode **Transfer Learning** menggunakan arsitektur **MobileNetV2** yang telah di-pra-latih pada ImageNet, model dilatih dengan teknik fine-tuning parsial (membekukan layer awal dan membuka 30+ layer teratas) agar mampu mengekstrak fitur visual kartun yang spesifik. 

Model mencapai **akurasi validasi >95%** dan **akurasi pengujian sebesar 94.40%**. Model akhir diekspor ke dalam tiga format industri standar untuk memudahkan integrasi pada platform server, aplikasi mobile (Android/iOS), dan peramban web (browser-based AI).

---

## ✨ Fitur Utama

- **Transfer Learning & Fine-Tuning**: Menggunakan backbone MobileNetV2 dengan penyesuaian bobot (unfreeze layer 120+) untuk konvergensi cepat dan akurasi tinggi.
- **Data Augmentation Komprehensif**: Mencegah overfitting dengan variasi rotasi, pergeseran lebar/tinggi, shear, zoom, dan horizontal flip menggunakan `ImageDataGenerator`.
- **Early Stopping Callback Otomatis**: Dilengkapi custom callback `AkurasiTinggiCallback` yang menghentikan iterasi latihan secara elegan ketika akurasi training dan validasi telah melampaui **95%**.
- **Multi-Format Export Ready**:
  - 🖥️ **TensorFlow SavedModel**: Siap di-deploy pada backend Python, Flask/FastAPI, atau TF Serving.
  - 📱 **TensorFlow Lite (`.tflite`)**: Model terkompresi ringan (~10 MB) untuk perangkat mobile dan IoT/Edge.
  - 🌐 **TensorFlow.js (`tfjs_model`)**: Siap dijalankan langsung di sisi client (browser) tanpa server backend.
- **Inference Pipeline**: Dilengkapi pengujian prediksi acak pada set gambar uji langsung dari model SavedModel yang telah diekspor.

---

## 📊 Informasi Dataset

Dataset yang digunakan berasal dari **The Simpsons Characters Dataset** yang dipublikasikan oleh Alexandre Attia di Kaggle.

- **Sumber Dataset**: [Kaggle - The Simpsons Characters Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset)
- **Total Karakter Terpilih**: 10 Karakter Ikonik
- **Rasio Pembagian Data**: 80% Train, 10% Validation, 10% Test

| No | Nama Karakter / Kelas | Train | Validation | Test | Total |
|:--:|:----------------------|:-----:|:----------:|:----:|:-----:|
| 1  | `bart_simpson`        |  -    |     -      |  -   |   ~   |
| 2  | `disco_stu`           |  -    |     -      |  -   |   ~   |
| 3  | `homer_simpson`       |  -    |     -      |  -   |   ~   |
| 4  | `krusty_the_clown`    |  -    |     -      |  -   |   ~   |
| 5  | `lisa_simpson`        |  -    |     -      |  -   |   ~   |
| 6  | `marge_simpson`       |  -    |     -      |  -   |   ~   |
| 7  | `moe_szyslak`         |  -    |     -      |  -   |   ~   |
| 8  | `ned_flanders`        |  -    |     -      |  -   |   ~   |
| 9  | `nelson_muntz`        |  -    |     -      |  -   |   ~   |
| 10 | `ralph_wiggum`        |  -    |     -      |  -   |   ~   |
| **Total** | **10 Kelas**   | **8,635** | **1,075** | **1,090** | **~10,800** |

---

## 🧠 Arsitektur dan Pelatihan Model

### Arsitektur Jaringan
```
Input Image (224 x 224 x 3)
         │
         ▼
MobileNetV2 (Pretrained ImageNet)
[Layer 0 s/d 119: Frozen | Layer 120+: Trainable]
         │
         ▼
GlobalAveragePooling2D
         │
         ▼
Dropout (Rate: 0.3)
         │
         ▼
Dense Layer (256 unit, ReLU activation)
         │
         ▼
Dense Output (10 unit, Softmax activation)
```

### Konfigurasi Hyperparameter
- **Input Dimension**: `(224, 224, 3)`
- **Batch Size**: `32`
- **Optimizer**: `RMSprop(learning_rate=0.0001)`
- **Loss Function**: `Categorical Crossentropy`
- **Metric**: `Accuracy`
- **Epochs Maksimum**: `20` (berhenti di Epoch 11 karena kriteria callback terpenuhi)

---

## 📈 Hasil Evaluasi & Performa

Pelatihan berhenti pada **Epoch 11** setelah model memenuhi kriteria akurasi di atas 95%:

| Metrik Evaluasi | Hasil Performa |
|:----------------|:--------------:|
| **Training Accuracy** | **98.91%** |
| **Training Loss**     | `0.0340`   |
| **Validation Accuracy** | **95.35%** |
| **Validation Loss**     | `0.2555`   |
| **Test Set Accuracy**   | **94.40%** |
| **Test Set Loss**       | `0.3476`   |

> Model menunjukkan kemampuan generalisasi yang sangat kuat tanpa indikasi overfitting berlebihan pada gambar pengujian di luar dataset latihan.

---

## 📁 Struktur Direktori

```text
├── saved_model/                 # Ekspor model standar TensorFlow SavedModel
│   ├── assets/
│   ├── variables/
│   │   ├── variables.data-00000-of-00001
│   │   └── variables.index
│   ├── fingerprint.pb
│   └── saved_model.pb
├── tfjs_model/                  # Ekspor model untuk deployment web via TensorFlow.js
│   ├── group1-shard1of3.bin
│   ├── group1-shard2of3.bin
│   ├── group1-shard3of3.bin
│   └── model.json
├── tflite/                      # Ekspor model untuk mobile/edge via TensorFlow Lite
│   ├── label.txt                # Daftar 10 kelas target
│   └── model.tflite             # Bobot terkompresi TFLite (~10 MB)
├── notebook.ipynb               # Jupyter Notebook alur lengkap (Prep, Train, Eval, Export)
├── requirements.txt             # Dependensi pustaka Python
└── README.md                    # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/<username>/simpsons-character-classifier.git
cd simpsons-character-classifier
```

### 2. Buat Lingkungan Virtual (Virtual Environment)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Notebook
Jalankan Jupyter Notebook untuk melihat alur kerja lengkap atau melatih kembali model:
```bash
jupyter notebook notebook.ipynb
```

---

## 🧪 Contoh Inferensi Menggunakan Python

Anda dapat menguji prediksi citra baru secara instan menggunakan `saved_model` yang telah tersedia:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Muat SavedModel
model = tf.saved_model.load("saved_model")
predict_fn = model.signatures["serving_default"]

# 2. Baca daftar label
with open("tflite/label.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 3. Muat dan preprocess gambar uji
img_path = "path/to/test_image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# 4. Prediksi
tensor_input = tf.constant(img_array, dtype=tf.float32)
predictions = predict_fn(tensor_input)
output_key = list(predictions.keys())[0]
probabilities = predictions[output_key].numpy()[0]

predicted_idx = np.argmax(probabilities)
confidence = probabilities[predicted_idx] * 100

print(f"Prediksi Karakter: {labels[predicted_idx]} ({confidence:.2f}%)")
```

---

## 📦 Deployment & Format Model

| Format | Direktori | Target Penggunaan | Cara Integrasi |
|:-------|:----------|:------------------|:---------------|
| **SavedModel** | `saved_model/` | Server / Backend / Cloud | `tf.saved_model.load("saved_model")` atau TF Serving Docker |
| **TensorFlow Lite** | `tflite/model.tflite` | Android / iOS / IoT (Raspberry Pi) | TensorFlow Lite Interpreter (`tflite_runtime`) |
| **TensorFlow.js** | `tfjs_model/model.json` | Web Browser / Node.js | `await tf.loadGraphModel('tfjs_model/model.json')` |

---

## 👨‍💻 Author

- **Nama**: Muhammad Bazwa Arigusna
- **Email**: [arigusnawawa11@gmail.com](mailto:arigusnawawa11@gmail.com)
- **ID Dicoding**: [bazwa_arigusna](https://www.dicoding.com/users/bazwa_arigusna)

---

## 📜 Lisensi & Atribusi

Proyek ini dibuat untuk keperluan portofolio Machine Learning dan pemenuhan kriteria submission Dicoding Academy.
Karakter *The Simpsons* merupakan hak cipta dari **Matt Groening** dan **20th Television / The Walt Disney Company**. Dataset disediakan oleh Alexandre Attia untuk tujuan penelitian dan edukasi non-komersial.
