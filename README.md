# Morse Code to Speech Berbasis Deteksi Gerakan Tangan

![Computer Vision](https://github.com/mharits-s/morse-object-detection/blob/main/assets/computervision.svg?raw=true)

## Overview
Morse Code to Speech Berbasis Deteksi Gerakan Tangan adalah proyek inovatif yang menggabungkan visi komputer, pembelajaran mesin, dan pengenalan gestur untuk mengonversi gerakan tangan menjadi kode Morse dan selanjutnya menjadi kata-kata yang diucapkan. Sistem ini dirancang untuk mengenali berbagai jenis gestur tangan dan menerjemahkannya ke dalam kode Morse, memberikan cara komunikasi yang unik dan interaktif.

## Cara Menggunakan
Untuk menggunakan sistem Morse Code to Speech, ikuti langkah-langkah berikut:

1. **Pengumpulan Dataset:**
    - Jalankan program `collect.py` untuk mengumpulkan dataset yang beragam dari gestur tangan.
    - Tekan tombol "s" untuk memulai merekam citra tangan untuk setiap gestur.
    - Dataset dikategorikan menjadi kelas seperti Dit, Dash, Enter, New, Next, dan Space.

2. **Pelatihan Model:**
    - Latih model pengenalan gestur tangan menggunakan platform "Teachable Machine".
    - Konfigurasikan pelatihan dengan 50 epoch, setiap batch berisi 16 citra, dan tingkat pembelajaran sebesar 0,001.
    - Model mempelajari untuk mengenali berbagai gestur tangan dari dataset yang dikumpulkan.

3. **Pengenalan Gestur:**
    - Implementasikan model yang sudah dilatih dalam `test_updated.py`.
    - Sistem mengenali gestur tangan secara real-time menggunakan visi komputer.
    - Gestur yang terdeteksi dikonversi menjadi kode Morse dan ditampilkan di layar.

4. **Output Suara:**
    - Pola kode Morse secara otomatis dikonversi menjadi kata-kata yang diucapkan menggunakan teknologi Text-to-Speech (TTS).
    - String yang didekode kemudian diputar sebagai output yang dapat didengar.

## Sample Output

![Sample](https://github.com/mharits-s/morse-object-detection/blob/main/assets/sample.gif?raw=true)

Dalam output contoh, sistem mengenali gestur seperti Dit, Dash, Next, dan New, memperlihatkan potensi dari Morse Code to Speech Berbasis Deteksi Gerakan Tangan.

## Dependensi
Pastikan Anda telah menginstal dependensi berikut:
- OpenCV
- [CVZone](https://github.com/cvzone/cvzone)
- Mediapipe
- NumPy
- TensorFlow
- gtts (Google Text-to-Speech)

## Peningkatan di Masa Depan
Proyek ini dapat ditingkatkan dengan beberapa cara:
- Menambahkan lebih banyak gestur untuk pengalaman komunikasi yang lebih kaya.
- Mengimplementasikan antarmuka yang ramah pengguna untuk meningkatkan kegunaan.
- Mengoptimalkan model untuk kinerja real-time.

Silakan berkontribusi dan jelajahi kemungkinan dari Morse Code to Speech Berbasis Deteksi Gerakan Tangan!
