# Deviasi

### PERUBAHAN API PUSTAKA (scikit-learn >= 1.2)

Pada file asli main.py, model NMF dipanggil menggunakan parameter NMF(..., alpha=0.0). Namun, pada scikit-learn versi 1.8.0, parameter alpha telah dihapus sehingga memicu TypeError: NMF.__init__() got an unexpected keyword argument 'alpha'.

Solusi:
Diimplementasikan mekanisme fallback adaptif menggunakan parameter alpha_W=0.0 pada notebook agar tetap kompatibel dengan berbagai versi scikit-learn.


### PERBAIKAN BUG VARIABEL GLOBAL PADA REPOSITORI ASLI

Pada fungsi Denoising(training_data, test_data, ...) di main.py, iterasi loop menggunakan variabel global training alih-alih parameter lokal training_data. Akibatnya, fungsi tidak dapat berjalan dengan benar ketika pengguna mengganti daftar file data latih.

Solusi:
Bug diperbaiki pada notebook sehingga fungsi mendukung pergantian data latih secara dinamis, baik White Noise maupun Air Noise.


### INKONSISTENSI EVALUASI PADA performance.py

Pada blok demo __main__ di performance.py, kode asli mengevaluasi pasangan sinyal yang tidak konsisten, yaitu voice_nmf dipasangkan dengan noise_onmf.

Solusi:
Evaluasi pada notebook replikasi dipisahkan berdasarkan pasangan model yang sesuai, yaitu:
- voice_nmf dengan noise_nmf.
- voice_onmf dengan noise_onmf.


### PENYELARASAN DIMENSI SINYAL (SIGNAL ALIGNMENT)

Sinyal asli yang terpotong atau terkontaminasi memiliki panjang 935.526 sampel, sedangkan hasil rekonstruksi iSTFT menghasilkan 935.936 sampel karena penyesuaian terhadap kelipatan hop size STFT.

Solusi:
Fungsi evaluasi pada notebook secara otomatis menyelaraskan panjang kedua sinyal dengan memotongnya berdasarkan panjang minimum (min(len(synth), len(orig))) sebelum proses proyeksi QR dilakukan. Hal ini mencegah ValueError: shapes not aligned.


### STABILITAS NUMERIK DAN PENANGANAN RANDOM SEED

Fungsi update_code_within_radius pada onmf.py rentan menghasilkan RuntimeWarning: invalid value encountered in scalar divide ketika norm matriks kode mendekati nol.

Solusi:
Ditambahkan faktor stabilisasi numerik epsilon (ε = 10^-12) untuk menghindari pembagian dengan nilai yang terlalu kecil.

Selain itu, kode asli tidak menggunakan seed deterministik (seed=None), sehingga menghasilkan sedikit variasi acak selama proses sampling patch.

Solusi:
Digunakan seed=42 untuk memastikan hasil replikasi lebih konsisten dan mendekati hasil yang dilaporkan dalam paper, dengan deviasi sebagai berikut:
- SDR/SAR pada White Noise: < 0.35 dB.
- Seluruh metrik pada Air Noise: < 0.05 dB.