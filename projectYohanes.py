# 1. Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# cek versi library
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"seaborn version: {sns.__version__}")

# Atur gaya visualisasi
sns.set_theme(style="whitegrid")

# 2. Membaca dataset
file_path = 'DATASET/ispu_dki_all.csv'
df_raw = pd.read_csv(file_path)

# 3. Menampilkan lima baris pertama
print("=== 5 Baris Pertama Dataset ===")
print(df_raw.head())

# 4. Memeriksa ukuran dataset
print(f"\nUkuran dataset: {df_raw.shape[0]} baris dan {df_raw.shape[1]} kolom.")

# 5. Menampilkan nama kolom
print("\n=== Daftar Kolom ===")
print(df_raw.columns.tolist())

print("="*65)
print("Analisa Tipe Data, Missing Value, dan Duplikat\n")

# 6. Memeriksa tipe data setiap kolom
print("=== Informasi Tipe Data ===")
df_raw.info()

# 7. Menghitung statistik deskriptif awal
print("\n=== Statistik Deskriptif ===")
print(df_raw.describe())

# 8. Menghitung missing values (data kosong)
print("\n=== Jumlah Data Kosong (Missing Values) ===")
print(df_raw.isnull().sum())

# 9. Memeriksa duplikasi data
jumlah_duplikat = df_raw.duplicated().sum()
print(f"\nJumlah baris duplikat: {jumlah_duplikat}")

print("="*65)
print("Pembentukan Target Biner dan Pencegahan Leakage\n")

# 10. Memeriksa kategori unik yang ada di data asli sebelum diubah
print("=== Nilai Unik pada Kolom 'category' ===")
print(df_raw['categori'].value_counts(dropna=False))

# 11. Membuat kolom target biner
# Kelas 0: Kualitas baik/sedang (Acceptable)
# Kelas 1: Kualitas tidak sehat/sangat tidak sehat/berbahaya (Unhealthy)
kategori_sehat = ['Baik', 'Sedang', 'BAIK', 'SEDANG', 'baik', 'sedang'] # Menangani variasi huruf kapital

# Gunakan np.where: jika isi 'category' ada di list kategori_sehat, beri nilai 0, selain itu 1
# (Catatan: kita perlu berhati-hati jika ada data 'category' yang kosong/NaN)
df_raw['target'] = np.where(df_raw['categori'].isin(kategori_sehat), 0, 1)

# Jika category kosong (NaN), kita bisa set targetnya menjadi NaN juga untuk sementara
df_raw.loc[df_raw['categori'].isnull(), 'target'] = np.nan

# Memeriksa distribusi target baru
print("\n=== Distribusi Target Biner (0 = Baik/Sedang, 1 = Tidak Sehat) ===")
print(df_raw['target'].value_counts(dropna=False))

# 12. Menghapus kolom penyebab LEAKAGE (kebocoran informasi) dan identitas non-prediktif
# Sesuaikan nama kolom dengan output df_raw.columns dari Bagian A
kolom_leakage = ['max', 'critical', 'categori']
kolom_identitas = ['tanggal', 'stasiun'] # Biasanya ini tidak dipakai untuk prediksi KNN

# Kita buat dataframe baru untuk analisis (df_clean sementara)
kolom_drop = [k for k in kolom_leakage + kolom_identitas if k in df_raw.columns]
df_eda = df_raw.drop(columns=kolom_drop)

print(f"\nKolom yang tersisa untuk fitur prediktor: {df_eda.columns.tolist()}")