# E-commerce Public Dataset Dashboard ✨

Dashboard interaktif berbasis Streamlit untuk menganalisis **Brazilian E-Commerce Public Dataset (Olist)**.

Dashboard ini bertujuan memberikan informasi mengenai performa penjualan, ketepatan pengiriman, ulasan pelanggan, performa kategori produk, serta segmentasi pelanggan menggunakan RFM Analysis.

## Struktur Folder

```text
submission/
├── dashboard/
│   ├── dashboard.py
│   └── main_data.csv
├── data/
│   ├── category_translation.csv
│   ├── customers.csv
│   ├── geolocation.csv
│   ├── order_items.csv
│   ├── order_payments.csv
│   ├── order_reviews.csv
│   ├── orders.csv
│   ├── products.csv
│   └── sellers.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt
```

## Setup Environment

Pastikan Python sudah terinstal pada komputer.

```bash
# Masuk ke folder proyek
cd submission

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Streamlit App

Jalankan perintah berikut dari folder utama `submission\dashboard`:

```bash
streamlit run dashboard/dashboard.py
```

Dashboard dapat diakses melalui browser pada:

```text
http://localhost:8503
```

## Analisis Dashboard

1. Overview Dashboard
2. Sales Performance
3. Delivery & Reviews
4. Product Performance
5. Customer Segmentation (RFM)

## Catatan Metodologi

- Analisis penjualan menggunakan pesanan berstatus `delivered`.
- Total Sales dan Monetary dihitung berdasarkan kolom `price`, tidak termasuk `freight_value`.
- Jumlah pesanan dihitung berdasarkan `order_id` unik.
- Review digabung menjadi rata-rata satu skor per `order_id`.
- RFM menggunakan `customer_unique_id`.
- Analisis pengiriman hanya menggunakan pesanan dengan tanggal aktual, tanggal estimasi, dan review yang tersedia.
- Analisis dilakukan berdasarkan periode September 2016 hingga Oktober 2018.

## Dependencies

- Streamlit
- Pandas
- NumPy
- Plotly

Lihat `requirements.txt` untuk daftar library dan versinya.