# E-commerce Public Dataset Dashboard ✨

Dashboard interaktif berbasis Streamlit untuk menganalisis Brazilian E-Commerce Public Dataset (Olist).

## Setup Environment - Shell/Terminal

Pastikan Python sudah terinstal di komputer.

```bash
# Masuk ke folder proyek
cd E-commerce-public-dataset-dashboard

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment (Windows)
venv\Scripts\activate

# Install library yang dibutuhkan
pip install -r requirements.txt
```

## Struktur Folder

```text
E-commerce-public-dataset-dashboard/
├── app.py
├── requirements.txt
├── README.md
└── data
    ├── orders.csv
    ├── order_items.csv
    ├── order_reviews.csv
    ├── products.csv
    ├── customers.csv
    └── category_translation.csv
```

## Run Streamlit App

```bash
streamlit run app.py
```

Dashboard dapat diakses melalui browser pada alamat:

```text
http://localhost:8501
```

## Analisis Dashboard

1. Overview Dashboard
2. Sales Performance
3. Delivery & Reviews
4. Product Performance
5. Customer Segmentation (RFM)

## Catatan Metodologi

- Analisis penjualan menggunakan pesanan berstatus `delivered`.
- Total sales dan Monetary dihitung berdasarkan kolom `price`, tidak termasuk `freight_value`.
- Jumlah pesanan dihitung berdasarkan `order_id` unik.
- Review digabung menjadi rata-rata satu skor per `order_id`.
- RFM menggunakan `customer_unique_id`.
- Analisis pengiriman hanya menggunakan pesanan dengan tanggal aktual, tanggal estimasi, dan review yang tersedia.

## Dependencies

- Streamlit
- Pandas
- NumPy
- Plotly
