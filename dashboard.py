from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Olist E-Commerce Dashboard", page_icon="🛒", layout="wide")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
START_DATE = pd.Timestamp("2016-09-01")
END_DATE = pd.Timestamp("2018-11-01")  # batas akhir eksklusif: sampai Oktober 2018

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid #e8edf4;
        min-width: 0;
    }

    [data-testid="stMetricValue"] {
        font-size: 20px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricValue"] div {
        overflow: visible !important;
        text-overflow: clip !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Dashboard Analisis E-Commerce Olist")
st.caption("Analisis penjualan, ketepatan pengiriman, kategori produk, dan segmentasi pelanggan")

# Nama file pada folder data disesuaikan dengan struktur folder milik pengguna.
FILE_NAMES = {
    "orders": "orders.csv",
    "items": "order_items.csv",
    "reviews": "order_reviews.csv",
    "products": "products.csv",
    "translation": "category_translation.csv",
    "customers": "customers.csv",
}

@st.cache_data(show_spinner="Memuat dan mengolah dataset...")
def load_data(data_dir_string):
    folder = Path(data_dir_string)
    data = {}
    for key, filename in FILE_NAMES.items():
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        data[key] = pd.read_csv(path)

    orders = data["orders"].copy()
    items = data["items"].copy()
    reviews = data["reviews"].copy()
    products = data["products"].copy()
    translation = data["translation"].copy()
    customers = data["customers"].copy()

    # Konversi tanggal mengikuti tahapan analisis pada notebook.
    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_columns:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Mengambil pesanan delivered pada periode September 2016-Oktober 2018.
    delivered = orders.loc[
        (orders["order_status"] == "delivered")
        & (orders["order_purchase_timestamp"] >= START_DATE)
        & (orders["order_purchase_timestamp"] < END_DATE)
    ].copy()

    # 1. Penjualan bulanan: nilai penjualan dari kolom price.
    sales_data = delivered.merge(items, on="order_id", how="inner")
    sales_data["month"] = sales_data["order_purchase_timestamp"].dt.to_period("M")
    monthly = sales_data.groupby("month").agg(
        total_sales=("price", "sum"),
        total_orders=("order_id", "nunique"),
    )
    all_months = pd.period_range("2016-09", "2018-10", freq="M")
    monthly = monthly.reindex(all_months, fill_value=0)
    monthly.index.name = "month"
    monthly = monthly.reset_index()
    monthly["month"] = monthly["month"].dt.to_timestamp()
    monthly["sales_change"] = monthly["total_sales"].diff()
    previous_sales = monthly["total_sales"].shift(1)
    monthly["sales_growth"] = np.where(
        previous_sales > 0,
        (monthly["total_sales"] - previous_sales) / previous_sales * 100,
        np.nan,
    )

    # 2. Ketepatan pengiriman dan review score.
    reviews_clean = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )
    delivery = delivered.dropna(subset=[
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ]).merge(reviews_clean, on="order_id", how="inner")
    delivery["delivery_difference"] = (
        delivery["order_delivered_customer_date"]
        - delivery["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    delivery["delivery_status"] = np.where(
        delivery["delivery_difference"] <= 0, "Tepat Waktu", "Terlambat"
    )
    review_comparison = delivery.groupby("delivery_status")["review_score"].mean()
    delivery_counts = delivery.groupby("delivery_status")["order_id"].nunique()

    # 3. Kontribusi penjualan dan jumlah pesanan per kategori produk.
    product_sales = (
        items.merge(delivered[["order_id"]], on="order_id", how="inner")
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(translation, on="product_category_name", how="left")
    )
    product_sales["product_category_name_english"] = product_sales[
        "product_category_name_english"
    ].fillna("Unknown")
    category_sales = product_sales.groupby("product_category_name_english").agg(
        total_sales=("price", "sum"),
        total_orders=("order_id", "nunique"),
        total_items=("order_item_id", "count"),
    ).reset_index().rename(columns={
        "product_category_name_english": "category"
    }).sort_values("total_sales", ascending=False)

    # 4. RFM mengikuti rumus dan batas skor pada notebook.
    rfm_orders = orders.loc[orders["order_status"] == "delivered"].copy()
    rfm_data = (
        rfm_orders.merge(items, on="order_id", how="inner")
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
    )
    rfm_data["order_purchase_timestamp"] = pd.to_datetime(
        rfm_data["order_purchase_timestamp"], errors="coerce"
    )
    snapshot_date = rfm_data["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = rfm_data.groupby("customer_unique_id").agg(
        Recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("price", "sum"),
    ).reset_index()

    if len(rfm) >= 4:
        rfm["R_Score"] = pd.qcut(
            rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]
        ).astype(int)
        rfm["F_Score"] = pd.qcut(
            rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
        ).astype(int)
        rfm["M_Score"] = pd.qcut(
            rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]
        ).astype(int)
        rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

        def segment_customer(score):
            if score >= 10:
                return "Champions"
            if score >= 8:
                return "Loyal Customers"
            if score >= 6:
                return "Potential Customers"
            if score >= 4:
                return "At Risk"
            return "Others"

        rfm["Customer_Segment"] = rfm["RFM_Score"].apply(segment_customer)
    else:
        rfm["Customer_Segment"] = "Belum dapat dikelompokkan"

    return delivered, sales_data, monthly, delivery, review_comparison, delivery_counts, category_sales, rfm

try:
    delivered, sales_data, monthly, delivery, review_comparison, delivery_counts, category_sales, rfm = load_data(str(DATA_DIR))
except Exception as error:
    st.error(f"Gagal memuat dataset: {error}")
    st.info("Pastikan folder data berada sejajar dengan folder dashboard dan nama file CSV sesuai struktur folder yang kamu kirim.")
    st.stop()

# Sidebar: filter tanggal hanya untuk tiga analisis EDA; RFM tetap mengikuti seluruh data delivered.
st.sidebar.header("Pengaturan Dashboard")
st.sidebar.caption("Periode analisis sesuai notebook: September 2016–Oktober 2018")
top_n = st.sidebar.slider("Jumlah kategori produk", min_value=5, max_value=20, value=10)

# KPI utama.
total_sales = sales_data["price"].sum()
total_orders = delivered["order_id"].nunique()
total_customers = delivered["customer_id"].nunique()
avg_order_value = total_sales / total_orders if total_orders else 0

st.subheader("Ringkasan Kinerja")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Penjualan",
    f"BRL {total_sales:,.0f}"
)

c2.metric(
    "Total Pesanan",
    f"{total_orders:,.0f}"
)

c3.metric(
    "Pelanggan",
    f"{total_customers:,.0f}"
)

c4.metric(
    "Rata-rata Nilai Pesanan",
    f"BRL {avg_order_value:,.0f}"
)

st.caption("Data penjualan menggunakan pesanan berstatus delivered dan kolom price, tidak termasuk freight_value.")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Penjualan Bulanan", "🚚 Pengiriman & Ulasan", "📦 Kategori Produk", "👥 Segmentasi RFM"
])

with tab1:
    st.subheader("Perkembangan Penjualan dan Jumlah Pesanan Bulanan")
    left, right = st.columns(2)
    with left:
        fig = px.line(monthly, x="month", y="total_sales", markers=True,
                      title="Total Penjualan per Bulan",
                      labels={"month": "Bulan", "total_sales": "Penjualan (BRL)"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(monthly, x="month", y="total_orders",
                     title="Jumlah Pesanan per Bulan",
                     labels={"month": "Bulan", "total_orders": "Jumlah Pesanan"})
        st.plotly_chart(fig, use_container_width=True)

    valid_changes = monthly.dropna(subset=["sales_change"])
    if not valid_changes.empty:
        highest_increase = valid_changes.loc[valid_changes["sales_change"].idxmax()]
        highest_decrease = valid_changes.loc[valid_changes["sales_change"].idxmin()]
        a, b = st.columns(2)
        a.metric("Peningkatan Penjualan Terbesar", f"BRL {highest_increase['sales_change']:,.2f}",
                 help=highest_increase["month"].strftime("%B %Y"))
        b.metric("Penurunan Penjualan Terbesar", f"BRL {highest_decrease['sales_change']:,.2f}",
                 help=highest_decrease["month"].strftime("%B %Y"))
    st.dataframe(monthly.round(2), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Perbandingan Review Score Berdasarkan Ketepatan Pengiriman")
    status_order = ["Tepat Waktu", "Terlambat"]
    review_df = review_comparison.reindex(status_order).dropna().rename("average_review").reset_index()
    review_df.columns = ["delivery_status", "average_review"]
    counts_df = delivery_counts.reindex(status_order).fillna(0).rename("total_orders").reset_index()
    counts_df.columns = ["delivery_status", "total_orders"]
    summary = review_df.merge(counts_df, on="delivery_status", how="outer")
    a, b = st.columns(2)
    with a:
        fig = px.bar(summary, x="delivery_status", y="average_review", text_auto=".2f",
                     title="Rata-rata Review Score", range_y=[0, 5],
                     labels={"delivery_status": "Status Pengiriman", "average_review": "Rata-rata Review Score"})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.pie(summary, names="delivery_status", values="total_orders", hole=0.4,
                     title="Jumlah Pesanan Berdasarkan Status Pengiriman")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    if set(status_order).issubset(set(review_df["delivery_status"])):
        timely = review_df.loc[review_df["delivery_status"] == "Tepat Waktu", "average_review"].iloc[0]
        late = review_df.loc[review_df["delivery_status"] == "Terlambat", "average_review"].iloc[0]
        st.info(f"Selisih rata-rata review score antara pengiriman tepat waktu dan terlambat adalah {timely - late:.2f} poin.")

with tab3:
    st.subheader("Kontribusi Penjualan dan Jumlah Pesanan per Kategori")
    top_sales = category_sales.nlargest(top_n, "total_sales").sort_values("total_sales")
    top_orders = category_sales.nlargest(top_n, "total_orders").sort_values("total_orders")
    a, b = st.columns(2)
    with a:
        fig = px.bar(top_sales, x="total_sales", y="category", orientation="h",
                     title=f"Top {top_n} Kategori Berdasarkan Nilai Penjualan",
                     labels={"total_sales": "Total Penjualan (BRL)", "category": "Kategori Produk"})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(top_orders, x="total_orders", y="category", orientation="h",
                     title=f"Top {top_n} Kategori Berdasarkan Jumlah Pesanan",
                     labels={"total_orders": "Jumlah Pesanan", "category": "Kategori Produk"})
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(category_sales.round(2), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Segmentasi Pelanggan Menggunakan RFM Analysis")
    segment_summary = rfm.groupby("Customer_Segment").agg(
        Total_Customers=("customer_unique_id", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean"),
    ).reset_index().sort_values("Total_Customers", ascending=False)
    a, b = st.columns(2)
    with a:
        fig = px.bar(segment_summary, x="Customer_Segment", y="Total_Customers",
                     title="Jumlah Pelanggan per Segmen",
                     labels={"Customer_Segment": "Segmen Pelanggan", "Total_Customers": "Jumlah Pelanggan"})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.box(rfm, x="Customer_Segment", y="Monetary",
                     title="Distribusi Nilai Pembelian per Segmen",
                     labels={"Customer_Segment": "Segmen Pelanggan", "Monetary": "Total Pembelian (BRL)"})
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(segment_summary.round(2), use_container_width=True, hide_index=True)
    st.caption("RFM dihitung dari seluruh pesanan delivered, sesuai notebook, bukan hanya periode EDA September 2016–Oktober 2018.")

st.divider()
st.caption("Sumber data: Brazilian E-Commerce Public Dataset (Olist).")
