import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    [data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e8edf4;
    }
    h1, h2, h3 {color: #17365D;}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Dashboard Analisis E-Commerce Olist")
st.caption("Dashboard interaktif berdasarkan proyek Brazilian E-Commerce Public Dataset (Olist)")
st.info(
    "Unggah file CSV yang digunakan pada notebook. Dashboard ini memakai pesanan "
    "berstatus delivered dan periode analisis default September 2016–Oktober 2018."
)

REQUIRED_FILES = {
    "orders_dataset.csv": "Data pesanan",
    "order_items_dataset.csv": "Detail item pesanan",
    "order_reviews_dataset.csv": "Ulasan pelanggan",
    "products_dataset.csv": "Data produk",
    "product_category_name_translation.csv": "Terjemahan kategori",
    "customers_dataset.csv": "Data pelanggan (untuk RFM)",
}

@st.cache_data(show_spinner="Membaca dan menyiapkan data...")
def read_uploaded_csv(file_bytes, filename):
    return pd.read_csv(BytesIO(file_bytes))

@st.cache_data(show_spinner="Mengolah data dashboard...")
def prepare_data(file_map):
    orders = pd.read_csv(BytesIO(file_map["orders_dataset.csv"]))
    items = pd.read_csv(BytesIO(file_map["order_items_dataset.csv"]))
    reviews = pd.read_csv(BytesIO(file_map["order_reviews_dataset.csv"]))
    products = pd.read_csv(BytesIO(file_map["products_dataset.csv"]))
    translation = pd.read_csv(BytesIO(file_map["product_category_name_translation.csv"]))
    customers = pd.read_csv(BytesIO(file_map["customers_dataset.csv"]))

    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    delivered = delivered.dropna(subset=["order_purchase_timestamp"])

    # Data penjualan bulanan
    sales_rows = delivered.merge(items, on="order_id", how="inner")
    sales_rows["month"] = sales_rows["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = sales_rows.groupby("month").agg(
        total_sales=("price", "sum"),
        total_orders=("order_id", "nunique")
    ).reset_index()

    # Data ketepatan pengiriman dan ulasan: satu skor rata-rata per pesanan
    reviews_one = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )
    delivery = delivered.dropna(subset=[
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ]).merge(reviews_one, on="order_id", how="inner")
    delivery["delivery_difference"] = (
        delivery["order_delivered_customer_date"] -
        delivery["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    delivery["delivery_status"] = np.where(
        delivery["delivery_difference"] <= 0, "Tepat Waktu", "Terlambat"
    )
    delivery_summary = delivery.groupby("delivery_status").agg(
        average_review=("review_score", "mean"),
        total_orders=("order_id", "nunique")
    ).reindex(["Tepat Waktu", "Terlambat"]).reset_index()

    # Kontribusi kategori produk
    product_rows = (
        sales_rows.merge(
            products[["product_id", "product_category_name"]],
            on="product_id", how="left"
        ).merge(translation, on="product_category_name", how="left")
    )
    category_col = "product_category_name_english"
    if category_col not in product_rows.columns:
        product_rows[category_col] = "Unknown"
    product_rows[category_col] = product_rows[category_col].fillna("Unknown")
    categories = product_rows.groupby(category_col).agg(
        total_sales=("price", "sum"),
        total_orders=("order_id", "nunique"),
        total_items=("order_item_id", "count")
    ).reset_index().rename(columns={category_col: "category"})

    # RFM per pelanggan unik, mengikuti logika notebook
    rfm_rows = (
        delivered.merge(items, on="order_id", how="inner")
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
    )
    snapshot = rfm_rows["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = rfm_rows.groupby("customer_unique_id").agg(
        Recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("price", "sum")
    ).reset_index()

    if len(rfm) >= 4:
        rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
        def segment(score):
            if score >= 10: return "Champions"
            if score >= 8: return "Loyal Customers"
            if score >= 6: return "Potential Customers"
            if score >= 4: return "At Risk"
            return "Others"
        rfm["Customer_Segment"] = rfm["RFM_Score"].apply(segment)
    else:
        rfm["Customer_Segment"] = "Belum dapat dikelompokkan"

    return monthly, delivery_summary, delivery, categories, rfm, delivered

with st.sidebar:
    st.header("⚙️ Pengaturan Dashboard")
    st.write("Unggah enam file CSV berikut. Nama file harus sesuai nama dataset.")
    uploads = st.file_uploader(
        "Pilih file dataset (CSV)",
        type=["csv"],
        accept_multiple_files=True
    )

file_map = {}
if uploads:
    for uploaded in uploads:
        file_map[uploaded.name] = uploaded.getvalue()

missing = [name for name in REQUIRED_FILES if name not in file_map]
if missing:
    st.warning("File yang masih diperlukan: " + ", ".join(missing))
    st.markdown("**File yang perlu diunggah:**")
    for filename, description in REQUIRED_FILES.items():
        status = "✅" if filename in file_map else "⬜"
        st.write(f"{status} `{filename}` — {description}")
    st.stop()

try:
    monthly, delivery_summary, delivery, categories, rfm, delivered = prepare_data(file_map)
except Exception as error:
    st.error(f"Terjadi kesalahan saat mengolah dataset: {error}")
    st.stop()

min_date = delivered["order_purchase_timestamp"].min().date()
max_date = delivered["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.divider()
    date_range = st.date_input(
        "Periode analisis",
        value=(max(min_date, pd.Timestamp("2016-09-01").date()),
               min(max_date, pd.Timestamp("2018-10-31").date())),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    top_n = st.slider("Jumlah kategori ditampilkan", 5, 20, 10)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

# Terapkan filter periode pada data transaksi
filtered_orders = delivered[
    delivered["order_purchase_timestamp"].between(start_ts, end_ts, inclusive="left")
]
filtered_order_ids = filtered_orders["order_id"].unique()
filtered_monthly = monthly[
    monthly["month"].between(start_ts.to_period("M").to_timestamp(),
                             end_ts.to_period("M").to_timestamp(), inclusive="both")
].copy()

# Ringkasan penjualan dan pesanan
filtered_items = filtered_orders[["order_id"]].merge(
    pd.read_csv(BytesIO(file_map["order_items_dataset.csv"])),
    on="order_id", how="inner"
)
total_sales = filtered_items["price"].sum()
total_orders = filtered_orders["order_id"].nunique()
total_customers = filtered_orders["customer_id"].nunique()
average_order_value = total_sales / total_orders if total_orders else 0

st.subheader("Ringkasan Kinerja")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Penjualan", f"BRL {total_sales:,.2f}")
k2.metric("Total Pesanan", f"{total_orders:,.0f}")
k3.metric("Pelanggan", f"{total_customers:,.0f}")
k4.metric("Rata-rata Nilai Pesanan", f"BRL {average_order_value:,.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Penjualan", "🚚 Pengiriman & Ulasan", "📦 Kategori Produk", "👥 Segmentasi RFM"
])

with tab1:
    st.subheader("Perkembangan Penjualan Bulanan")
    if not filtered_monthly.empty:
        left, right = st.columns(2)
        with left:
            fig = px.line(
                filtered_monthly, x="month", y="total_sales", markers=True,
                title="Total Penjualan per Bulan",
                labels={"month": "Bulan", "total_sales": "Penjualan (BRL)"}
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                filtered_monthly, x="month", y="total_orders",
                title="Jumlah Pesanan per Bulan",
                labels={"month": "Bulan", "total_orders": "Jumlah Pesanan"}
            )
            st.plotly_chart(fig, use_container_width=True)
        filtered_monthly["sales_change"] = filtered_monthly["total_sales"].diff()
        st.markdown("**Data penjualan bulanan**")
        st.dataframe(filtered_monthly, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada transaksi pada periode yang dipilih.")

with tab2:
    st.subheader("Ketepatan Pengiriman dan Kepuasan Pelanggan")
    delivery_filtered = delivery[
        delivery["order_purchase_timestamp"].between(start_ts, end_ts, inclusive="left")
    ]
    if not delivery_filtered.empty:
        summary = delivery_filtered.groupby("delivery_status").agg(
            average_review=("review_score", "mean"),
            total_orders=("order_id", "nunique")
        ).reindex(["Tepat Waktu", "Terlambat"]).reset_index()
        a, b = st.columns(2)
        with a:
            fig = px.bar(
                summary, x="delivery_status", y="average_review",
                text_auto=".2f", title="Rata-rata Skor Ulasan",
                labels={"delivery_status": "Status Pengiriman", "average_review": "Rata-rata Review Score"}
            )
            fig.update_yaxes(range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.pie(
                summary, names="delivery_status", values="total_orders",
                title="Proporsi Pesanan Berdasarkan Ketepatan Pengiriman", hole=.45
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        timely = summary.loc[summary["delivery_status"] == "Tepat Waktu", "average_review"]
        late = summary.loc[summary["delivery_status"] == "Terlambat", "average_review"]
        if not timely.empty and not late.empty:
            st.write(f"Selisih rata-rata skor ulasan: **{timely.iloc[0] - late.iloc[0]:.2f} poin**.")
    else:
        st.info("Tidak tersedia data pengiriman dan ulasan pada periode yang dipilih.")

with tab3:
    st.subheader("Kontribusi Penjualan Berdasarkan Kategori Produk")
    # Kategori dihitung ulang sesuai pesanan yang lolos filter tanggal
    items_df = pd.read_csv(BytesIO(file_map["order_items_dataset.csv"]))
    products_df = pd.read_csv(BytesIO(file_map["products_dataset.csv"]))
    translation_df = pd.read_csv(BytesIO(file_map["product_category_name_translation.csv"]))
    category_filtered = (
        filtered_orders[["order_id"]].merge(items_df, on="order_id", how="inner")
        .merge(products_df[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(translation_df, on="product_category_name", how="left")
    )
    category_filtered["product_category_name_english"] = category_filtered[
        "product_category_name_english"
    ].fillna("Unknown")
    category_filtered = category_filtered.groupby("product_category_name_english").agg(
        total_sales=("price", "sum"),
        total_orders=("order_id", "nunique")
    ).reset_index().rename(columns={"product_category_name_english": "category"})
    c1, c2 = st.columns(2)
    top_sales = category_filtered.nlargest(top_n, "total_sales")
    top_orders = category_filtered.nlargest(top_n, "total_orders")
    with c1:
        fig = px.bar(
            top_sales.sort_values("total_sales"), x="total_sales", y="category",
            orientation="h", title=f"Top {top_n} Kategori Berdasarkan Penjualan",
            labels={"total_sales": "Penjualan (BRL)", "category": "Kategori"}
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            top_orders.sort_values("total_orders"), x="total_orders", y="category",
            orientation="h", title=f"Top {top_n} Kategori Berdasarkan Pesanan",
            labels={"total_orders": "Jumlah Pesanan", "category": "Kategori"}
        )
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(category_filtered.sort_values("total_sales", ascending=False),
                 use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Segmentasi Pelanggan Menggunakan RFM")
    segment_summary = rfm.groupby("Customer_Segment").agg(
        Total_Customers=("customer_unique_id", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean")
    ).reset_index()
    if not segment_summary.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                segment_summary, x="Customer_Segment", y="Total_Customers",
                title="Jumlah Pelanggan per Segmen",
                labels={"Customer_Segment": "Segmen Pelanggan", "Total_Customers": "Jumlah Pelanggan"}
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(
                rfm, x="Customer_Segment", y="Monetary",
                title="Distribusi Nilai Pembelian per Segmen",
                labels={"Customer_Segment": "Segmen Pelanggan", "Monetary": "Nilai Pembelian (BRL)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(segment_summary.round(2), use_container_width=True, hide_index=True)
        st.caption("Segmentasi RFM dihitung dari seluruh periode data delivered, mengikuti notebook, bukan hanya rentang tanggal filter dashboard.")
    else:
        st.info("Data pelanggan belum cukup untuk segmentasi RFM.")

st.divider()
st.caption("Catatan: Nilai penjualan dihitung dari kolom `price` pada item pesanan, tidak termasuk biaya pengiriman (`freight_value`), mengikuti perhitungan pada notebook.")
