import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# OLIST BUSINESS ANALYTICS DASHBOARD
# Dataset: E-commerce-public-dataset-clean
# =========================================================

st.set_page_config(
    page_title="Olist | E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
    min-height: 110px;
}

[data-testid="stMetricLabel"] {
    color: var(--text-color) !important;
    font-size: 14px !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-color) !important;
    font-size: clamp(20px, 1.8vw, 30px) !important;
}

[data-testid="stMetricValue"] > div {
    overflow: visible !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"

REQUIRED_FILES = {
    "orders": "orders.csv",
    "items": "order_items.csv",
    "reviews": "order_reviews.csv",
    "products": "products.csv",
    "customers": "customers.csv",
    "category": "category_translation.csv",
}

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

@st.cache_data
def load_data():
    missing = [
        filename for filename in REQUIRED_FILES.values()
        if not (DATA_DIR / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "File berikut belum ditemukan di folder data/: "
            + ", ".join(missing)
        )

    orders = pd.read_csv(DATA_DIR / REQUIRED_FILES["orders"])
    items = pd.read_csv(DATA_DIR / REQUIRED_FILES["items"])
    reviews = pd.read_csv(DATA_DIR / REQUIRED_FILES["reviews"])
    products = pd.read_csv(DATA_DIR / REQUIRED_FILES["products"])
    customers = pd.read_csv(DATA_DIR / REQUIRED_FILES["customers"])
    category = pd.read_csv(DATA_DIR / REQUIRED_FILES["category"])

    for col in DATE_COLUMNS:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    if "shipping_limit_date" in items.columns:
        items["shipping_limit_date"] = pd.to_datetime(
            items["shipping_limit_date"], errors="coerce"
        )

    # Satu skor rata-rata per order, mengikuti proses cleaning notebook.
    reviews_clean = (
        reviews.groupby("order_id", as_index=False)
        .agg(review_score=("review_score", "mean"))
    )

    return orders, items, reviews_clean, products, customers, category


def money(value):
    return f"R$ {value:,.2f}"


def make_rfm(orders, items, customers, start_date, end_date):
    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_purchase_timestamp"].notna()
        & (orders["order_purchase_timestamp"] >= pd.Timestamp(start_date))
        & (orders["order_purchase_timestamp"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
    ].copy()

    data = delivered.merge(items, on="order_id", how="inner")
    data = data.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )

    data = data.dropna(subset=["customer_unique_id", "order_purchase_timestamp"])
    if data.empty:
        return pd.DataFrame()

    snapshot = data["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = data.groupby("customer_unique_id").agg(
        Recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("price", "sum")
    ).reset_index()

    # Rank terlebih dahulu agar qcut tetap dapat digunakan saat banyak nilai sama.
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        4, labels=[4, 3, 2, 1]
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        4, labels=[1, 2, 3, 4]
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        4, labels=[1, 2, 3, 4]
    ).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def segment(score):
        if score >= 10:
            return "Champions"
        elif score >= 8:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Customers"
        elif score >= 4:
            return "At Risk"
        return "Others"

    rfm["Customer_Segment"] = rfm["RFM_Score"].apply(segment)
    return rfm


# ------------------------- LOAD -------------------------
st.markdown("""
<style>
    .main {background-color: #F7F9FC;}
    [data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #E8EDF5;
    }
    h1, h2, h3 {color: #172B4D;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Olist Business Analytics")
st.caption("E-Commerce Public Dataset | Sales, Delivery, Product & Customer Insights")

try:
    orders, items, reviews_clean, products, customers, category = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.info(
        "Pastikan enam file CSV hasil cleaning berada di folder `data/` "
        "dengan nama file yang sesuai. Lihat README.md."
    )
    st.stop()

# ------------------------- SIDEBAR FILTER -------------------------
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Overview",
        "Sales Performance",
        "Delivery & Reviews",
        "Product Performance",
        "Customer Segmentation (RFM)",
    ]
)

min_date = orders["order_purchase_timestamp"].min()
max_date = orders["order_purchase_timestamp"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Kolom order_purchase_timestamp tidak memiliki tanggal valid.")
    st.stop()

default_start = max(pd.Timestamp("2016-09-01"), min_date.normalize())
default_end = min(pd.Timestamp("2018-10-31"), max_date.normalize())

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Periode")
date_range = st.sidebar.date_input(
    "Tanggal pembelian",
    value=(default_start.date(), default_end.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start.date(), default_end.date()

if start_date > end_date:
    st.sidebar.error("Tanggal awal tidak boleh melewati tanggal akhir.")
    st.stop()

start_ts = pd.Timestamp(start_date)
end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

# Dataset transaksi delivered sesuai definisi analisis pada notebook.
delivered_orders = orders[
    (orders["order_status"] == "delivered")
    & orders["order_purchase_timestamp"].notna()
    & (orders["order_purchase_timestamp"] >= start_ts)
    & (orders["order_purchase_timestamp"] < end_exclusive)
].copy()

sales_data = delivered_orders.merge(items, on="order_id", how="inner")

# Dataset delivery + review
delivery_base = delivered_orders[
    delivered_orders["order_delivered_customer_date"].notna()
    & delivered_orders["order_estimated_delivery_date"].notna()
].copy()

delivery_data = delivery_base.merge(reviews_clean, on="order_id", how="inner")
delivery_data["delivery_difference"] = (
    delivery_data["order_delivered_customer_date"]
    - delivery_data["order_estimated_delivery_date"]
).dt.total_seconds() / 86400
delivery_data["delivery_status"] = np.where(
    delivery_data["delivery_difference"] <= 0,
    "Tepat Waktu",
    "Terlambat"
)

# Dataset kategori produk
product_sales = sales_data.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)
product_sales = product_sales.merge(
    category,
    on="product_category_name",
    how="left"
)
product_sales["product_category_name_english"] = (
    product_sales["product_category_name_english"].fillna("Unknown")
)

# ------------------------- PAGE: OVERVIEW -------------------------
if page == "Overview":
    st.subheader("Ringkasan Performa E-Commerce")
    st.write(
        "Halaman ini menyajikan gambaran umum penjualan, pesanan, "
        "kepuasan pelanggan, dan pengiriman pada periode yang dipilih."
    )

    total_sales = sales_data["price"].sum()
    total_orders = sales_data["order_id"].nunique()
    total_customers = delivered_orders["customer_id"].nunique()
    avg_review = delivery_data["review_score"].mean() if not delivery_data.empty else 0
    late_pct = (
        (delivery_data["delivery_status"] == "Terlambat").mean() * 100
        if not delivery_data.empty else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", money(total_sales))
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Customers", f"{total_customers:,}")
    c4.metric("Average Review", f"{avg_review:.2f} / 5")
    c5.metric("Late Delivery", f"{late_pct:.1f}%")

    st.markdown("---")
    monthly = (
        sales_data.assign(month=sales_data["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(total_sales=("price", "sum"), total_orders=("order_id", "nunique"))
        .reset_index()
    )
    left, right = st.columns(2)
    with left:
        fig = px.line(
            monthly, x="month", y="total_sales", markers=True,
            title="Tren Nilai Penjualan Bulanan",
            labels={"month": "Bulan", "total_sales": "Penjualan (BRL)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(
            monthly, x="month", y="total_orders", markers=True,
            title="Tren Jumlah Pesanan Bulanan",
            labels={"month": "Bulan", "total_orders": "Jumlah Pesanan"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Catatan: nilai penjualan menggunakan kolom price dan tidak memasukkan freight_value.")

# ------------------------- PAGE: SALES -------------------------
elif page == "Sales Performance":
    st.subheader("1. Sales Performance")
    st.markdown(
        "Analisis perkembangan total nilai penjualan dan jumlah pesanan "
        "Olist setiap bulan."
    )

    monthly = (
        sales_data.assign(month=sales_data["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(total_sales=("price", "sum"), total_orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("month")
    )
    if monthly.empty:
        st.warning("Tidak ada transaksi delivered pada periode yang dipilih.")
    else:
        monthly["sales_change"] = monthly["total_sales"].diff()
        monthly["sales_growth"] = monthly["total_sales"].pct_change() * 100

        a, b, c = st.columns(3)
        a.metric("Total Sales", money(monthly["total_sales"].sum()))
        b.metric("Total Orders", f"{monthly['total_orders'].sum():,}")
        best_month = monthly.loc[monthly["total_sales"].idxmax()]
        c.metric("Bulan Penjualan Tertinggi", best_month["month"].strftime("%b %Y"))

        fig = px.line(
            monthly, x="month", y="total_sales", markers=True,
            title="Tren Total Nilai Penjualan Bulanan",
            labels={"month": "Bulan", "total_sales": "Penjualan (BRL)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            monthly, x="month", y="total_orders", markers=True,
            title="Tren Jumlah Pesanan Bulanan",
            labels={"month": "Bulan", "total_orders": "Jumlah Pesanan"}
        )
        st.plotly_chart(fig, use_container_width=True)

        changes = monthly.dropna(subset=["sales_change"])
        fig = px.bar(
            changes, x="month", y="sales_change",
            title="Perubahan Penjualan Month-over-Month",
            labels={"month": "Bulan", "sales_change": "Perubahan Penjualan (BRL)"},
            color="sales_change",
            color_continuous_scale=["#D96C6C", "#E8EDF5", "#249B83"]
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Ringkasan perubahan penjualan**")
        if not changes.empty:
            inc = changes.loc[changes["sales_change"].idxmax()]
            dec = changes.loc[changes["sales_change"].idxmin()]
            x, y = st.columns(2)
            x.metric("Kenaikan Terbesar", money(inc["sales_change"]), inc["month"].strftime("%B %Y"))
            y.metric("Penurunan Terbesar", money(dec["sales_change"]), dec["month"].strftime("%B %Y"))

        st.dataframe(monthly, use_container_width=True, hide_index=True)

# ------------------------- PAGE: DELIVERY -------------------------
elif page == "Delivery & Reviews":
    st.subheader("2. Delivery & Customer Reviews")
    st.markdown(
        "Membandingkan rata-rata skor ulasan antara pesanan tepat waktu "
        "dan terlambat."
    )

    if delivery_data.empty:
        st.warning("Tidak ada data pengiriman dan ulasan yang sesuai filter.")
    else:
        summary = delivery_data.groupby("delivery_status").agg(
            total_orders=("order_id", "nunique"),
            average_review=("review_score", "mean")
        ).reset_index()
        summary["percentage"] = summary["total_orders"] / summary["total_orders"].sum() * 100

        ontime = summary.loc[summary["delivery_status"] == "Tepat Waktu"]
        late = summary.loc[summary["delivery_status"] == "Terlambat"]
        avg_on = ontime["average_review"].iloc[0] if not ontime.empty else 0
        avg_late = late["average_review"].iloc[0] if not late.empty else 0
        late_percent = late["percentage"].iloc[0] if not late.empty else 0

        a, b, c = st.columns(3)
        a.metric("Rata-rata Review Tepat Waktu", f"{avg_on:.2f} / 5")
        b.metric("Rata-rata Review Terlambat", f"{avg_late:.2f} / 5")
        c.metric("Persentase Pesanan Terlambat", f"{late_percent:.1f}%")

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                summary, x="delivery_status", y="average_review",
                title="Rata-rata Review Berdasarkan Status Pengiriman",
                text_auto=".2f",
                labels={"delivery_status": "Status Pengiriman", "average_review": "Rata-rata Review"}
            )
            fig.update_yaxes(range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                summary, x="delivery_status", y="total_orders",
                title="Jumlah Pesanan Berdasarkan Status Pengiriman",
                text_auto=True,
                labels={"delivery_status": "Status Pengiriman", "total_orders": "Jumlah Pesanan"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "Pesanan tanpa tanggal pengiriman aktual, estimasi pengiriman, "
            "atau review tidak dimasukkan dalam perbandingan."
        )

# ------------------------- PAGE: PRODUCT -------------------------
elif page == "Product Performance":
    st.subheader("3. Product Performance")
    st.markdown(
        "Menampilkan kategori produk dengan nilai penjualan dan jumlah pesanan tertinggi."
    )

    if product_sales.empty:
        st.warning("Tidak ada data produk pada periode yang dipilih.")
    else:
        category_sales = product_sales.groupby(
            "product_category_name_english", as_index=False
        ).agg(
            total_sales=("price", "sum"),
            total_orders=("order_id", "nunique")
        )

        top_sales = category_sales.nlargest(10, "total_sales").sort_values("total_sales")
        top_orders = category_sales.nlargest(10, "total_orders").sort_values("total_orders")

        top_cat = category_sales.loc[category_sales["total_sales"].idxmax()]
        top_order_cat = category_sales.loc[category_sales["total_orders"].idxmax()]
        a, b = st.columns(2)
        a.metric("Kategori Penjualan Tertinggi", top_cat["product_category_name_english"], money(top_cat["total_sales"]))
        b.metric("Kategori Pesanan Terbanyak", top_order_cat["product_category_name_english"], f'{int(top_order_cat["total_orders"]):,} pesanan')

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                top_sales, x="total_sales", y="product_category_name_english",
                orientation="h", title="Top 10 Kategori Berdasarkan Nilai Penjualan",
                labels={"total_sales": "Penjualan (BRL)", "product_category_name_english": "Kategori"}
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                top_orders, x="total_orders", y="product_category_name_english",
                orientation="h", title="Top 10 Kategori Berdasarkan Jumlah Pesanan",
                labels={"total_orders": "Jumlah Pesanan", "product_category_name_english": "Kategori"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            category_sales.sort_values("total_sales", ascending=False),
            use_container_width=True, hide_index=True
        )

# ------------------------- PAGE: RFM -------------------------
elif page == "Customer Segmentation (RFM)":
    st.subheader("4. Customer Segmentation — RFM Analysis")
    st.markdown(
        "Segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary. "
        "Perhitungan mengikuti pendekatan pada notebook analisis."
    )

    rfm = make_rfm(orders, items, customers, start_date, end_date)
    if rfm.empty:
        st.warning("Data tidak cukup untuk menghitung RFM pada periode yang dipilih.")
    else:
        segment_summary = rfm.groupby("Customer_Segment", as_index=False).agg(
            Total_Customers=("customer_unique_id", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean")
        ).sort_values("Total_Customers", ascending=False)

        a, b, c = st.columns(3)
        a.metric("Total Customers", f'{len(rfm):,}')
        a.metric("Average Frequency", f'{rfm["Frequency"].mean():.2f} order/customer')
        c.metric("Average Monetary", money(rfm["Monetary"].mean()))

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                segment_summary.sort_values("Total_Customers", ascending=False),
                x="Customer_Segment", y="Total_Customers",
                title="Jumlah Pelanggan per Segmen RFM",
                labels={"Customer_Segment": "Segmen", "Total_Customers": "Jumlah Pelanggan"},
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                segment_summary.sort_values("Average_Monetary", ascending=False),
                x="Customer_Segment", y="Average_Monetary",
                title="Rata-rata Monetary per Segmen",
                labels={"Customer_Segment": "Segmen", "Average_Monetary": "Monetary (BRL)"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Ringkasan RFM per segmen**")
        st.dataframe(segment_summary, use_container_width=True, hide_index=True)

        st.markdown("**Data pelanggan hasil segmentasi**")
        st.dataframe(
            rfm.sort_values("RFM_Score", ascending=False),
            use_container_width=True, hide_index=True
        )

st.markdown("---")
st.caption(
    "Olist Business Analytics | Nilai penjualan/Monetary menggunakan kolom price "
    "dan tidak memasukkan freight_value."
)
