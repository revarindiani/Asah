import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


st.set_page_config(
    page_title="Olist Business Analytics",
    page_icon="📊",
    layout="wide"
)

# dashboard.py berada di folder dashboard/, sedangkan CSV berada
# di folder data/ yang sejajar dengan dashboard/.
DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"

START_DATE = pd.Timestamp("2016-09-01")
END_DATE_EXCLUSIVE = pd.Timestamp("2018-11-01")


@st.cache_data

def load_data():
    required_files = {
        "orders": "orders.csv",
        "items": "order_items.csv",
        "reviews": "order_reviews.csv",
        "products": "products.csv",
        "customers": "customers.csv",
        "category_translation": "category_translation.csv",
    }

    missing_files = [
        filename for filename in required_files.values()
        if not (DATA_DIR / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"File tidak ditemukan di {DATA_DIR}: {', '.join(missing_files)}"
        )

    orders = pd.read_csv(DATA_DIR / required_files["orders"])
    items = pd.read_csv(DATA_DIR / required_files["items"])
    reviews = pd.read_csv(DATA_DIR / required_files["reviews"])
    products = pd.read_csv(DATA_DIR / required_files["products"])
    customers = pd.read_csv(DATA_DIR / required_files["customers"])
    category_translation = pd.read_csv(
        DATA_DIR / required_files["category_translation"]
    )

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(orders[column], errors="coerce")

    # Sama seperti notebook: gabungkan beberapa review menjadi satu skor/order.
    reviews_clean = (
        reviews.groupby("order_id", as_index=False)
        .agg(review_score=("review_score", "mean"))
    )

    return (
        orders,
        items,
        reviews_clean,
        products,
        customers,
        category_translation,
    )


def format_brl(value):
    return f"BRL {value:,.2f}"


def format_brl_compact(value):
    if abs(value) >= 1_000_000:
        return f"BRL {value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"BRL {value / 1_000:.2f}K"
    return format_brl(value)


def prepare_analysis_data(
    orders,
    items,
    reviews_clean,
    products,
    category_translation,
    start_date,
    end_date_exclusive,
):
    # Mengikuti notebook: hanya pesanan berstatus delivered.
    delivered_orders = orders[
        (orders["order_status"] == "delivered")
        & orders["order_purchase_timestamp"].notna()
        & (orders["order_purchase_timestamp"] >= start_date)
        & (orders["order_purchase_timestamp"] < end_date_exclusive)
    ].copy()

    # Pertanyaan 1: penjualan bulanan.
    sales_data = delivered_orders.merge(items, on="order_id", how="inner")
    if sales_data.empty:
        monthly_sales = pd.DataFrame(
            columns=["month", "total_sales", "total_orders", "sales_change",
                     "sales_growth", "orders_change"]
        )
    else:
        monthly_sales = (
            sales_data.groupby(
                sales_data["order_purchase_timestamp"].dt.to_period("M")
            )
            .agg(
                total_sales=("price", "sum"),
                total_orders=("order_id", "nunique"),
            )
        )

        first_month = start_date.to_period("M")
        last_month = (end_date_exclusive - pd.Timedelta(days=1)).to_period("M")
        all_months = pd.period_range(
            start=first_month, end=last_month, freq="M"
        )
        monthly_sales = monthly_sales.reindex(all_months, fill_value=0)
        monthly_sales.index.name = "month"
        monthly_sales = monthly_sales.reset_index()
        monthly_sales["month"] = monthly_sales["month"].dt.to_timestamp()
        monthly_sales["sales_change"] = monthly_sales["total_sales"].diff()
        previous_sales = monthly_sales["total_sales"].shift(1)
        monthly_sales["sales_growth"] = np.where(
            previous_sales > 0,
            (monthly_sales["total_sales"] - previous_sales)
            / previous_sales * 100,
            np.nan,
        )
        monthly_sales["orders_change"] = monthly_sales["total_orders"].diff()

    # Pertanyaan 2: ketepatan pengiriman dan review.
    delivery_data = delivered_orders[
        delivered_orders["order_delivered_customer_date"].notna()
        & delivered_orders["order_estimated_delivery_date"].notna()
    ].copy()
    delivery_data = delivery_data.merge(
        reviews_clean, on="order_id", how="inner"
    )
    if not delivery_data.empty:
        delivery_data["delivery_difference"] = (
            delivery_data["order_delivered_customer_date"]
            - delivery_data["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400
        delivery_data["delivery_status"] = np.where(
            delivery_data["delivery_difference"] <= 0,
            "Tepat Waktu",
            "Terlambat",
        )
        review_comparison = (
            delivery_data.groupby("delivery_status")["review_score"]
            .mean()
            .reindex(["Tepat Waktu", "Terlambat"])
        )
        delivery_counts = (
            delivery_data["delivery_status"]
            .value_counts()
            .reindex(["Tepat Waktu", "Terlambat"], fill_value=0)
            .rename_axis("delivery_status")
            .reset_index(name="total_orders")
        )
    else:
        review_comparison = pd.Series(
            index=["Tepat Waktu", "Terlambat"], dtype=float,
            name="review_score"
        )
        delivery_counts = pd.DataFrame({
            "delivery_status": ["Tepat Waktu", "Terlambat"],
            "total_orders": [0, 0],
        })

    # Pertanyaan 3: kontribusi kategori produk.
    delivered_order_ids = delivered_orders[["order_id"]].drop_duplicates()
    product_sales = items.merge(
        delivered_order_ids, on="order_id", how="inner"
    )
    product_sales = product_sales.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )
    product_sales = product_sales.merge(
        category_translation,
        on="product_category_name",
        how="left",
    )
    product_sales["product_category_name_english"] = (
        product_sales["product_category_name_english"].fillna("Unknown")
    )

    category_sales = (
        product_sales.groupby("product_category_name_english")
        .agg(
            total_sales=("price", "sum"),
            total_orders=("order_id", "nunique"),
            total_items=("order_item_id", "count"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .reset_index(drop=True)
    )
    top_sales = category_sales.nlargest(10, "total_sales").sort_values(
        "total_sales", ascending=True
    )
    top_orders = category_sales.nlargest(10, "total_orders").sort_values(
        "total_orders", ascending=True
    )

    return (
        delivered_orders,
        sales_data,
        monthly_sales,
        delivery_data,
        review_comparison,
        delivery_counts,
        product_sales,
        category_sales,
        top_sales,
        top_orders,
    )


def calculate_rfm(orders, items, customers):
    # Mengikuti notebook: RFM menggunakan seluruh pesanan delivered,
    # bukan hanya periode yang dipilih pada filter dashboard.
    rfm_orders = orders[orders["order_status"] == "delivered"].copy()
    rfm_data = rfm_orders.merge(items, on="order_id", how="inner")
    rfm_data = rfm_data.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )
    rfm_data = rfm_data.dropna(
        subset=["customer_unique_id", "order_purchase_timestamp"]
    )

    if rfm_data.empty:
        return pd.DataFrame()

    snapshot_date = (
        rfm_data["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    )
    rfm = rfm_data.groupby("customer_unique_id").agg(
        Recency=(
            "order_purchase_timestamp",
            lambda values: (snapshot_date - values.max()).days,
        ),
        Frequency=("order_id", "nunique"),
        Monetary=("price", "sum"),
    ).reset_index()

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        4,
        labels=[4, 3, 2, 1],
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)
    rfm["RFM_Score"] = (
        rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    )

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
    return rfm


# ============================================================
# LOAD DATA
# ============================================================
st.title("📊 Olist Business Analytics")
st.caption(
    "E-Commerce Public Dataset | Sales, Delivery, Product & Customer Insights"
)

try:
    (
        orders_df,
        order_items_df,
        reviews_clean,
        products_df,
        customers_df,
        category_translation_df,
    ) = load_data()
except Exception as error:
    st.error(f"Gagal memuat data: {error}")
    st.info(
        "Pastikan folder data berada sejajar dengan folder dashboard, "
        "dan nama keenam file CSV sama seperti pada struktur folder."
    )
    st.stop()

# ============================================================
# SIDEBAR FILTER & NAVIGATION
# ============================================================
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Overview",
        "Sales Performance",
        "Delivery & Reviews",
        "Product Performance",
        "Customer Segmentation (RFM)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Periode")
date_range = st.sidebar.date_input(
    "Tanggal pembelian",
    value=(START_DATE.date(), (END_DATE_EXCLUSIVE - pd.Timedelta(days=1)).date()),
    min_value=START_DATE.date(),
    max_value=(END_DATE_EXCLUSIVE - pd.Timedelta(days=1)).date(),
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    selected_start = pd.Timestamp(date_range[0])
    selected_end_exclusive = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    selected_start = START_DATE
    selected_end_exclusive = END_DATE_EXCLUSIVE

if selected_start >= selected_end_exclusive:
    st.sidebar.error("Rentang tanggal tidak valid.")
    st.stop()

(
    delivered_orders,
    sales_data,
    monthly_sales,
    delivery_data,
    review_comparison,
    delivery_counts,
    product_sales,
    category_sales,
    top_sales,
    top_orders,
) = prepare_analysis_data(
    orders_df,
    order_items_df,
    reviews_clean,
    products_df,
    category_translation_df,
    selected_start,
    selected_end_exclusive,
)

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "Overview":
    st.subheader("Ringkasan Performa E-Commerce")
    st.write(
        "Halaman ini menyajikan gambaran umum penjualan, pesanan, "
        "kepuasan pelanggan, dan pengiriman pada periode yang dipilih."
    )

    total_sales = sales_data["price"].sum() if not sales_data.empty else 0
    total_orders = (
        sales_data["order_id"].nunique() if not sales_data.empty else 0
    )
    total_customers = (
        delivered_orders["customer_id"].nunique()
        if not delivered_orders.empty else 0
    )
    average_review = (
        delivery_data["review_score"].mean()
        if not delivery_data.empty else 0
    )
    late_orders = (
        (delivery_data["delivery_status"] == "Terlambat").sum()
        if not delivery_data.empty else 0
    )
    late_percentage = (
        late_orders / delivery_data["order_id"].nunique() * 100
        if not delivery_data.empty and delivery_data["order_id"].nunique() > 0
        else 0
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Sales", format_brl_compact(total_sales))
    metric_cols[1].metric("Total Orders", f"{total_orders:,}")
    metric_cols[2].metric("Customers", f"{total_customers:,}")
    metric_cols[3].metric("Average Review", f"{average_review:.2f} / 5")
    metric_cols[4].metric("Late Delivery", f"{late_percentage:.1f}%")

    st.markdown("---")
    if monthly_sales.empty:
        st.warning("Tidak ada data penjualan pada periode yang dipilih.")
    else:
        left, right = st.columns(2)
        with left:
            fig_sales = px.line(
                monthly_sales,
                x="month",
                y="total_sales",
                markers=True,
                title="Perkembangan Total Penjualan Olist per Bulan",
                labels={"month": "Bulan", "total_sales": "Total Penjualan (BRL)"},
            )
            st.plotly_chart(fig_sales, use_container_width=True)
        with right:
            fig_orders = px.bar(
                monthly_sales,
                x="month",
                y="total_orders",
                title="Perkembangan Jumlah Pesanan Olist per Bulan",
                labels={"month": "Bulan", "total_orders": "Jumlah Pesanan"},
            )
            st.plotly_chart(fig_orders, use_container_width=True)

    st.caption(
        "Catatan: analisis penjualan menggunakan pesanan berstatus delivered. "
        "Nilai penjualan dihitung dari kolom price dan tidak memasukkan freight_value."
    )

# ============================================================
# PAGE: BUSINESS QUESTION 1
# ============================================================
elif page == "Sales Performance":
    st.subheader("Pertanyaan 1: Perkembangan Penjualan dan Jumlah Pesanan Bulanan")
    st.write(
        "Analisis total nilai penjualan dan jumlah pesanan Olist setiap bulan."
    )

    if monthly_sales.empty:
        st.warning("Tidak ada data penjualan pada periode yang dipilih.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", format_brl_compact(monthly_sales["total_sales"].sum()))
        col2.metric("Total Orders", f"{int(monthly_sales['total_orders'].sum()):,}")
        best_month = monthly_sales.loc[monthly_sales["total_sales"].idxmax()]
        col3.metric(
            "Bulan Penjualan Tertinggi",
            best_month["month"].strftime("%b %Y"),
        )

        # Grafik mengikuti notebook: garis untuk penjualan, batang untuk pesanan.
        fig, axes = None, None
        chart_left, chart_right = st.columns(2)
        with chart_left:
            fig_sales = px.line(
                monthly_sales,
                x="month",
                y="total_sales",
                markers=True,
                title="Perkembangan Total Penjualan Olist per Bulan",
                labels={"month": "Bulan", "total_sales": "Total Penjualan (BRL)"},
            )
            fig_sales.update_layout(xaxis_tickformat="%b %Y")
            st.plotly_chart(fig_sales, use_container_width=True)
        with chart_right:
            fig_orders = px.bar(
                monthly_sales,
                x="month",
                y="total_orders",
                title="Perkembangan Jumlah Pesanan Olist per Bulan",
                labels={"month": "Bulan", "total_orders": "Jumlah Pesanan"},
            )
            fig_orders.update_layout(xaxis_tickformat="%b %Y")
            st.plotly_chart(fig_orders, use_container_width=True)

        st.markdown("**Tabel perkembangan penjualan dan jumlah pesanan**")
        st.dataframe(
            monthly_sales.round(2),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Catatan: bulan dengan nilai nol dapat terjadi karena filter hanya "
            "menggunakan pesanan delivered; hal ini tidak otomatis berarti "
            "tidak ada transaksi pembelian pada bulan tersebut."
        )

# ============================================================
# PAGE: BUSINESS QUESTION 2
# ============================================================
elif page == "Delivery & Reviews":
    st.subheader("Pertanyaan 2: Ketepatan Pengiriman dan Kepuasan Pelanggan")
    st.write(
        "Perbandingan rata-rata review score antara pesanan tepat waktu dan terlambat."
    )

    if delivery_data.empty:
        st.warning("Tidak ada data pengiriman dan ulasan pada periode yang dipilih.")
    else:
        review_values = review_comparison.dropna()
        on_time_score = review_comparison.get("Tepat Waktu", np.nan)
        late_score = review_comparison.get("Terlambat", np.nan)
        on_time_count = int(
            delivery_counts.loc[
                delivery_counts["delivery_status"] == "Tepat Waktu", "total_orders"
            ].sum()
        )
        late_count = int(
            delivery_counts.loc[
                delivery_counts["delivery_status"] == "Terlambat", "total_orders"
            ].sum()
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Review Tepat Waktu",
            f"{on_time_score:.2f}" if pd.notna(on_time_score) else "-",
        )
        metric_cols[1].metric(
            "Review Terlambat",
            f"{late_score:.2f}" if pd.notna(late_score) else "-",
        )
        metric_cols[2].metric("Pesanan Tepat Waktu", f"{on_time_count:,}")
        metric_cols[3].metric("Pesanan Terlambat", f"{late_count:,}")

        review_plot = review_values.rename("average_review").rename_axis(
            "delivery_status"
        ).reset_index()
        fig = px.bar(
            review_plot,
            x="delivery_status",
            y="average_review",
            text_auto=".2f",
            title="Perbandingan Rata-rata Review Score Berdasarkan Ketepatan Pengiriman",
            labels={
                "delivery_status": "Status Pengiriman",
                "average_review": "Rata-rata Review Score",
            },
        )
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Rata-rata review score berdasarkan status pengiriman**")
        st.dataframe(
            review_plot,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Jumlah pesanan berdasarkan status pengiriman**")
        st.dataframe(
            delivery_counts,
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# PAGE: BUSINESS QUESTION 3
# ============================================================
elif page == "Product Performance":
    st.subheader("Pertanyaan 3: Kontribusi Penjualan Berdasarkan Kategori Produk")
    st.write(
        "Menampilkan kategori produk dengan nilai penjualan dan jumlah pesanan tertinggi."
    )

    if category_sales.empty:
        st.warning("Tidak ada data kategori produk pada periode yang dipilih.")
    else:
        best_sales_category = category_sales.loc[category_sales["total_sales"].idxmax()]
        best_orders_category = category_sales.loc[category_sales["total_orders"].idxmax()]
        metric_cols = st.columns(2)
        metric_cols[0].metric(
            "Penjualan Tertinggi",
            best_sales_category["product_category_name_english"],
            format_brl_compact(best_sales_category["total_sales"]),
        )
        metric_cols[1].metric(
            "Pesanan Terbanyak",
            best_orders_category["product_category_name_english"],
            f"{int(best_orders_category['total_orders']):,} pesanan",
        )

        left, right = st.columns(2)
        with left:
            fig_sales = px.bar(
                top_sales,
                x="total_sales",
                y="product_category_name_english",
                orientation="h",
                title="Top 10 Kategori Produk Berdasarkan Nilai Penjualan",
                labels={
                    "total_sales": "Total Penjualan (BRL)",
                    "product_category_name_english": "Kategori Produk",
                },
            )
            st.plotly_chart(fig_sales, use_container_width=True)
        with right:
            fig_orders = px.bar(
                top_orders,
                x="total_orders",
                y="product_category_name_english",
                orientation="h",
                title="Top 10 Kategori Produk Berdasarkan Jumlah Pesanan",
                labels={
                    "total_orders": "Jumlah Pesanan",
                    "product_category_name_english": "Kategori Produk",
                },
            )
            st.plotly_chart(fig_orders, use_container_width=True)

        st.markdown("**Tabel kontribusi penjualan dan pesanan setiap kategori**")
        st.dataframe(
            category_sales.round(2),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# PAGE: ADVANCED ANALYSIS - RFM
# ============================================================
elif page == "Customer Segmentation (RFM)":
    st.subheader("Analisis Lanjutan: Customer Segmentation (RFM)")
    st.write(
        "Segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary. "
        "Perhitungan mengikuti notebook dan menggunakan seluruh pesanan delivered."
    )

    rfm = calculate_rfm(orders_df, order_items_df, customers_df)
    if rfm.empty:
        st.warning("Data tidak cukup untuk melakukan analisis RFM.")
    else:
        segment_summary = (
            rfm.groupby("Customer_Segment", as_index=False)
            .agg(
                Total_Customers=("customer_unique_id", "count"),
                Average_Recency=("Recency", "mean"),
                Average_Frequency=("Frequency", "mean"),
                Average_Monetary=("Monetary", "mean"),
            )
        )

        segment_order = [
            "Potential Customers", "Loyal Customers", "Champions", "At Risk", "Others"
        ]
        segment_counts = (
            rfm["Customer_Segment"].value_counts()
            .reindex(segment_order, fill_value=0)
            .rename_axis("Customer_Segment")
            .reset_index(name="Total_Customers")
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric("Total Pelanggan", f"{len(rfm):,}")
        metric_cols[1].metric("Rata-rata Frequency", f"{rfm['Frequency'].mean():.2f}")
        metric_cols[2].metric(
            "Rata-rata Monetary", format_brl_compact(rfm["Monetary"].mean())
        )

        left, right = st.columns(2)
        with left:
            fig_segment = px.bar(
                segment_counts,
                x="Customer_Segment",
                y="Total_Customers",
                title="Distribusi Pelanggan Berdasarkan Segmen RFM",
                labels={
                    "Customer_Segment": "Customer Segment",
                    "Total_Customers": "Jumlah Pelanggan",
                },
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        with right:
            # Boxplot sejalan dengan visualisasi notebook.
            fig_box = px.box(
                rfm,
                x="Customer_Segment",
                y="Monetary",
                category_orders={"Customer_Segment": segment_order},
                title="Distribusi Nilai Pembelian Berdasarkan Segmen RFM",
                labels={
                    "Customer_Segment": "Customer Segment",
                    "Monetary": "Total Nilai Pembelian Produk (BRL)",
                },
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**Ringkasan RFM berdasarkan segmen pelanggan**")
        st.dataframe(
            segment_summary.round(2),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Data pelanggan hasil segmentasi RFM**")
        st.dataframe(
            rfm.sort_values("RFM_Score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.caption(
    "Olist Business Analytics | Nilai penjualan dan Monetary dihitung dari "
    "kolom price, tidak termasuk freight_value."
)
