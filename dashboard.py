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
