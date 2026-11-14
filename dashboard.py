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
