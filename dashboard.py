            category_sales.sort_values(
                "total_sales",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# ------------------------- PAGE: RFM -------------------------

elif page == "Customer Segmentation (RFM)":
    st.subheader("4. Customer Segmentation — RFM Analysis")
    st.markdown(
        "Segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary. "
        "Perhitungan mengikuti pendekatan pada notebook analisis."
    )

    rfm = make_rfm(orders, items, customers)

    if rfm.empty:
        st.warning(
            "Data tidak cukup untuk menghitung RFM pada periode yang dipilih."
        )
    else:
        segment_summary = rfm.groupby(
            "Customer_Segment",
            as_index=False
        ).agg(
            Total_Customers=("customer_unique_id", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean")
        ).sort_values(
            "Total_Customers",
            ascending=False
        )

        a, b, c = st.columns(3)
        a.metric("Total Customers", f"{len(rfm):,}")
        b.metric(
            "Average Frequency",
            f'{rfm["Frequency"].mean():.2f} order/customer'
        )
        c.metric(
            "Average Monetary",
            money(rfm["Monetary"].mean())
        )

        left, right = st.columns(2)

        with left:
            fig = px.bar(
                segment_summary.sort_values(
                    "Total_Customers",
                    ascending=False
                ),
                x="Customer_Segment",
                y="Total_Customers",
                title="Jumlah Pelanggan per Segmen RFM",
                labels={
                    "Customer_Segment": "Segmen",
                    "Total_Customers": "Jumlah Pelanggan"
                },
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.box(
                rfm,
                x="Customer_Segment",
                y="Monetary",
                title="Distribusi Nilai Pembelian Berdasarkan Segmen RFM",
                labels={
                    "Customer_Segment": "Customer Segment",
                    "Monetary": "Total Nilai Pembelian Produk (BRL)"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Ringkasan RFM per segmen**")
        st.dataframe(
            segment_summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**Data pelanggan hasil segmentasi**")
        st.dataframe(
            rfm.sort_values("RFM_Score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Monetary dihitung dari kolom price dan tidak memasukkan "
            "freight_value. RFM menggunakan seluruh pesanan berstatus delivered, sesuai notebook."
        )


# ------------------------- FOOTER -------------------------

st.markdown("---")
st.caption(
    "Olist Business Analytics | Nilai penjualan dan Monetary "
    "menggunakan kolom price, tidak termasuk freight_value."
)
