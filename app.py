import pandas as pd
import streamlit as st

st.title("Neural Stock Demand Forecasting")

df = pd.read_csv("data/ecommerce_inventory_demand.csv")

df["date"] = pd.to_datetime(df["date"])

categories = sorted(df["product_category"].unique())

selected_category = st.selectbox(
    "Select Product Category",
    categories
)

date_range = st.date_input(
    "Select Date Range",
    value=(
        df["date"].min().date(),
        df["date"].max().date()
    ),
    key="date_range"
)

if len(date_range) != 2:
    st.info("Please select both start date and end date")
    st.stop()

start_date, end_date = date_range

filtered_df = df[
    (df["product_category"] == selected_category)
    & (df["date"] >= pd.to_datetime(start_date))
    & (df["date"] <= pd.to_datetime(end_date))
].copy()

if filtered_df.empty:
    st.error("No data available for selected date range")
    st.stop()

avg_demand = filtered_df["units_sold"].mean()

prediction = round(avg_demand, 2)

st.subheader("Filter Summary")

st.write(f"Records Found: {len(filtered_df)}")

st.write(
    f"Average Units Sold: {round(avg_demand, 2)}"
)

st.subheader("Predicted Demand")

st.metric(
    label="Forecasted Demand",
    value=prediction
)

# Review proof section
st.subheader("Filter Summary")

st.write(
    f"Records Found: {len(filtered_df)}"
)

st.write(
    f"Average Units Sold: {round(avg_demand, 2)}"
)

forecast_df = pd.DataFrame(
    {
        "Week": [
            "Week 1",
            "Week 2",
            "Week 3",
            "Week 4"
        ],
        "Forecast": [
            round(prediction, 2),
            round(prediction * 1.03, 2),
            round(prediction * 1.05, 2),
            round(prediction * 1.08, 2)
        ]
    }
)

st.subheader("4 Week Demand Forecast")

st.line_chart(
    forecast_df.set_index("Week")
)

current_stock = filtered_df["stock_on_hand"].iloc[-1]

reorder_df = pd.DataFrame(
    {
        "Category": [selected_category],
        "Current Stock": [current_stock],
        "Forecast Demand": [
            forecast_df["Forecast"].max()
        ]
    }
)

st.subheader("Reorder Alert Table")

st.dataframe(
    reorder_df,
    use_container_width=True
)

if forecast_df["Forecast"].max() > current_stock:

    st.warning(
        "Reorder Alert: Forecast exceeds stock level"
    )

else:

    st.success(
        "Stock level is sufficient"
    )

st.subheader("Sample Filtered Data")

st.dataframe(
    filtered_df[
        [
            "date",
            "product_category",
            "units_sold",
            "stock_on_hand"
        ]
    ].head(10),
    use_container_width=True
)

csv = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="forecast.csv",
    mime="text/csv"
)