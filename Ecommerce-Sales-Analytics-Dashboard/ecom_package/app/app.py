import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="E-commerce Sales Analytics Dashboard", layout="wide", page_icon="🛒")

# ---------- Load Data & Model ----------
@st.cache_data
def load_data():
    df = pd.read_csv("global_ecommerce_sales_cleaned.csv", parse_dates=["Order_Date"])
    return df

@st.cache_resource
def load_model():
    with open("sales_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model_bundle = load_model()
model = model_bundle["model"]

# ---------- Sidebar Filters ----------
st.sidebar.title("🛒 Filters")
regions = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
categories = st.sidebar.multiselect("Product Category", sorted(df["Product_Category"].unique()), default=sorted(df["Product_Category"].unique()))
segments = st.sidebar.multiselect("Customer Segment", sorted(df["Customer_Segment"].unique()), default=sorted(df["Customer_Segment"].unique()))
date_range = st.sidebar.date_input("Order Date Range", [df["Order_Date"].min(), df["Order_Date"].max()])

mask = (
    df["Region"].isin(regions)
    & df["Product_Category"].isin(categories)
    & df["Customer_Segment"].isin(segments)
    & (df["Order_Date"] >= pd.to_datetime(date_range[0]))
    & (df["Order_Date"] <= pd.to_datetime(date_range[1]))
)
fdf = df[mask]

st.title("🛒 E-commerce Sales Analytics Dashboard")
st.caption("Dataset: Global E-Commerce Sales & Customer Data (Kaggle) — "
           "https://www.kaggle.com/datasets/muhammadaammartufail/global-e-commerce-sales-and-customer-data")

if fdf.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ---------- KPI Cards ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"${fdf['Total_Sales'].sum():,.0f}")
k2.metric("Total Orders", f"{fdf['Order_ID'].nunique():,}")
k3.metric("Total Profit", f"${fdf['Profit'].sum():,.0f}")
margin = fdf['Profit'].sum() / fdf['Total_Sales'].sum() * 100 if fdf['Total_Sales'].sum() else 0
k4.metric("Profit Margin", f"{margin:.1f}%")

st.divider()

tabs = st.tabs(["📈 Sales Overview", "📦 Products & Categories", "🌍 Regional", "👥 Customers", "🔮 Sales Prediction"])

# ---------- Tab 1: Sales Overview ----------
with tabs[0]:
    st.subheader("Monthly Sales Trend")
    monthly = fdf.groupby(fdf["Order_Date"].dt.to_period("M")).agg(
        Total_Sales=("Total_Sales", "sum"), Orders=("Order_ID", "count")
    ).reset_index()
    monthly["Order_Date"] = monthly["Order_Date"].dt.to_timestamp()
    fig = px.line(monthly, x="Order_Date", y="Total_Sales", markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        yearly = fdf.groupby(fdf["Order_Date"].dt.year).agg(
            Revenue=("Total_Sales", "sum"), Profit=("Profit", "sum")
        ).reset_index().rename(columns={"Order_Date": "Year"})
        fig2 = px.bar(yearly, x=yearly.columns[0], y=["Revenue", "Profit"], barmode="group",
                      title="Yearly Revenue vs Profit")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        pay = fdf["Payment_Method"].value_counts().reset_index()
        pay.columns = ["Payment_Method", "Count"]
        fig3 = px.pie(pay, names="Payment_Method", values="Count", title="Payment Method Distribution", hole=0.35)
        st.plotly_chart(fig3, use_container_width=True)

# ---------- Tab 2: Products & Categories ----------
with tabs[1]:
    st.subheader("Top Selling Products")
    top_products = fdf.groupby("Product_Name")["Total_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    fig4 = px.bar(top_products, x="Total_Sales", y="Product_Name", orientation="h", title="Top 10 Products by Revenue")
    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Category-wise Sales & Profit")
    cat_perf = fdf.groupby("Product_Category").agg(Total_Sales=("Total_Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    fig5 = px.bar(cat_perf, x="Product_Category", y=["Total_Sales", "Profit"], barmode="group",
                  title="Category Performance")
    st.plotly_chart(fig5, use_container_width=True)

# ---------- Tab 3: Regional ----------
with tabs[2]:
    st.subheader("Regional Sales Performance")
    region_perf = fdf.groupby("Region")["Total_Sales"].sum().sort_values(ascending=False).reset_index()
    fig6 = px.bar(region_perf, x="Region", y="Total_Sales", title="Sales by Region")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Sales by Country")
    country_perf = fdf.groupby("Country")["Total_Sales"].sum().sort_values(ascending=False).reset_index()
    fig7 = px.choropleth(country_perf, locations="Country", locationmode="country names",
                          color="Total_Sales", title="Sales by Country", color_continuous_scale="Blues")
    st.plotly_chart(fig7, use_container_width=True)

# ---------- Tab 4: Customers ----------
with tabs[3]:
    st.subheader("Customer Segment Analysis")
    seg = fdf.groupby("Customer_Segment").agg(
        Total_Sales=("Total_Sales", "sum"),
        Avg_Order=("Total_Sales", "mean"),
        Orders=("Order_ID", "count")
    ).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig8 = px.bar(seg, x="Customer_Segment", y="Total_Sales", title="Total Sales by Segment")
        st.plotly_chart(fig8, use_container_width=True)
    with c2:
        fig9 = px.bar(seg, x="Customer_Segment", y="Avg_Order", title="Average Order Value by Segment")
        st.plotly_chart(fig9, use_container_width=True)
    st.dataframe(seg, use_container_width=True)

# ---------- Tab 5: Sales Prediction ----------
with tabs[4]:
    st.subheader("Predict Sales for a New Order")
    st.write(f"Model in use: **{model_bundle['model_name']}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        segment_in = st.selectbox("Customer Segment", sorted(df["Customer_Segment"].unique()))
        country_in = st.selectbox("Country", sorted(df["Country"].unique()))
        region_in = st.selectbox("Region", sorted(df["Region"].unique()))
    with c2:
        category_in = st.selectbox("Product Category", sorted(df["Product_Category"].unique()))
        quantity_in = st.number_input("Quantity", min_value=1, max_value=50, value=2)
        unit_price_in = st.number_input("Unit Price ($)", min_value=1.0, max_value=5000.0, value=50.0)
    with c3:
        discount_in = st.slider("Discount (%)", 0, 50, 0)
        shipping_in = st.number_input("Shipping Cost ($)", min_value=0.0, max_value=200.0, value=10.0)
        dow_in = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    month_in = st.slider("Month", 1, 12, 6)
    quarter_in = (month_in - 1) // 3 + 1

    if st.button("Predict Sales", type="primary"):
        input_df = pd.DataFrame([{
            "Customer_Segment": segment_in, "Country": country_in, "Region": region_in,
            "Product_Category": category_in, "Quantity": quantity_in, "Unit_Price": unit_price_in,
            "Discount_Percent": discount_in, "Shipping_Cost": shipping_in,
            "Month": month_in, "Quarter": quarter_in, "DayOfWeek": dow_in
        }])
        pred = model.predict(input_df)[0]
        st.success(f"### Predicted Total Sales: ${pred:,.2f}")

    st.divider()
    st.subheader("Model Performance Comparison")
    metrics_df = pd.DataFrame(model_bundle["metrics"]).T
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)
