import streamlit as st
import pandas as pd

from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Sales Data Input",
    layout="centered"
)

# Title and description
st.title("Sales Data Input Form")
st.markdown("Enter your sales data below to preview before forecasting:")

# Create form for data input
with st.form("data_input_form"):
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        order_date = st.date_input("Order Date", value=datetime(2014, 1, 1))
    with col2:
        ship_date = st.date_input("Ship Date", value=datetime(2014, 1, 3))
    
    # Categorical inputs
    ship_mode = st.selectbox("Ship Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
    segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
    category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    sub_category = st.selectbox("Sub-Category", ["Chairs", "Tables", "Phones", "Art", "Binders"])
    
    # Numeric inputs
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        quantity = st.number_input("Quantity", min_value=1, value=5)
        discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
    with col4:
        price_per_unit = st.number_input("Price per Unit", min_value=0.01, value=49.99, step=5.0)
        profit = st.number_input("Profit", value=99.99)
    with col5:
        time_taken = st.number_input("Time Taken (days)", min_value=1, value=3)
    
    # Sales lags
    st.markdown("---")
    st.markdown("**Sales Lags (Previous Periods)**")
    lag_cols = st.columns(5)
    sales_lags = []
    for i, col in enumerate(lag_cols, start=1):
        with col:
            sales_lags.append(st.number_input(f"Lag {i}", value=1000.0/i, key=f"lag_{i}"))
    
    # Submit button
    submitted = st.form_submit_button("Preview Data")

# Display the data when submitted
if submitted:
    st.markdown("---")
    st.subheader("Data Preview")
    
    # Create single row dataframe
    data = {
        'Order Date': [order_date],
        'Ship Date': [ship_date],
        'Ship Mode': [ship_mode],
        'Segment': [segment],
        'Category': [category],
        'Sub-Category': [sub_category],
        'Quantity': [quantity],
        'Discount': [discount],
        'Price per Unit': [price_per_unit],
        'Profit': [profit],
        'Time Taken': [time_taken],
        'Sales Lag 1': [sales_lags[0]],
        'Sales Lag 2': [sales_lags[1]],
        'Sales Lag 3': [sales_lags[2]],
        'Sales Lag 4': [sales_lags[3]],
        'Sales Lag 5': [sales_lags[4]]
    }
    
    preview_df = pd.DataFrame(data)
    
    # Format display
    st.dataframe(
        preview_df.style.format({
            'Price per Unit': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Discount': '{:.1f}%',
            'Sales Lag 1': '{:,.2f}',
            'Sales Lag 2': '{:,.2f}',
            'Sales Lag 3': '{:,.2f}',
            'Sales Lag 4': '{:,.2f}',
            'Sales Lag 5': '{:,.2f}'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    # Generate 30-day table button
    if st.button("Generate 30-Day Table"):
        future_dates = pd.date_range(order_date, periods=30, freq='D')
        full_data = []
        
        for date in future_dates:
            full_data.append({
                'Order Date': date,
                'Ship Date': ship_date,
                'Ship Mode': ship_mode,
                'Segment': segment,
                'Category': category,
                'Sub-Category': sub_category,
                'Quantity': quantity,
                'Discount': discount,
                'Price per Unit': price_per_unit,
                'Profit': profit,
                'Time Taken': time_taken,
                'Sales Lag 1': sales_lags[0],
                'Sales Lag 2': sales_lags[1],
                'Sales Lag 3': sales_lags[2],
                'Sales Lag 4': sales_lags[3],
                'Sales Lag 5': sales_lags[4]
            })
        
        month_df = pd.DataFrame(full_data)
        
        st.markdown("---")
        st.subheader("30-Day Generated Data")
        st.dataframe(
            month_df.style.format({
                'Price per Unit': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.1f}%',
                'Sales Lag 1': '{:,.2f}',
                'Sales Lag 2': '{:,.2f}',
                'Sales Lag 3': '{:,.2f}',
                'Sales Lag 4': '{:,.2f}',
                'Sales Lag 5': '{:,.2f}'
            }),
            hide_index=True,
            use_container_width=True,
            height=800
        )
        
        # Download button
        csv = month_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="30_day_sales_data.csv",
            mime="text/csv"
        )