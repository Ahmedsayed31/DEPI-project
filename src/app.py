from preprocessing import read_config
from predict import predict
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

config = read_config()
# Load dataset

def load_data():
    return pd.read_csv(config['paths']['processed_data_path'])

df = load_data()

# Data transformation
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['year'] = df['Order Date'].dt.year
df['month'] = df['Order Date'].dt.month
df['day_of_week'] = df['Order Date'].dt.dayofweek
df['week_of_year'] = df['Order Date'].dt.isocalendar().week

# Sidebar navigation with icons
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Model Deployment"], 
                        format_func=lambda x: " Dashboard" if x == "Dashboard" else " Model Deployment")

# Sidebar image with better styling
st.sidebar.markdown("---")

if page == "Dashboard":
    # Dashboard Page with modern header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #3498db, #9b59b6); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">Sales & Demand Analysis Dashboard</h1>
        <p style="color: white; margin: 5px 0 0 0;">Welcome to the Sales & Demand Analysis Dashboard. 
    Here, you can explore  sales trends, profit analysis, and future demand predictions based on historical data.
    This dashboard allows you to apply various filters and visualize sales performance across different dimensions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters in sidebar with better organization
    st.sidebar.header(" Dashboard Filters")

    with st.sidebar.expander(" Date Range", expanded=True):
        date_range = st.date_input(
            "Select Date Range",
            value=[df['Order Date'].min(), df['Order Date'].max()],
            min_value=df['Order Date'].min(),
            max_value=df['Order Date'].max()
        )
    
    with st.sidebar.expander("Category Filters", expanded=True):
        cat_filter = st.selectbox(
            "Select Category",
            ['Ship Mode', 'Region', 'Category', 'City', 'Product Name', 'Segment']
        )
    
    # Apply filters
    filtered_df = df[
        (df['Order Date'] >= pd.to_datetime(date_range[0])) & 
        (df['Order Date'] <= pd.to_datetime(date_range[1]))
    ]
    if cat_filter:
        filtered_df = filtered_df[filtered_df[cat_filter].notna()]
    # Key Metrics in cards layout
    st.subheader("📊 Key Metrics")
    
    col1, col2,col3,col4  = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}", 
                 help="Total revenue in selected period")
    with col2:
        st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}",
                 help="Total profit in selected period")
    
    with col3:
        st.metric("Avg Discount", f"{filtered_df['Discount'].mean():.2f}%",
                 help="Average discount percentage")
    with col4:
        st.metric("Total Orders", filtered_df.shape[0],
                 help="Number of orders in selected period")

    # Main content area with cards
    st.markdown("---")
    
    # Row 1: Sales Trend
    with st.container():
        st.markdown("### 📈 Sales Trend Over Time")
        st.write("Observation: This chart shows the daily sales trend over the selected period. Look for patterns like seasonality, spikes, or declines that may indicate business trends or external factors affecting sales.")
        sales_trend = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
        fig1 = px.line(sales_trend, x='Order Date', y='Sales', 
                      title="Daily Sales Trend", height=400,
                      color_discrete_sequence=['#3498db'])
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                         paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        st.write("Observation: This chart tracks the relationship between monthly sales and profits. Look for months with high sales but relatively low profits, which may indicate excessive discounting or low-margin product sales. Also identify seasonal patterns that repeat annually.")

     #Monthly Sales & Profit Trend
    st.subheader("Monthly Sales & Profit Trend")

    monthly_trend = filtered_df.groupby(['year', 'month']).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    monthly_trend['Month-Year'] = pd.to_datetime(monthly_trend[['year', 'month']].assign(day=1))

    fig6 = px.line(monthly_trend, x='Month-Year', y=['Sales', 'Profit'], 
                labels={'value': 'Amount', 'Month-Year': 'Date'},
                title="Monthly Sales and Profit Trend",
                markers=True)
    st.plotly_chart(fig6, use_container_width=True)
    
 
    
    # Row 2: Category Analysis
    col1, = st.columns(1)
    with col1:
        st.markdown(f"### Sales by {cat_filter}")
        st.write(f"Observation: This breakdown shows sales distribution across different {cat_filter.lower()} categories. Identify which categories are driving the most revenue and which may need attention.")
        sales_by_cat = filtered_df.groupby(cat_filter)['Sales'].sum().reset_index()
        fig2 = px.bar(sales_by_cat, x=cat_filter, y='Sales', 
                     color=cat_filter, height=400,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    col2, = st.columns(1)
    with col2:
        st.markdown("### Top Performing Products")
        st.write("Observation: The top 10 products by sales volume. This helps identify bestsellers and opportunities for product promotion or inventory focus.")
        top_products = filtered_df.groupby('Category')['Sales'].sum().nlargest(10).reset_index()
        fig3 = px.bar(top_products, y='Sales', x='Category', height=400,
                     color='Sales', color_continuous_scale='Blues')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Visualization 4: Average Discount vs Profit by Category
    st.subheader("Average Discount vs Profit by Category")
    discount_profit = filtered_df.groupby('Category').agg({
        'Discount': 'mean',
        'Profit': 'sum'
    }).reset_index()

    fig3 = px.bar(discount_profit, x='Category', y='Profit', 
                color='Discount', color_continuous_scale='Blues',
                title="Profit by Category with Average Discount")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Row 3: Segment Analysis
    st.markdown("### 📊 Segment Performance Analysis")
    st.write("Observation: Comparing average sales and profit across customer segments reveals which segments are most valuable and where to focus marketing efforts.")
    

    # Calculate the averages
    segment_profit = filtered_df.groupby('Segment')['Profit'].mean().reset_index()
    segment_sales = filtered_df.groupby('Segment')['Sales'].mean().reset_index()
    
    # Create subplot figure
    fig_segment = make_subplots(rows=1, cols=2, subplot_titles=('Average Sales by Segment', 'Average Profit by Segment'))
    
    # Add sales trace
    fig_segment.add_trace(
        go.Bar(
            x=segment_sales['Segment'],
            y=segment_sales['Sales'],
            marker_color=px.colors.qualitative.Set2,
            name='Sales'
        ),
        row=1, col=1
    )
    
    # Add profit trace
    fig_segment.add_trace(
        go.Bar(
            x=segment_profit['Segment'],
            y=segment_profit['Profit'],
            marker_color=px.colors.qualitative.Set2,
            name='Profit'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig_segment.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_segment, use_container_width=True)
    
    # Row 4: Average Sales by Region
    st.markdown("### Average Sales by Region")
    st.write("Observation: Regional performance comparison helps identify strong and weak markets, guiding regional marketing strategies and resource allocation.")
    avg_sales_by_region = df.groupby('Region')['Sales'].mean().sort_values(ascending=False).reset_index()
    fig_avg = px.bar(avg_sales_by_region, x='Region', y='Sales',
             title='Average Sales by Region',
             text='Sales',
             color='Region')
    fig_avg.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_avg.update_layout(yaxis_title='Average Sales', xaxis_title='Region')
    st.plotly_chart(fig_avg, use_container_width=True)
    
    # Row 5: Composition Charts
    st.markdown("### Sales Composition")
    st.write("Observation: These pie charts provide a clear view of how sales are distributed across regions and product categories, highlighting dominant areas and potential imbalances.")
    
    col5, col6 = st.columns(2)
    with col5:
        fig4 = px.pie(filtered_df, names='Region', values='Sales', 
                     title="Sales by Region",
                     hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col6:
        fig5 = px.pie(filtered_df, names='Category', values='Sales', 
                     title="Sales by Product Category",
                     hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig5, use_container_width=True)
    
    # Row 6: Data Preview
    with st.expander("🔍 View Filtered Data", expanded=False):
        st.write("Observation: The raw data table allows for detailed examination of individual transactions and validation of the aggregated metrics shown in the visualizations.")
        st.dataframe (filtered_df.head(100).style.background_gradient(cmap='Blues'))

elif page == "Model Deployment":

    st.title("🤖 Sales Forecast Dashboard")

    uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.spinner("Processing and predicting..."):
            rmse, r2, fig = predict(df)

        # Metrics Cards
        col1, col2 = st.columns(2)
        col1.metric("📉 RMSE", f"{rmse:.2f}")
        col2.metric("🎯 R² Score", f"{r2:.3f}")

        # Forecast Plot
        st.plotly_chart(fig, use_container_width=True)
