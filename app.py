import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "joblib", "gdown", "scikit-learn"],
               capture_output=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); color: #e0e0e0; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-right: 1px solid #2e2e4e; }
div[data-testid="metric-container"] { background: linear-gradient(135deg, #1e1e3a, #2a2a4a); border: 1px solid #3a3a6a; border-radius: 12px; padding: 15px; box-shadow: 0 4px 15px rgba(78,78,247,0.1); }
div[data-testid="metric-container"] label { color: #9090c0 !important; font-size: 13px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 26px !important; font-weight: 700 !important; }
div[data-baseweb="select"] { background-color: #1e1e3a !important; border: 1px solid #3a3a6a !important; border-radius: 10px !important; }
div[data-baseweb="select"] * { background-color: #1e1e3a !important; color: #e0e0e0 !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] * { background-color: #1e1e3a !important; color: #e0e0e0 !important; }
.stTextInput input, .stNumberInput input { background-color: #1e1e3a !important; color: #e0e0e0 !important; border: 1px solid #3a3a6a !important; border-radius: 10px !important; }
.stButton > button { background: linear-gradient(135deg, #4e4ef7, #7b2ff7) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 12px 24px !important; font-size: 15px !important; font-weight: 600 !important; width: 100% !important; box-shadow: 0 4px 15px rgba(78,78,247,0.3) !important; }
.stTabs [data-baseweb="tab-list"] { background: #1e1e3a; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #9090c0; border-radius: 8px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #4e4ef7, #7b2ff7) !important; color: white !important; }
label { color: #b0b0d0 !important; font-weight: 600 !important; font-size: 14px !important; }
h1, h2, h3 { color: white !important; }
hr { border-color: #2e2e4e !important; }
.info-card { background: linear-gradient(135deg, #1e1e3a, #2a2a4a); border: 1px solid #3a3a6a; border-radius: 12px; padding: 16px; margin: 8px 0; border-left: 4px solid #4e4ef7; }
.segment-card { padding: 28px; border-radius: 16px; text-align: center; margin-top: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    if not os.path.exists('online_retail_cleaned.csv'):
        with st.spinner('📥 Downloading dataset from Google Drive... please wait'):
            file_id = "15_VsUgce27UDzuylpdRFFf4WubYiqLyT"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, 'online_retail_cleaned.csv', quiet=False)

    df = pd.read_csv('online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    with st.spinner('🔄 Building RFM model...'):
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        cluster_rfm = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
        high_value = cluster_rfm['Monetary'].idxmax()
        at_risk = cluster_rfm['Recency'].idxmax()
        regular_clus = cluster_rfm['Frequency'].idxmax()
        def label_cluster(c):
            if c == high_value: return 'High-Value'
            elif c == at_risk: return 'At-Risk'
            elif c == regular_clus: return 'Regular'
            else: return 'Occasional'
        rfm['Segment'] = rfm['Cluster'].apply(label_cluster)

    with st.spinner('🔄 Building recommendation engine...'):
        customer_product = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)
        product_matrix = customer_product.T
        cosine_sim = cosine_similarity(product_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=product_matrix.index, columns=product_matrix.index)
        product_list = list(cosine_sim_df.index)

    return df, rfm, scaler, kmeans, cosine_sim_df, product_list


df, rfm, scaler, kmeans, cosine_sim_df, product_list = load_all()

with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    st.metric("Total Transactions", f"{df['InvoiceNo'].nunique():,}")
    st.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    st.metric("Total Products", f"{df['Description'].nunique():,}")
    st.markdown("---")
    st.markdown("### 🌍 Filter by Country")
    countries = ["All"] + sorted(df['Country'].unique().tolist())
    selected_country = st.selectbox("Select Country", countries)
    st.markdown("---")
    st.markdown("### 📅 Filter by Date Range")
    min_date = df['InvoiceDate'].min().date()
    max_date = df['InvoiceDate'].max().date()
    date_range = st.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

filtered_df = df.copy()
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['InvoiceDate'].dt.date >= date_range[0]) & (filtered_df['InvoiceDate'].dt.date <= date_range[1])]

st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:42px; font-weight:700; background: linear-gradient(135deg, #4e4ef7, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🛒 Shopper Spectrum</h1>
    <p style='color:#9090c0; font-size:17px; margin-top:-10px;'>Customer Segmentation & Product Recommendation System</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("🧑 Customers", f"{filtered_df['CustomerID'].nunique():,}")
k2.metric("🧾 Transactions", f"{filtered_df['InvoiceNo'].nunique():,}")
k3.metric("💰 Revenue", f"£{filtered_df['TotalPrice'].sum():,.0f}")
k4.metric("📦 Products Sold", f"{filtered_df['Quantity'].sum():,.0f}")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Product Recommendation","👤 Customer Segmentation","📊 EDA Dashboard","🔍 Customer Explorer"])

with tab1:
    st.header("🎯 Product Recommendation Engine")
    st.markdown("Find products frequently bought together using collaborative filtering.")
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔎 Search for a Product")
        search_term = st.text_input("Type to filter products:", placeholder="e.g. HEART")
        filtered_products = [p for p in sorted(product_list) if search_term.upper() in p.upper()] if search_term else sorted(product_list)
        selected_product = st.selectbox("Select Product:", options=filtered_products if filtered_products else sorted(product_list))
        top_n = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)
        if st.button("🔍 Get Recommendations"):
            if selected_product in cosine_sim_df.index:
                sim_scores = cosine_sim_df[selected_product].sort_values(ascending=False)
                sim_scores = sim_scores.drop(selected_product).head(top_n)
                st.session_state['recommendations'] = sim_scores
                st.session_state['rec_product'] = selected_product
    with col2:
        st.subheader("✨ Recommended Products")
        if 'recommendations' in st.session_state:
            st.success(f"Top {len(st.session_state['recommendations'])} products similar to:")
            st.markdown(f"**📦 {st.session_state['rec_product']}**")
            for i, (prod, score) in enumerate(st.session_state['recommendations'].items(), 1):
                bar_width = int(score * 100)
                color = "#4e4ef7" if i == 1 else "#7b2ff7" if i == 2 else "#a855f7"
                st.markdown(f"""<div class='info-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div><span style='color:#4e4ef7; font-weight:700; font-size:18px;'>#{i}</span>
                        <span style='color:white; font-weight:600; margin-left:10px;'>{prod}</span></div>
                        <span style='color:#9090c0; font-size:13px;'>{score:.0%} match</span>
                    </div>
                    <div style='margin-top:8px; background:#2a2a4a; border-radius:6px; height:6px;'>
                        <div style='width:{bar_width}%; background:linear-gradient(90deg,{color},#a855f7); border-radius:6px; height:6px;'></div>
                    </div></div>""", unsafe_allow_html=True)
        else:
            st.info("👈 Select a product and click **Get Recommendations**.")

with tab2:
    st.header("👤 Customer Segmentation Predictor")
    st.markdown("Enter customer behaviour values to predict their segment.")
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📥 Enter Customer Data")
        recency = st.slider("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
        frequency = st.slider("Frequency (number of purchases)", min_value=1, max_value=200, value=5)
        monetary = st.slider("Monetary (total spend £)", min_value=0, max_value=50000, value=500, step=50)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recency", f"{recency}d")
        c2.metric("Frequency", f"{frequency}x")
        c3.metric("Monetary", f"£{monetary:,}")
        predict_btn = st.button("🔮 Predict Customer Segment")
    with col2:
        st.subheader("🎯 Prediction Result")
        if predict_btn:
            input_scaled = scaler.transform(np.array([[recency, frequency, monetary]]))
            cluster = kmeans.predict(input_scaled)[0]
            cluster_rfm = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
            high_value = cluster_rfm['Monetary'].idxmax()
            at_risk = cluster_rfm['Recency'].idxmax()
            regular_clus = cluster_rfm['Frequency'].idxmax()
            if cluster == high_value:
                label, color, desc = "💰 High-Value Customer", "linear-gradient(135deg,#f5a623,#f7971e)", "Top spender who shops frequently and recently!"
                tips = ["🎁 Offer VIP loyalty rewards","📧 Send personalised offers","🌟 Invite to exclusive events"]
            elif cluster == at_risk:
                label, color, desc = "⚠️ At-Risk Customer", "linear-gradient(135deg,#ff4b4b,#c0392b)", "Hasn't purchased in a long time. Needs win-back campaign!"
                tips = ["📩 Send a win-back email","💸 Offer a discount coupon","📞 Personal outreach"]
            elif cluster == regular_clus:
                label, color, desc = "✅ Regular Customer", "linear-gradient(135deg,#2ecc71,#27ae60)", "Buys regularly but not yet a top spender!"
                tips = ["📦 Upsell premium products","🔔 Send restock notifications","⭐ Encourage reviews"]
            else:
                label, color, desc = "🔵 Occasional Customer", "linear-gradient(135deg,#3498db,#2980b9)", "Shops rarely. Needs engagement campaigns!"
                tips = ["🎯 Re-engagement ads","🛒 Abandoned cart reminders","🎉 Seasonal promotions"]
            st.markdown(f"""<div class='segment-card' style='background:{color};'>
                <h2 style='color:white; font-size:26px; margin:0;'>{label}</h2>
                <p style='color:rgba(255,255,255,0.9); font-size:15px; margin-top:10px;'>{desc}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("#### 💡 Recommended Actions:")
            for tip in tips:
                st.markdown(f"<div class='info-card' style='border-left-color:#a855f7;'><span style='color:white;'>{tip}</span></div>", unsafe_allow_html=True)
        else:
            st.info("👈 Use the sliders and click **Predict Customer Segment**.")

with tab3:
    st.header("📊 EDA Dashboard")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 Top Countries by Transactions")
        n = st.slider("Top N countries", 5, 20, 10, key="cs")
        tc = df['Country'].value_counts().head(n)
        fig, ax = plt.subplots(figsize=(7,4), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        ax.barh(tc.index[::-1], tc.values[::-1], color=plt.cm.Blues(np.linspace(0.4,0.9,len(tc))))
        ax.tick_params(colors='white', labelsize=9)
        ax.set_xlabel('Transactions', color='white')
        ax.spines[['top','right','left','bottom']].set_color('#2e2e4e')
        plt.tight_layout(); st.pyplot(fig)
    with col2:
        st.subheader("🏆 Top Selling Products")
        n2 = st.slider("Top N products", 5, 20, 10, key="ps")
        tp = filtered_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(n2)
        fig, ax = plt.subplots(figsize=(7,4), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        ax.barh(tp.index[::-1], tp.values[::-1], color=plt.cm.Greens(np.linspace(0.4,0.9,len(tp))))
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Quantity Sold', color='white')
        ax.spines[['top','right','left','bottom']].set_color('#2e2e4e')
        plt.tight_layout(); st.pyplot(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📈 Monthly Sales Trend")
        fd = filtered_df.copy()
        fd['Month'] = fd['InvoiceDate'].dt.to_period('M').astype(str)
        monthly = fd.groupby('Month')['TotalPrice'].sum()
        fig, ax = plt.subplots(figsize=(7,4), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        ax.fill_between(range(len(monthly)), monthly.values, alpha=0.3, color='#4e4ef7')
        ax.plot(range(len(monthly)), monthly.values, color='#4e4ef7', marker='o', linewidth=2, markersize=4)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=7, color='white')
        ax.tick_params(colors='white'); ax.set_ylabel('Revenue (£)', color='white')
        ax.spines[['top','right','left','bottom']].set_color('#2e2e4e')
        plt.tight_layout(); st.pyplot(fig)
    with col4:
        st.subheader("🍩 Segment Distribution")
        seg_counts = rfm['Segment'].value_counts()
        colors_pie = ['#f5a623','#2ecc71','#3498db','#ff4b4b']
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        wedges, texts, autotexts = ax.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%', colors=colors_pie[:len(seg_counts)], startangle=90, wedgeprops={'edgecolor':'#1e1e2e','linewidth':2})
        for t in texts: t.set_color('white')
        for t in autotexts: t.set_color('white'); t.set_fontsize(10)
        plt.tight_layout(); st.pyplot(fig)
    st.subheader("📉 RFM Distributions")
    fig, axes = plt.subplots(1,3, figsize=(14,3), facecolor='#1e1e2e')
    for ax, col, color in zip(axes, ['Recency','Frequency','Monetary'], ['#4e4ef7','#2ecc71','#f5a623']):
        ax.set_facecolor('#1e1e2e')
        ax.hist(rfm[col], bins=40, color=color, alpha=0.8, edgecolor='none')
        ax.set_title(col, color='white', fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[['top','right','left','bottom']].set_color('#2e2e4e')
    plt.tight_layout(); st.pyplot(fig)
    st.subheader("📋 Key Business Metrics")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    m2.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    m3.metric("Total Transactions", f"{df['InvoiceNo'].nunique():,}")
    m4.metric("Total Products", f"{df['Description'].nunique():,}")

with tab4:
    st.header("🔍 Customer Explorer")
    st.markdown("---")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("🔎 Search Customer")
        customer_ids = sorted(df['CustomerID'].unique().tolist())
        selected_customer = st.selectbox("Select Customer ID:", customer_ids)
        if st.button("🔍 Explore Customer"):
            st.session_state['selected_customer'] = selected_customer
    with col2:
        st.subheader("📋 Customer Profile")
        if 'selected_customer' in st.session_state:
            cust_id = st.session_state['selected_customer']
            cust_df = df[df['CustomerID'] == cust_id]
            ref = df['InvoiceDate'].max() + pd.Timedelta(days=1)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Customer ID", str(cust_id))
            m2.metric("Recency", f"{(ref - cust_df['InvoiceDate'].max()).days} days")
            m3.metric("Frequency", f"{cust_df['InvoiceNo'].nunique()} orders")
            m4.metric("Total Spent", f"£{cust_df['TotalPrice'].sum():,.2f}")
            if cust_id in rfm['CustomerID'].values:
                seg = rfm[rfm['CustomerID'] == cust_id]['Segment'].values[0]
                seg_colors = {'High-Value':'#f5a623','Regular':'#2ecc71','Occasional':'#3498db','At-Risk':'#ff4b4b'}
                color = seg_colors.get(str(seg), '#4e4ef7')
                st.markdown(f"<div style='background:{color}; padding:12px 20px; border-radius:10px; display:inline-block; margin:10px 0;'><b style='color:white; font-size:16px;'>Segment: {seg}</b></div>", unsafe_allow_html=True)
            st.markdown("**🛍️ Recent Purchases:**")
            st.dataframe(cust_df[['InvoiceDate','Description','Quantity','UnitPrice','TotalPrice']].sort_values('InvoiceDate', ascending=False).head(10).reset_index(drop=True), use_container_width=True)
        else:
            st.info("👈 Select a Customer ID and click **Explore Customer**.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#5050a0; font-size:13px;'>🛒 Shopper Spectrum &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; Customer Segmentation & Product Recommendation</p>", unsafe_allow_html=True)
