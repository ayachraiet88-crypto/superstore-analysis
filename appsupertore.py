import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Configuration de la page (DOIT être la première commande Streamlit du fichier)
st.set_page_config(page_title="SmartStore AI", page_icon="📊", layout="wide")

# ------------------------------
# Chargement des données (mis en cache pour ne pas recharger à chaque interaction)
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
    df['Is_Profitable'] = (df['Profit'] > 0).astype(int)
    return df

df = load_data()

# ------------------------------
# Menu de navigation (sidebar)
# ------------------------------
st.sidebar.title("SMARTSTORE AI")
page = st.sidebar.radio(
    "Navigation",
    [" Dashboard", " Product Analysis", "Customer Segmentation", "Sales Forecast", " Product Simulator"]
)

# ------------------------------
# Page Dashboard
# ------------------------------
if page == " Dashboard":
    st.title(" Executive Dashboard")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    col3.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col4.metric("Profit Margin", f"{df['Profit'].sum()/df['Sales'].sum()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        sales_region = df.groupby('Region')['Sales'].sum().reset_index()
        fig = px.bar(sales_region, x='Region', y='Sales', title="Sales by Region")
        st.plotly_chart(fig, width='stretch')

    with col2:
        profit_cat = df.groupby('Category')['Profit'].sum().reset_index()
        fig = px.bar(profit_cat, x='Category', y='Profit', title="Profit by Category",
                     orientation='v', color='Category')
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Évolution mensuelle des ventes et du profit")
    monthly = df.groupby(df['Order Date'].dt.to_period('M').astype(str)).agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum')
    ).reset_index().rename(columns={'Order Date': 'Month'})
    fig = px.line(monthly, x='Month', y=['Total_Sales', 'Total_Profit'], title="Sales & Profit par mois")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Top / Bottom Produits")
    product_summary = df.groupby('Product Name').agg(
        Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum')
    ).reset_index()
    st.dataframe(product_summary.sort_values('Total_Profit', ascending=False).head(10), width='stretch')

# ------------------------------
# Page Product Analysis
# ------------------------------
elif page == " Product Analysis":
    st.title(" Product Analysis")

    product_summary = df.groupby('Product Name').agg(
        Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum')
    ).reset_index()

    fig = px.scatter(
        product_summary,
        x='Total_Sales', y='Total_Profit', hover_data=['Product Name'],
        title="Sales vs Profit par Produit"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Produits (Profit)")
        st.dataframe(product_summary.sort_values('Total_Profit', ascending=False).head(10), width='stretch')

    with col2:
        st.subheader("Worst 10 Produits (Profit)")
        st.dataframe(product_summary.sort_values('Total_Profit', ascending=True).head(10), width='stretch')

    st.divider()
    st.subheader("Discount vs Profit")
    fig = px.scatter(df, x='Discount', y='Profit', opacity=0.4, title="Relation Discount → Profit")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, width='stretch')

# ------------------------------
# Page Customer Segmentation
# ------------------------------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation (RFM)")

    rfm = pd.read_csv("powerbi_exports/customer_rfm.csv")

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", f"{rfm['Customer ID'].nunique():,}")
    col2.metric("Avg Customer Value", f"${rfm['Monetary'].mean():,.0f}")

    fig = px.bar(rfm['Segment'].value_counts().reset_index(), x='Segment', y='count',
                 title="Répartition des Segments", color='Segment')
    st.plotly_chart(fig, width='stretch')

    st.subheader("Clients à risque")
    st.dataframe(
        rfm[rfm['Segment'] == 'At Risk'][['Customer_Name', 'Recency', 'Frequency', 'Monetary']],
        width='stretch'
    )

# ------------------------------
# Page Sales Forecast (Prophet)
# ------------------------------
elif page == " Sales Forecast":
    st.title("Sales Forecast")
    st.markdown("Prévision des ventes futures avec **Prophet** (modèle de séries temporelles de Meta/Facebook).")

    horizon = st.slider("Nombre de mois à prédire", min_value=1, max_value=12, value=6)

    # Préparer les données au format attendu par Prophet : colonnes 'ds' (date) et 'y' (valeur)
    monthly = df.groupby(df['Order Date'].dt.to_period('M').dt.to_timestamp())['Sales'].sum().reset_index()
    monthly.columns = ['ds', 'y']

    try:
        from prophet import Prophet

        with st.spinner("Entraînement du modèle Prophet..."):
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(monthly)

            future = model.make_future_dataframe(periods=horizon, freq='MS')
            forecast = model.predict(future)

        # Graphique : historique + prévision + intervalle de confiance
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['ds'], y=monthly['y'], mode='lines+markers', name='Historique',
                                  line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prévision',
                                  line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                                  line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                                  line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
                                  name='Intervalle de confiance'))
        fig.update_layout(title="Prévision des ventes mensuelles", xaxis_title="Date", yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')

        st.subheader(f"Détail des {horizon} prochains mois prédits")
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).copy()
        forecast_display.columns = ['Mois', 'Prévision', 'Borne basse', 'Borne haute']
        forecast_display['Mois'] = forecast_display['Mois'].dt.strftime('%Y-%m')
        for c in ['Prévision', 'Borne basse', 'Borne haute']:
            forecast_display[c] = forecast_display[c].round(0).map('${:,.0f}'.format)
        st.dataframe(forecast_display, width='stretch')

        st.subheader("Composantes du modèle (tendance & saisonnalité)")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except ImportError:
        st.error("Prophet n'est pas installé. Lance `pip install prophet` dans Anaconda Prompt puis relance l'app.")

# ------------------------------
# Page Product Simulator
# ------------------------------
elif page == " Product Simulator":
    st.title(" Product Simulator")
    st.markdown("Simule la rentabilité d'une commande avant de la valider, et teste différents niveaux de remise.")

    try:
        model = joblib.load('model_rf.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
    except FileNotFoundError:
        st.error("Fichiers modèle introuvables. Assure-toi que model_rf.pkl, scaler.pkl et model_columns.pkl "
                 "sont bien dans le même dossier que app.py (générés dans Jupyter avec joblib.dump).")
        st.stop()

    st.subheader("Paramètres de la commande")
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", sorted(df['Category'].unique()))
        sub_category = st.selectbox("Sub-Category", sorted(df[df['Category'] == category]['Sub-Category'].unique()))
    with col2:
        region = st.selectbox("Region", sorted(df['Region'].unique()))
        segment = st.selectbox("Segment", sorted(df['Segment'].unique()))
    with col3:
        ship_mode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
        quantity = st.number_input("Quantity", min_value=1, max_value=50, value=3)

    sales = st.number_input("Sales ($)", min_value=1.0, max_value=20000.0, value=500.0, step=10.0)
    discount = st.slider("Discount (%)", min_value=0, max_value=80, value=20) / 100

    def predict_profitability(cat, subcat, reg, seg, ship, qty, sal, disc):
        row = pd.DataFrame([{
            'Category': cat, 'Sub-Category': subcat, 'Region': reg, 'Segment': seg,
            'Ship Mode': ship, 'Discount': disc, 'Quantity': qty, 'Sales': sal
        }])
        row_encoded = pd.get_dummies(row, columns=['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode'])
        # Aligner les colonnes avec celles utilisées à l'entraînement (ajoute les colonnes manquantes à 0)
        row_encoded = row_encoded.reindex(columns=model_columns, fill_value=False)
        proba = model.predict_proba(row_encoded)[0][1]
        pred = model.predict(row_encoded)[0]
        return pred, proba

    if st.button(" Prédire la rentabilité", type="primary"):
        pred, proba = predict_profitability(category, sub_category, region, segment, ship_mode,
                                             quantity, sales, discount)

        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.success(f"Profitable (probabilité: {proba*100:.1f}%)")
            else:
                st.error(f" Non profitable (probabilité de perte: {(1-proba)*100:.1f}%)")
        with col2:
            estimated_profit = sales * (0.30 - discount)  # estimation simplifiée pour affichage indicatif
            st.metric("Profit estimé (indicatif)", f"${estimated_profit:,.0f}")

        st.divider()
        st.subheader("What-if Analysis — impact de la remise sur la rentabilité")
        discounts_range = np.arange(0, 0.81, 0.05)
        probas = []
        for d in discounts_range:
            _, p = predict_profitability(category, sub_category, region, segment, ship_mode,
                                          quantity, sales, d)
            probas.append(p * 100)

        fig = px.line(x=discounts_range * 100, y=probas, markers=True,
                      labels={'x': 'Discount (%)', 'y': 'Probabilité d\'être profitable (%)'},
                      title="Impact de la remise sur la probabilité de rentabilité")
        fig.add_vline(x=discount * 100, line_dash="dash", line_color="red",
                      annotation_text="Remise actuelle")
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, width='stretch')

        st.caption(" La ligne pointillée grise (50%) marque le seuil où le modèle bascule sa décision "
                   "entre 'profitable' et 'non profitable'.")