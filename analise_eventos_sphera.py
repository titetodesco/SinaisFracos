import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
from io import BytesIO

# --- CONFIGURAÇÃO
st.set_page_config(page_title="Análise de Eventos Offshore", layout="wide")

# --- FUNÇÃO PARA CARREGAR DO GITHUB
@st.cache_data(ttl=3600)
def load_data_from_github():
    url = "https://raw.githubusercontent.com/titetodesco/sphera/main/TRATADO_safeguardOffShore.xlsx"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Erro ao baixar o arquivo do GitHub!")
        return None
    return pd.read_excel(BytesIO(response.content))

# --- SIDEBAR
st.sidebar.title("⚙️ Configurações")
if st.sidebar.button("🔁 Atualizar Dados do GitHub"):
    st.cache_data.clear()

df = load_data_from_github()
if df is None:
    st.stop()

# --- FILTRO DE LOCATION
locations = sorted(df["Location"].dropna().unique())
selected_locations = st.sidebar.multiselect("Filtrar por Location", locations, default=locations)
df_filtered = df[df["Location"].isin(selected_locations)] if selected_locations else df.copy()

# --- MENU DE ANÁLISE
analises = [
    "Resumo dos Dados",
    "Heatmap Location × Risk Area",
    "Heatmap Location × Human Factor",
    "Heatmap Task × Human Factor",
    "Heatmap Risk Area × Human Factor",
    "Tendência Temporal de Eventos",
    "Top Tasks por Risk Area",
]
selected_analise = st.sidebar.radio("Escolha a análise:", analises)

# --- ÁREA PRINCIPAL

if selected_analise == "Resumo dos Dados":
    st.header("📊 Resumo dos Dados")
    st.write(f"Total de eventos únicos: {df_filtered['Event ID'].nunique():,}")
    st.write(f"Período dos eventos: {df_filtered['Date Occurred'].min().date()} até {df_filtered['Date Occurred'].max().date()}")
    st.write("Tipos de evento:")
    st.dataframe(df_filtered["Event Type"].value_counts().rename("Contagem"))

    st.subheader("Amostra dos dados")
    st.dataframe(df_filtered.head(25))

    st.subheader("Contagem por Location")
    st.bar_chart(df_filtered.groupby("Location")["Event ID"].nunique())

elif selected_analise == "Heatmap Location × Risk Area":
    st.header("📍 Heatmap Location × Risk Area")
    pivot = (
        df_filtered.drop_duplicates(subset=["Event ID", "Location", "Risk Area"])
        .pivot_table(index="Location", columns="Risk Area", values="Event ID", aggfunc="nunique", fill_value=0)
    )
    plt.figure(figsize=(12, min(8, 0.4*len(pivot))))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    plt.title("Nº de Eventos distintos por Location e Risk Area")
    st.pyplot(plt.gcf())
    plt.clf()

elif selected_analise == "Heatmap Location × Human Factor":
    st.header("📍 Heatmap Location × Human Factor")
    pivot = (
        df_filtered.drop_duplicates(subset=["Event ID", "Location", "Event: Human Factors"])
        .pivot_table(index="Location", columns="Event: Human Factors", values="Event ID", aggfunc="nunique", fill_value=0)
    )
    plt.figure(figsize=(12, min(8, 0.4*len(pivot))))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Oranges", linewidths=0.5)
    plt.title("Nº de Eventos distintos por Location e Human Factor")
    st.pyplot(plt.gcf())
    plt.clf()

elif selected_analise == "Heatmap Task × Human Factor":
    st.header("🔗 Heatmap Task × Human Factor")
    pivot = (
        df_filtered.drop_duplicates(subset=["Event ID", "Task / Activity", "Event: Human Factors"])
        .pivot_table(index="Task / Activity", columns="Event: Human Factors", values="Event ID", aggfunc="nunique", fill_value=0)
    )
    # Seleciona só os top N mais comuns para visual não ficar ilegível
    N = 20
    top_tasks = pivot.sum(axis=1).sort_values(ascending=False).head(N).index
    top_hf = pivot.sum(axis=0).sort_values(ascending=False).head(N).index
    pivot = pivot.loc[top_tasks, top_hf]
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Top Tasks × Human Factor")
    st.pyplot(plt.gcf())
    plt.clf()

elif selected_analise == "Heatmap Risk Area × Human Factor":
    st.header("🔗 Heatmap Risk Area × Human Factor")
    pivot = (
        df_filtered.drop_duplicates(subset=["Event ID", "Risk Area", "Event: Human Factors"])
        .pivot_table(index="Risk Area", columns="Event: Human Factors", values="Event ID", aggfunc="nunique", fill_value=0)
    )
    N = 20
    top_risk = pivot.sum(axis=1).sort_values(ascending=False).head(N).index
    top_hf = pivot.sum(axis=0).sort_values(ascending=False).head(N).index
    pivot = pivot.loc[top_risk, top_hf]
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Purples", linewidths=0.5)
    plt.title("Top Risk Areas × Human Factor")
    st.pyplot(plt.gcf())
    plt.clf()

elif selected_analise == "Tendência Temporal de Eventos":
    st.header("📈 Tendência Temporal dos Eventos")
    df_filtered["AnoMes"] = pd.to_datetime(df_filtered["Date Occurred"], errors="coerce").dt.to_period("M")
    tendencia = (
        df_filtered.drop_duplicates(subset=["Event ID"])
        .groupby(["AnoMes", "Event Type"])["Event ID"].count()
        .unstack(fill_value=0)
    )
    fig = px.line(tendencia, x=tendencia.index.astype(str), y=tendencia.columns, markers=True)
    fig.update_layout(title="Evolução dos eventos por tipo (incidentes, near miss etc)", xaxis_title="Ano-Mês", yaxis_title="Qtd Eventos")
    st.plotly_chart(fig, use_container_width=True)

elif selected_analise == "Top Tasks por Risk Area":
    st.header("🏆 Top Tasks por Risk Area")
    grouped = (
        df_filtered.drop_duplicates(subset=["Event ID", "Risk Area", "Task / Activity"])
        .groupby(["Risk Area", "Task / Activity"])["Event ID"].nunique()
        .reset_index(name="Qtd Eventos")
    )
    N = 12
    for risk in grouped["Risk Area"].value_counts().head(N).index:
        st.subheader(f"🔹 {risk}")
        top_tasks = (
            grouped[grouped["Risk Area"] == risk]
            .sort_values("Qtd Eventos", ascending=False)
            .head(5)
        )
        fig = px.bar(top_tasks, x="Task / Activity", y="Qtd Eventos", title=f"Top 5 Tasks em '{risk}'", text="Qtd Eventos")
        fig.update_layout(xaxis_title="Task / Activity", yaxis_title="Qtd Eventos", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Selecione uma análise no menu.")

st.caption("App por @titetodesco & ChatGPT - Última atualização: 2024-07")
