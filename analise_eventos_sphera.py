import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.express as px
except ImportError:
    import sys
    sys.exit("O pacote plotly não está instalado. Corrija o requirements e faça o deploy novamente.")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import requests
import ast

# Defina a sua senha secreta aqui
PASSWORD = "cdshell"  # Troque por uma senha forte

def check_password():
    """Exibe um campo de senha e retorna True se a senha estiver correta."""
    st.sidebar.header("🔒 Área protegida")
    password = st.sidebar.text_input("Digite a senha para acessar o app:", type="password")
    if password == PASSWORD:
        return True
    elif password:
        st.sidebar.error("Senha incorreta. Tente novamente.")
        return False
    else:
        return False

if not check_password():
    st.stop()  # Interrompe o app até digitar a senha correta


st.set_page_config(layout="wide")
st.title("🔎 Detecção de Sinais Fracos em Eventos Offshore")

# URLs dos arquivos no GitHub
URL_EVENTOS = "https://raw.githubusercontent.com/titetodesco/SinaisFracos/main/TRATADO_safeguardOffShore.xlsx"
URL_WEAK = "https://raw.githubusercontent.com/titetodesco/SinaisFracos/main/DicionarioWaekSignals.xlsx"

def carregar_excel_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_excel(BytesIO(r.content))

@st.cache_data
def load_data():
    df_eventos = carregar_excel_url(URL_EVENTOS)
    df_dict    = carregar_excel_url(URL_WEAK)
    return df_eventos, df_dict

if st.button("🔄 Atualizar dados do GitHub"):
    st.cache_data.clear()

df_eventos, df_dict = load_data()

# Ajuste de nomes se necessário:
if "eventoID" in df_eventos.columns and "Event ID" not in df_eventos.columns:
    df_eventos = df_eventos.rename(columns={"eventoID": "Event ID"})

# Conversão robusta de data
if "Date Occurred" in df_eventos.columns:
    df_eventos["Date Occurred"] = pd.to_datetime(df_eventos["Date Occurred"], errors="coerce", infer_datetime_format=True)
    if df_eventos["Date Occurred"].isna().mean() > 0.9:
        df_eventos["Date Occurred"] = pd.to_datetime(df_eventos["Date Occurred"], errors="coerce", dayfirst=True, infer_datetime_format=True)

# Sidebar
modo = st.sidebar.selectbox("Modo de Detecção", ["Embeddings", "Fuzzy"])
thresh = st.sidebar.slider("Threshold de Similaridade", 0.3, 0.95, 0.5, 0.01)

# REMOVE DUPLICIDADE: considera apenas 1 linha por evento para a análise (usando Event ID)
df_unicos = df_eventos.drop_duplicates(subset=["Event ID"]).copy()

# --- DETECÇÃO DE WEAK SIGNALS ---
if modo == "Embeddings":
    st.subheader("🧠 Modo Embeddings")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    if "embedding" in df_unicos.columns:
        st.success("Embeddings pré‑calculados detectados.")
        emb_eventos = df_unicos["embedding"].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x).tolist()
    else:
        st.info("Gerando embeddings das descrições…")
        emb_eventos = modelo.encode(df_unicos["Description"].astype(str).tolist(), show_progress_bar=True)
        df_unicos["embedding"] = [e.tolist() for e in emb_eventos]
    emb_dict = modelo.encode(df_dict["Termo (EN)"].astype(str).tolist(), show_progress_bar=False)
    resultados = []
    for emb in emb_eventos:
        sim = cosine_similarity([emb], emb_dict)[0]
        hits = [df_dict["Termo (EN)"].iloc[i] for i, s in enumerate(sim) if s >= thresh]
        resultados.append("; ".join(hits))
else:
    from rapidfuzz import fuzz
    st.subheader("🔍 Modo Fuzzy Matching")
    terms = df_dict["Termo (EN)"].astype(str).tolist()
    resultados = []
    for desc in df_unicos["Description"].astype(str):
        hits = [t for t in terms if fuzz.partial_ratio(t.lower(), desc.lower()) >= int(thresh*100)]
        resultados.append("; ".join(hits))

df_unicos["Weak Signals Found"] = resultados

# ---- Explode para análises ----
df_exp = df_unicos.copy()
df_exp["Weak Signals Found"] = df_exp["Weak Signals Found"].fillna("")
df_exp["Weak Signals Found"] = df_exp["Weak Signals Found"].apply(lambda x: [s.strip() for s in x.split(";") if s.strip()])
df_exp = df_exp.explode("Weak Signals Found")

# --- Filtros ---
locs  = df_exp["Location"].dropna().unique().tolist()
risks = df_exp["Risk Area"].dropna().unique().tolist()
sel_loc  = st.multiselect("Location", locs, default=locs)
sel_risk = st.multiselect("Risk Area", risks, default=risks)
df_fil = df_exp[df_exp["Location"].isin(sel_loc) & df_exp["Risk Area"].isin(sel_risk)]

st.markdown("## 📊 Análises Visuais")

# --- Frequência Geral ---
freq = df_fil["Weak Signals Found"].value_counts().reset_index()
freq.columns = ["Weak Signal", "Freq"]
st.plotly_chart(px.bar(freq, x="Weak Signal", y="Freq", text="Freq", title="Frequência Geral de Weak Signals"), use_container_width=True)

# --- Frequência por Location ---
st.plotly_chart(
    px.histogram(df_fil, y="Location", color="Weak Signals Found", title="Frequência de Weak Signals por Location", histfunc="count", height=500),
    use_container_width=True
)

# --- Heatmap Risk Area x Weak Signal ---
if not df_fil.empty and "Risk Area" in df_fil.columns:
    pivot = pd.pivot_table(df_fil, index="Risk Area", columns="Weak Signals Found", aggfunc="size", fill_value=0)
    st.plotly_chart(px.imshow(pivot, text_auto=True, aspect="auto", title="Heatmap: Risk Area × Weak Signal"), use_container_width=True)

# --- Dispersão Temporal ---
if "Date Occurred" in df_fil.columns:
    df_t = df_fil.dropna(subset=["Date Occurred"])
    if not df_t.empty:
        fig_time = px.scatter(
            df_t,
            x="Date Occurred",
            y=df_t["Weak Signals Found"].astype(str),
            color="Location",
            title="🕒 Dispersão Temporal dos Weak Signals",
            height=450
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.plotly_chart(
            px.histogram(
                df_t,
                x="Location",
                color="Weak Signals Found",
                histfunc="count",
                title="📍 Weak Signals por Location ao Longo do Tempo",
                height=450
            ),
            use_container_width=True
        )
    else:
        st.info("Sem datas válidas para o gráfico temporal.")

# --- Download do resultado ---
st.markdown("### ⬇️ Baixar planilha de resultados")
buffer = BytesIO()
df_unicos.to_excel(buffer, index=False, engine="openpyxl")
st.download_button("📥 Download XLSX", data=buffer.getvalue(), file_name="eventos_weak_signals.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
