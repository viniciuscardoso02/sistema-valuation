"""
Sistema de Valuation Inteligente - Home 2 Invest
-------------------------------------------------
App Streamlit + DuckDB para análise de comparáveis imobiliários a partir da
base ITBI fatiada em arquivos base_itbi_limpa_*.parquet.

Princípio central desta versão: RESILIÊNCIA A SCHEMA DRIFT.
Nenhum trecho assume que uma coluna existe. O schema real dos parquets é lido
uma vez (DESCRIBE) e todas as consultas/transformações se adaptam a ele, de
modo que faltas de colunas (ex.: 'Ano_Construcao_Geo' no fallback textual)
nunca derrubam o app.
"""

import os
import glob
import random
import unicodedata
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import folium
from geopy.geocoders import Nominatim
import altair as alt

# --- Renderizador de mapa compatível com versões nova e antiga do streamlit-folium ---
try:
    from streamlit_folium import st_folium

    def render_map(m):
        try:
            st_folium(m, height=480, use_container_width=True, returned_objects=[])
        except TypeError:  # versões antigas sem use_container_width
            st_folium(m, width=1100, height=480, returned_objects=[])

except Exception:
    from streamlit_folium import folium_static

    def render_map(m):
        folium_static(m, width=1100, height=480)

try:
    from folium.plugins import MarkerCluster
    HAS_CLUSTER = True
except Exception:
    HAS_CLUSTER = False


# ============================================================================
# 1. CONFIGURAÇÃO E LOCALIZAÇÃO DOS DADOS
# ============================================================================
st.set_page_config(page_title="Valuation Home 2 Invest", layout="wide")
st.title("🏢 Sistema de Valuation Inteligente - Home 2 Invest")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# caminho absoluto com barras normais (funciona em Linux/Streamlit Cloud e Windows)
PARQUET_GLOB = os.path.join(APP_DIR, "base_itbi_limpa_*.parquet").replace("\\", "/")

arquivos_parquet = glob.glob(PARQUET_GLOB)
if not arquivos_parquet:  # fallback: cwd (Streamlit Cloud roda da raiz do repo)
    PARQUET_GLOB = "base_itbi_limpa_*.parquet"
    arquivos_parquet = glob.glob(PARQUET_GLOB)

if not arquivos_parquet:
    st.error("Arquivos de dados (base_itbi_limpa_*.parquet) não encontrados no repositório.")
    st.stop()

# Nomes canônicos esperados das colunas do ITBI
COL_VAL = "Valor de Transação (declarado pelo contribuinte)"
COL_AREA = "Área Construída (m2)"
COL_TERR = "Área do Terreno (m2)"
COL_ANO = "Ano_Construcao_Geo"
COL_USO = "Descrição do uso (IPTU)"
COL_LOGR = "Nome do Logradouro"
COL_NUM = "Número"
COL_BAIRRO = "Bairro"
COL_DATA = "Data de Transação"
COL_SQL = "N° do Cadastro (SQL)"


# ============================================================================
# 2. INTROSPECÇÃO DE SCHEMA  (a chave da resiliência)
# ============================================================================
@st.cache_data(show_spinner=False)
def get_available_columns(glob_path):
    """Retorna o conjunto de colunas realmente presentes nos parquets."""
    try:
        q = f"DESCRIBE SELECT * FROM read_parquet('{glob_path}', union_by_name=true)"
        return set(duckdb.query(q).df()["column_name"].tolist())
    except Exception:
        return set()


COLS = get_available_columns(PARQUET_GLOB)


def has(col):
    return col in COLS


# ============================================================================
# 3. HELPERS
# ============================================================================
def formata_moeda(valor):
    try:
        if valor is None or pd.isna(valor):
            return "-"
        return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def remover_acentos(txt):
    if txt is None or (isinstance(txt, float) and pd.isna(txt)):
        return ""
    txt = str(txt).upper().strip()
    return "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")


def extrair_palavras_chave_rua(nome_rua):
    """Extrai termos relevantes, mantendo apenas alfanuméricos (evita quebra/injeção SQL)."""
    rua_limpa = remover_acentos(nome_rua)
    termos_ignorados = {
        "RUA", "AVENIDA", "AV", "ALAMEDA", "TRAVESSA", "PRACA", "DOS", "DAS",
        "DE", "DO", "DA", "PROFESSOR", "DR", "DOUTOR",
    }
    palavras = []
    for p in rua_limpa.split():
        p = "".join(ch for ch in p if ch.isalnum())
        if len(p) > 2 and p not in termos_ignorados:
            palavras.append(p)
    return palavras


def sql_str(s):
    """Escapa aspas simples para literais SQL seguros."""
    return str(s).replace("'", "''")


def col_as_str(df, name):
    """Retorna a coluna como string; se ausente, série vazia do tamanho certo (não quebra)."""
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series([""] * len(df), index=df.index)


def to_num(df, name):
    """Garante coluna numérica; se ausente, cria como NaN (impede KeyError downstream)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


# ============================================================================
# 4. LISTA DE BAIRROS
# ============================================================================
@st.cache_data(show_spinner=False)
def carregar_lista_bairros(glob_path):
    if COL_BAIRRO not in get_available_columns(glob_path):
        return ["Selecione..."]
    try:
        q = (f'SELECT DISTINCT "{COL_BAIRRO}" AS b '
             f"FROM read_parquet('{glob_path}', union_by_name=true) "
             f'WHERE "{COL_BAIRRO}" IS NOT NULL')
        df_b = duckdb.query(q).df()
        return ["Selecione..."] + sorted(df_b["b"].astype(str).unique())
    except Exception:
        return ["Selecione..."]


bairros_disp = carregar_lista_bairros(PARQUET_GLOB)


# ============================================================================
# 5. SIDEBAR
# ============================================================================
st.sidebar.header("📍 Parâmetros de Busca")
rua = st.sidebar.text_input("Logradouro (Busca Inteligente por Raio)")
num = st.sidebar.text_input("Número (Opcional)")
raio = st.sidebar.slider("Raio de busca vizinhança (metros)", 100, 2500, 500)

st.sidebar.markdown("**OU**")
bairro_alvo = st.sidebar.selectbox("Buscar por Bairro Inteiro", bairros_disp)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros de Ativo")
tipo = st.sidebar.selectbox("Uso do Imóvel", ["Residenciais", "Apartamentos"])

ano_min, ano_max = st.sidebar.slider(
    "Ano da Transação", min_value=2010, max_value=date.today().year,
    value=(2010, date.today().year),
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações Avançadas")
remover_outliers = st.sidebar.toggle("Remover Outliers (Método IQR)", value=True)


# ============================================================================
# 6. CONSTRUÇÃO DO FILTRO DE USO (defensivo)
# ============================================================================
def build_uso_filter(tipo):
    if not has(COL_USO):
        return ""  # base sem coluna de uso -> não filtra por tipo
    if tipo == "Residenciais":
        return (f' AND (UPPER("{COL_USO}") LIKE \'%RESIDÊN%\' '
                f'OR UPPER("{COL_USO}") LIKE \'%CASA%\')')
    if tipo == "Apartamentos":
        return f' AND UPPER("{COL_USO}") LIKE \'%APARTAMENTO%\''
    return ""


condicao_extra = build_uso_filter(tipo)


# ============================================================================
# 7. EXECUÇÃO DA BUSCA
# ============================================================================
def run_query(sql):
    return duckdb.query(sql).df()


if rua or bairro_alvo != "Selecione...":
    with st.spinner("Compilando histórico e aplicando regras de inteligência imobiliária..."):

        df_bruto = pd.DataFrame()
        lat_c, lon_c = None, None

        try:
            if rua:
                # 7.1 tenta geocodificar (com user_agent randômico para mitigar 429)
                loc = None
                try:
                    rand_id = random.randint(10000, 99999)
                    geolocator = Nominatim(user_agent=f"h2i_valuation_engine_{rand_id}")
                    endereco = f"{rua}, {num}, São Paulo, SP" if num else f"{rua}, São Paulo, SP"
                    loc = geolocator.geocode(endereco, timeout=8)
                except Exception:
                    loc = None

                tem_geo = has("Latitude") and has("Longitude")

                # 7.2 busca por raio (geocodificou E base tem coordenadas)
                if loc and tem_geo:
                    lat_c, lon_c = loc.latitude, loc.longitude
                    st.success(f"📍 Endereço Alvo Localizado: **{loc.address.split(',')[0]}**")
                    query = f"""
                    WITH base_distancia AS (
                        SELECT *,
                        (6371000 * acos(LEAST(1.0, GREATEST(-1.0,
                            cos(radians({lat_c})) * cos(radians(TRY_CAST("Latitude" AS DOUBLE))) *
                            cos(radians(TRY_CAST("Longitude" AS DOUBLE)) - radians({lon_c})) +
                            sin(radians({lat_c})) * sin(radians(TRY_CAST("Latitude" AS DOUBLE)))
                        ))) AS dist_metros
                        FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
                        WHERE "Latitude" IS NOT NULL AND "Longitude" IS NOT NULL {condicao_extra}
                    )
                    SELECT * FROM base_distancia
                    WHERE dist_metros <= {raio}
                    ORDER BY dist_metros
                    """
                    df_bruto = run_query(query)

                # 7.3 fallback textual (geo falhou OU raio não trouxe nada)
                if df_bruto.empty and has(COL_LOGR):
                    palavras = extrair_palavras_chave_rua(rua)
                    if palavras:
                        if not loc:
                            st.warning("⚠️ Modo de contingência ativado: serviço de mapas "
                                       "indisponível no servidor compartilhado. Puxando "
                                       "histórico textual da rua.")
                        else:
                            st.info("ℹ️ Nenhum imóvel dentro do raio; exibindo histórico "
                                    "textual do logradouro.")
                        cond_rua = " AND ".join(
                            [f"UPPER(\"{COL_LOGR}\") LIKE '%{sql_str(p)}%'" for p in palavras]
                        )
                        query = f"""
                        SELECT * FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
                        WHERE ({cond_rua}) {condicao_extra}
                        """
                        df_bruto = run_query(query)
                    elif not has("Latitude"):
                        st.warning("Base sem coordenadas e sem termos de busca válidos no logradouro.")

            else:
                # 7.4 busca por bairro
                if has(COL_BAIRRO):
                    query = f"""
                    SELECT * FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
                    WHERE "{COL_BAIRRO}" = '{sql_str(bairro_alvo)}' {condicao_extra}
                    """
                    df_bruto = run_query(query)
                else:
                    st.warning("A base não possui coluna de Bairro.")

        except Exception as e:
            st.error(f"Erro no processamento da consulta: {e}")
            st.stop()

        # --------------------------------------------------------------------
        # 8. PROCESSAMENTO / VALUATION
        # --------------------------------------------------------------------
        if df_bruto.empty:
            st.warning("Nenhum comparável localizado para os parâmetros informados.")
            st.stop()

        df = df_bruto.copy()

        # 8.1 normalização numérica robusta (cria como NaN se a coluna faltar)
        df[COL_VAL] = to_num(df, COL_VAL)
        df[COL_AREA] = to_num(df, COL_AREA)
        df[COL_TERR] = to_num(df, COL_TERR)
        df[COL_ANO] = to_num(df, COL_ANO)            # <- garante existência; resolve o KeyError
        df["Latitude"] = to_num(df, "Latitude")
        df["Longitude"] = to_num(df, "Longitude")
        df["Ano_Transacao"] = to_num(df, "Ano_Transacao")

        # 8.2 deriva o ano da transação a partir da data, se necessário
        if df["Ano_Transacao"].isna().all() and COL_DATA in df.columns:
            anos = df[COL_DATA].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
            df["Ano_Transacao"] = pd.to_numeric(anos, errors="coerce")

        # 8.3 linhas mínimas válidas + proteção contra divisão por zero
        df = df.dropna(subset=[COL_VAL, COL_AREA])
        df = df[df[COL_AREA] > 0]
        if df.empty:
            st.warning("Nenhuma transação com valor e área construída válidos.")
            st.stop()

        # 8.4 filtro por ano da transação
        df = df[df["Ano_Transacao"].notna()]
        df = df[(df["Ano_Transacao"] >= ano_min) & (df["Ano_Transacao"] <= ano_max)]
        if df.empty:
            st.warning("Nenhuma transação dentro do período selecionado.")
            st.stop()

        # 8.5 preço por m² construído
        df["Preco_m2"] = df[COL_VAL] / df[COL_AREA]

        # 8.6 remoção de outliers (IQR) — aplicada sobre os comparáveis do período
        if remover_outliers and len(df) >= 4:
            Q1 = df["Preco_m2"].quantile(0.25)
            Q3 = df["Preco_m2"].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df = df[(df["Preco_m2"] >= lim_inf) & (df["Preco_m2"] <= lim_sup)]
        if df.empty:
            st.warning("Nenhum imóvel restou após o filtro de outliers (IQR).")
            st.stop()

        # 8.7 categorização (à prova de colunas/valores ausentes)
        chave_col = (COL_SQL if COL_SQL in df.columns
                     else (COL_LOGR if COL_LOGR in df.columns else None))
        if chave_col is None:
            df["Chave_Imovel"] = df.index.astype(str)
        else:
            df["Chave_Imovel"] = col_as_str(df, chave_col) + "_" + col_as_str(df, COL_NUM)

        grp = df.groupby("Chave_Imovel")[COL_AREA]
        df["Max_Area_Historica"] = grp.transform("max")
        df["Min_Area_Historica"] = grp.transform("min")
        df["Qtd_Areas_Unicas"] = grp.transform("nunique")
        df["Houve_Expansao"] = (
            (df["Qtd_Areas_Unicas"] > 1)
            & ((df["Max_Area_Historica"] - df["Min_Area_Historica"]) > 10)
        )

        # condlist explicitamente em ndarray booleano (resolve o "invalid entry in condlist")
        c1 = (df[COL_ANO].fillna(0) >= 2018).to_numpy(dtype=bool)
        c2_series = df["Houve_Expansao"].fillna(False) & (df[COL_AREA] == df["Max_Area_Historica"])
        c2 = c2_series.fillna(False).to_numpy(dtype=bool)
        df["Categoria"] = np.select(
            [c1, c2],
            ["Modernizado (≥ 2018)", "Retrofit (Expansão)"],
            default="Antigo",
        )

        df = df.drop(
            columns=["Max_Area_Historica", "Min_Area_Historica",
                     "Qtd_Areas_Unicas", "Houve_Expansao"],
            errors="ignore",
        )

        # --------------------------------------------------------------------
        # 9. SAÍDAS
        # --------------------------------------------------------------------
        st.markdown("### 📊 Resumo do Valuation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Amostras Resgatadas", len(df))
        m2.metric("Mediana R$/m²", formata_moeda(df["Preco_m2"].median()))
        m3.metric("Média R$/m²", formata_moeda(df["Preco_m2"].mean()))
        m4.metric(
            "Faixa R$/m²",
            f'{formata_moeda(df["Preco_m2"].min())} — {formata_moeda(df["Preco_m2"].max())}',
        )

        # 9.1 mapa
        df_geo = df.dropna(subset=["Latitude", "Longitude"])
        df_geo = df_geo[df_geo["Latitude"].between(-90, 90) & df_geo["Longitude"].between(-180, 180)]
        if not df_geo.empty:
            st.markdown("### 🗺️ Distribuição Geográfica")
            centro = [lat_c, lon_c] if (lat_c and lon_c) else \
                     [df_geo["Latitude"].mean(), df_geo["Longitude"].mean()]
            m = folium.Map(location=centro, zoom_start=15, tiles="CartoDB positron")

            if lat_c and lon_c:
                folium.Marker([lat_c, lon_c], tooltip="Endereço Alvo",
                              icon=folium.Icon(color="red", icon="star")).add_to(m)
                folium.Circle([lat_c, lon_c], radius=raio, color="#1f77b4",
                              fill=True, fill_opacity=0.05).add_to(m)

            cores = {"Modernizado (≥ 2018)": "green",
                     "Retrofit (Expansão)": "orange",
                     "Antigo": "blue"}
            camada = MarkerCluster().add_to(m) if HAS_CLUSTER else m
            for _, r in df_geo.iterrows():
                popup = (f"<b>R$/m²:</b> {formata_moeda(r['Preco_m2'])}<br>"
                         f"<b>Valor:</b> {formata_moeda(r[COL_VAL])}<br>"
                         f"<b>Área:</b> {r[COL_AREA]:.0f} m²<br>"
                         f"<b>Categoria:</b> {r['Categoria']}")
                folium.CircleMarker(
                    [r["Latitude"], r["Longitude"]], radius=5,
                    color=cores.get(r["Categoria"], "gray"),
                    fill=True, fill_opacity=0.85,
                    popup=folium.Popup(popup, max_width=260),
                ).add_to(camada)
            render_map(m)
        else:
            st.info("ℹ️ Sem coordenadas válidas nesta amostra (modo textual) — mapa "
                    "indisponível, mas o histórico abaixo é válido.")

        # 9.2 distribuição de preço/m²
        st.markdown("### 📈 Distribuição de Preço/m²")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Preco_m2:Q", bin=alt.Bin(maxbins=30), title="Preço por m² (R$)"),
                y=alt.Y("count()", title="Nº de transações"),
                tooltip=[alt.Tooltip("count()", title="Transações")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

        # 9.3 tabela de comparáveis
        st.markdown("### 📋 Transações Comparáveis")
        df_show = df.drop(columns=["Chave_Imovel"], errors="ignore").copy()
        if "dist_metros" in df_show.columns:
            df_show = df_show.sort_values("dist_metros")
        else:
            df_show = df_show.sort_values("Preco_m2")
        st.dataframe(df_show, use_container_width=True)

else:
    st.info("Informe um logradouro (busca por raio/contingência textual) ou selecione um "
            "bairro na barra lateral para iniciar a análise.")
