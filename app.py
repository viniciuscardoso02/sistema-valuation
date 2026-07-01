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

try:
    from folium.plugins import HeatMap
    HAS_HEATMAP = True
except Exception:
    HAS_HEATMAP = False


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
COL_DISTRITO = "Distrito"
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


def coord_sql(col):
    """
    Expressão SQL que converte uma coluna de coordenada para DOUBLE de forma
    blindada, INDEPENDENTE de como ela veio do parquet (DOUBLE já limpo, ou
    VARCHAR com vírgula decimal '-23,55'). Faz: CAST p/ texto -> troca vírgula
    por ponto -> TRY_CAST p/ DOUBLE. Coordenadas não têm separador de milhar,
    então trocar só a vírgula é seguro. Isto torna a busca por raio imune a
    fatias do ETL que não tiveram a vírgula corrigida.
    """
    return f"TRY_CAST(REPLACE(CAST(\"{col}\" AS VARCHAR), ',', '.') AS DOUBLE)"


def get_coord(df, name, parsed_name):
    """Coordenada numérica no lado Python: usa a coluna já parseada pelo SQL
    (_lat/_lon) se existir; senão converte a original tratando a vírgula."""
    if parsed_name in df.columns:
        return pd.to_numeric(df[parsed_name], errors="coerce")
    if name in df.columns:
        s = df[name].astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


# ============================================================================
# 4. LISTA DE DISTRITOS
# ============================================================================
@st.cache_data(show_spinner=False)
def carregar_lista_distritos(glob_path):
    if COL_DISTRITO not in get_available_columns(glob_path):
        return ["Selecione..."]
    try:
        q = (f'SELECT DISTINCT "{COL_DISTRITO}" AS d '
             f"FROM read_parquet('{glob_path}', union_by_name=true) "
             f'WHERE "{COL_DISTRITO}" IS NOT NULL')
        df_d = duckdb.query(q).df()
        return ["Selecione..."] + sorted(df_d["d"].astype(str).unique())
    except Exception:
        return ["Selecione..."]


distritos_disp = carregar_lista_distritos(PARQUET_GLOB)


# Caminho do GeoJSON de distritos (para desenhar o contorno no mapa)
GEOJSON_DISTRITOS = os.path.join(APP_DIR, "distritos_sp.geojson")
if not os.path.exists(GEOJSON_DISTRITOS):
    GEOJSON_DISTRITOS = "distritos_sp.geojson"


@st.cache_data(show_spinner=False)
def carregar_geojson_distritos(caminho):
    """Carrega o GeoJSON dos distritos uma vez. Retorna dict {nome: feature} e o
    FeatureCollection completo. Se o arquivo não existir, retorna (None, None)."""
    try:
        import json
        with open(caminho, "r", encoding="utf-8") as fh:
            fc = json.load(fh)
        por_nome = {}
        for feat in fc.get("features", []):
            nome = feat.get("properties", {}).get("Distrito")
            if nome:
                por_nome[str(nome)] = feat
        return por_nome, fc
    except Exception:
        return None, None


DISTRITOS_GEO, _ = carregar_geojson_distritos(GEOJSON_DISTRITOS)


# --- Zoneamento (LPUOS 2016) buscado sob demanda por área, direto do GeoSampa ---
WFS_GEOSAMPA = "http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
CAMADA_ZONEAMENTO = "geoportal:zoneamento_2016_map1"
CAMPO_ZONA = "cd_zoneamento_perimetro"


def _cor_zona(sigla):
    """Cor estável por família de zona (ZER, ZM, ZEIS, ZEU, ZC, ZPI...)."""
    s = (sigla or "").upper()
    if s.startswith("ZEIS"):
        return "#e6550d"   # habitação de interesse social
    if s.startswith("ZER"):
        return "#31a354"   # exclusivamente residencial
    if s.startswith("ZEU") or s.startswith("ZEM"):
        return "#756bb1"   # eixos de estruturação (adensamento)
    if s.startswith("ZC"):
        return "#3182bd"   # centralidades
    if s.startswith("ZM"):
        return "#f2c744"   # mista
    if s.startswith("ZPI") or s.startswith("ZDE"):
        return "#969696"   # predominantemente industrial / desenvolvimento
    if "PRAÇA" in s or "CANTEIRO" in s or s.startswith("ZEP"):
        return "#a1d99b"   # verde / praças / proteção ambiental
    return "#bdbdbd"       # demais


@st.cache_data(show_spinner=False, ttl=3600)
def buscar_zoneamento_bbox(min_lon, min_lat, max_lon, max_lat):
    """Baixa do WFS apenas os polígonos de zona que intersectam o bbox informado.
    Retorna um GeoJSON (dict) ou None. Cacheado por 1h para não repetir chamadas."""
    import requests
    # BBOX no WFS 2.0.0 com EPSG:4326 usa ordem lat,lon (min,max)
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon},urn:ogc:def:crs:EPSG::4326"
    try:
        r = requests.get(WFS_GEOSAMPA, params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": CAMADA_ZONEAMENTO, "outputFormat": "application/json",
            "srsName": "EPSG:4326", "bbox": bbox, "count": 4000,
        }, timeout=60)
        return r.json()
    except Exception:
        return None


def bbox_de_feature(feature):
    """Retorna (min_lon, min_lat, max_lon, max_lat) de um polígono/multipolígono."""
    xs, ys = [], []
    geom = feature["geometry"]
    def _walk(coords):
        for c in coords:
            if isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                _walk(c)
    _walk(geom["coordinates"])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def centroide_distrito(feature):
    """Centro aproximado de um polígono (média dos vértices), sem depender de libs geo."""
    try:
        coords = []
        geom = feature["geometry"]
        partes = geom["coordinates"]
        # MultiPolygon -> lista de polígonos; Polygon -> lista de anéis
        anel_iter = partes if geom["type"] == "MultiPolygon" else [partes]
        for poly in anel_iter:
            ext = poly[0] if geom["type"] == "MultiPolygon" else poly
            for ring in ([ext] if geom["type"] == "MultiPolygon" else partes):
                for x, y in ring:
                    coords.append((x, y))
        if not coords:
            return None
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return [lat, lon]
    except Exception:
        return None


# ============================================================================
# 5. SIDEBAR
# ============================================================================
st.sidebar.header("📍 Parâmetros de Busca")
rua = st.sidebar.text_input("Logradouro (Busca Inteligente por Raio)")
num = st.sidebar.text_input("Número (Opcional)")
raio = st.sidebar.slider("Raio de busca vizinhança (metros)", 100, 2500, 500)

st.sidebar.markdown("**OU**")
distrito_alvo = st.sidebar.selectbox("Buscar por Distrito Inteiro", distritos_disp)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros de Ativo")
tipo = st.sidebar.selectbox("Uso do Imóvel", ["Residenciais", "Apartamentos"])

ano_min, ano_max = st.sidebar.slider(
    "Ano da Transação", min_value=2010, max_value=date.today().year,
    value=(2010, date.today().year),
)

st.sidebar.markdown("**Filtro por área (alvo ± margem)**")
MARGEM = 0.20  # ±20% em torno do valor digitado
area_constr_alvo = st.sidebar.number_input(
    "Área construída alvo (m²) — 0 = sem filtro", min_value=0, value=0, step=10,
)
area_terr_alvo = st.sidebar.number_input(
    "Área de terreno alvo (m²) — 0 = sem filtro", min_value=0, value=0, step=10,
)
if area_constr_alvo > 0:
    st.sidebar.caption(f"↳ Construída: {area_constr_alvo*(1-MARGEM):.0f}–"
                       f"{area_constr_alvo*(1+MARGEM):.0f} m²")
if area_terr_alvo > 0:
    st.sidebar.caption(f"↳ Terreno: {area_terr_alvo*(1-MARGEM):.0f}–"
                       f"{area_terr_alvo*(1+MARGEM):.0f} m²")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações Avançadas")
remover_outliers = st.sidebar.toggle("Remover Outliers (Método IQR)", value=True)
filtro_status = st.sidebar.radio(
    "Mostrar imóveis",
    ["Todos", "Só modernizados", "Só antigos"],
    help=("Modernizado = ano de construção ≥ 2018 OU ampliação de área no mesmo "
          "imóvel. Filtra as transações exibidas em todo o relatório."),
)

st.sidebar.markdown("---")
st.sidebar.header("🔥 Mapa de Calor")
heatmap_modo = st.sidebar.radio(
    "Camada de calor no mapa",
    ["Desligado", "Densidade de transações", "Preço/m² de terreno"],
    help=("Densidade: regiões com mais transações ficam quentes. "
          "Preço/m² de terreno: regiões mais caras ficam quentes (ponderado pelo valor)."),
)

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Zoneamento")
mostrar_zoneamento = st.sidebar.toggle(
    "Mostrar zoneamento (LPUOS 2016)", value=False,
    help=("Desenha as zonas de uso do solo (ZER, ZM, ZEIS, ZEU...) sobre o mapa "
          "do distrito. As zonas são buscadas na hora do GeoSampa, só para a "
          "área visível."),
)


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


if rua or distrito_alvo != "Selecione...":
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

                    lat_e = coord_sql("Latitude")
                    lon_e = coord_sql("Longitude")
                    dist_expr = (
                        f"6371000 * acos(LEAST(1.0, GREATEST(-1.0, "
                        f"cos(radians({lat_c})) * cos(radians(_lat)) * "
                        f"cos(radians(_lon) - radians({lon_c})) + "
                        f"sin(radians({lat_c})) * sin(radians(_lat)))))"
                    )

                    # --- Diagnóstico do funil de coordenadas (por que sobram poucas?) ---
                    uso_bool = condicao_extra.strip()
                    uso_bool = uso_bool[4:] if uso_bool.startswith("AND ") else uso_bool
                    uso_bool = uso_bool if uso_bool else "TRUE"
                    try:
                        q_diag = f"""
                        WITH parsed AS (
                            SELECT "Latitude" AS lat_raw, "Longitude" AS lon_raw,
                                   {lat_e} AS _lat, {lon_e} AS _lon
                            FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
                        )
                        SELECT
                          COUNT(*) AS total_geral,
                          COUNT(*) FILTER (WHERE {uso_bool}) AS total_uso,
                          COUNT(*) FILTER (WHERE {uso_bool} AND lat_raw IS NOT NULL AND lon_raw IS NOT NULL) AS coord_preenchida,
                          COUNT(*) FILTER (WHERE {uso_bool} AND _lat IS NOT NULL AND _lon IS NOT NULL) AS coord_valida,
                          COUNT(*) FILTER (WHERE {uso_bool} AND _lat IS NOT NULL AND _lon IS NOT NULL AND {dist_expr} <= {raio}) AS dentro_raio
                        FROM parsed
                        """
                        diag = duckdb.query(q_diag).df().iloc[0]
                    except Exception:
                        diag = None

                    if diag is not None:
                        with st.expander("🔍 Diagnóstico de cobertura de coordenadas", expanded=False):
                            d1, d2, d3, d4, d5 = st.columns(5)
                            d1.metric("Total (uso)", int(diag["total_uso"]))
                            d2.metric("Coord. preenchida", int(diag["coord_preenchida"]))
                            d3.metric("Coord. válida", int(diag["coord_valida"]))
                            d4.metric("Dentro do raio", int(diag["dentro_raio"]))
                            d5.metric("Base inteira", int(diag["total_geral"]))
                            preench = int(diag["coord_preenchida"])
                            valida = int(diag["coord_valida"])
                            uso = int(diag["total_uso"])
                            if preench > 0 and valida < preench * 0.9:
                                st.warning(f"⚠️ {preench - valida} linhas têm coordenada preenchida "
                                           f"mas **inválida** (provável vírgula decimal não convertida no "
                                           f"ETL). A conversão blindada desta versão já as recupera no cálculo.")
                            if uso > 0 and preench < uso * 0.5:
                                st.warning(f"⚠️ Apenas {preench} de {uso} transações têm coordenada "
                                           f"preenchida ({preench/uso*100:.0f}%). Isto é **cobertura "
                                           f"incompleta de geocodificação no ETL** — nenhuma busca por "
                                           f"raio recupera linhas sem coordenada.")

                    query = f"""
                    WITH parsed AS (
                        SELECT *,
                               {lat_e} AS _lat,
                               {lon_e} AS _lon
                        FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
                    ),
                    base_distancia AS (
                        SELECT *, ({dist_expr}) AS dist_metros
                        FROM parsed
                        WHERE _lat IS NOT NULL AND _lon IS NOT NULL {condicao_extra}
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
                # 7.4 busca por distrito
                if has(COL_DISTRITO):
                    query = f"""
                    SELECT * FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
                    WHERE "{COL_DISTRITO}" = '{sql_str(distrito_alvo)}' {condicao_extra}
                    """
                    df_bruto = run_query(query)
                else:
                    st.warning("A base não possui coluna de Distrito.")

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
        df["Latitude"] = get_coord(df, "Latitude", "_lat")
        df["Longitude"] = get_coord(df, "Longitude", "_lon")
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

        # 8.4b filtro por área (alvo ± margem de 20%) — construída e/ou terreno
        if area_constr_alvo > 0:
            lo_c, hi_c = area_constr_alvo * (1 - MARGEM), area_constr_alvo * (1 + MARGEM)
            df = df[df[COL_AREA].between(lo_c, hi_c)]
        if area_terr_alvo > 0 and COL_TERR in df.columns:
            lo_t, hi_t = area_terr_alvo * (1 - MARGEM), area_terr_alvo * (1 + MARGEM)
            df = df[df[COL_TERR].between(lo_t, hi_t)]
        if df.empty:
            st.warning("Nenhuma transação dentro das faixas de área selecionadas. "
                       "Tente alargar a margem ou ajustar os alvos.")
            st.stop()

        # 8.5 preço por m² construído (e de terreno, para o mapa de calor)
        df["Preco_m2"] = df[COL_VAL] / df[COL_AREA]
        df["Preco_m2_Terreno"] = np.where(
            df[COL_TERR] > 0, df[COL_VAL] / df[COL_TERR], np.nan
        )

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

        # 8.7 classificação Modernizado x Antigo
        #     Regra 1 (principal): ano de construção (Ano_Construcao_Geo) >= 2018.
        #              O valor 0 significa "desconhecido": não é tratado como
        #              antigo automaticamente — nesse caso decide a Regra 2.
        #     Regra 2 (retrofit): a área construída AUMENTOU de forma relevante
        #              ao longo do tempo no MESMO imóvel (mesmo SQL) — ampliação.
        #              Critério estrito para não inflar a categoria: ganho de
        #              área > 20% E > 15 m² entre a transação mais antiga e a
        #              mais recente do mesmo SQL.
        chave_col = COL_SQL if COL_SQL in df.columns else None
        if chave_col is None:
            df["Chave_Imovel"] = df.index.astype(str)
        else:
            df["Chave_Imovel"] = col_as_str(df, chave_col)

        regra_area = pd.Series(False, index=df.index)
        if "Ano_Transacao" in df.columns and df.groupby("Chave_Imovel").ngroups < len(df):
            # ordena por ano e compara área da 1ª x última transação de cada imóvel
            tmp = df[["Chave_Imovel", "Ano_Transacao", COL_AREA]].copy()
            tmp = tmp.sort_values(["Chave_Imovel", "Ano_Transacao"])
            primeira = tmp.groupby("Chave_Imovel")[COL_AREA].first()
            ultima = tmp.groupby("Chave_Imovel")[COL_AREA].last()
            ganho_abs = (ultima - primeira)
            ganho_rel = ganho_abs / primeira.replace(0, np.nan)
            ampliou = (ganho_abs > 15) & (ganho_rel > 0.20)
            chaves_retrofit = set(ampliou[ampliou.fillna(False)].index)
            regra_area = df["Chave_Imovel"].isin(chaves_retrofit)

        ano_constr = df[COL_ANO].fillna(0)
        regra_ano = ano_constr >= 2018

        eh_modernizado = (regra_ano | regra_area.fillna(False)).to_numpy(dtype=bool)
        df["Status"] = np.where(eh_modernizado, "Modernizado", "Antigo")

        df = df.drop(columns=["Chave_Imovel"], errors="ignore")

        # 8.8 filtro de status (Todos / Só modernizados / Só antigos)
        if filtro_status == "Só modernizados":
            df = df[df["Status"] == "Modernizado"]
        elif filtro_status == "Só antigos":
            df = df[df["Status"] == "Antigo"]
        if df.empty:
            st.warning(f"Nenhuma transação na categoria '{filtro_status}' para este recorte.")
            st.stop()

        # --------------------------------------------------------------------
        # 9. SAÍDAS — RELATÓRIO DE AVALIAÇÃO
        # --------------------------------------------------------------------
        # Preço/m² de terreno só faz sentido onde há área de terreno
        terreno_valido = df["Preco_m2_Terreno"].dropna()
        terreno_valido = terreno_valido[terreno_valido > 0]

        # Mínimo/máximo do R$/m² (faixa) e média (valor de referência)
        min_c, max_c = df["Preco_m2"].min(), df["Preco_m2"].max()
        media_c = df["Preco_m2"].mean()
        if not terreno_valido.empty:
            min_t, max_t = terreno_valido.min(), terreno_valido.max()
            media_t = terreno_valido.mean()
        else:
            min_t = max_t = media_t = np.nan

        # cabeçalho do relatório
        contexto = (f"Logradouro: {rua}" if rua else f"Distrito: {distrito_alvo}")
        st.markdown("## 📑 Relatório de Avaliação Imobiliária")
        st.markdown(f"**{contexto}**  ·  {len(df):,} transações comparáveis analisadas")

        # ---- FAIXA DE VALOR ESTIMADA (só quando a pessoa digitou a área) ----
        faixas_estimadas = []  # cada item: (rótulo, valor_min, valor_max)

        if area_constr_alvo > 0:
            vmin_c = min_c * area_constr_alvo
            vmax_c = max_c * area_constr_alvo
            faixas_estimadas.append(("Construção", vmin_c, vmax_c))

        if area_terr_alvo > 0 and not np.isnan(min_t):
            vmin_t = min_t * area_terr_alvo
            vmax_t = max_t * area_terr_alvo
            faixas_estimadas.append(("Terreno", vmin_t, vmax_t))

        if faixas_estimadas:
            st.markdown("### 💰 Faixa de valor estimada para o imóvel")

            # média das faixas disponíveis (se houver as duas, faz a média; se uma só, usa ela)
            vmin_final = np.mean([f[1] for f in faixas_estimadas])
            vmax_final = np.mean([f[2] for f in faixas_estimadas])

            cprinc1, cprinc2 = st.columns(2)
            cprinc1.metric("Valor estimado — mínimo", formata_moeda(vmin_final))
            cprinc2.metric("Valor estimado — máximo", formata_moeda(vmax_final))

            # detalhamento por base (construção / terreno)
            with st.expander("Como esta faixa foi calculada"):
                for rotulo, vmn, vmx in faixas_estimadas:
                    area_ref = area_constr_alvo if rotulo == "Construção" else area_terr_alvo
                    st.markdown(
                        f"- **{rotulo}** ({area_ref:.0f} m²): "
                        f"{formata_moeda(vmn)} a {formata_moeda(vmx)}"
                    )
                if len(faixas_estimadas) == 2:
                    st.markdown("- **Faixa final** = média das faixas de construção e terreno.")
                st.caption("Limites calculados pelo menor e maior preço/m² dos comparáveis "
                           "após o filtro de outliers (se ativado na barra lateral).")
        else:
            st.info("💡 Informe a **área construída** e/ou **área de terreno** na barra lateral "
                    "para obter a faixa de valor estimada do imóvel.")

        # ---- INDICADORES DE MERCADO (sempre por MÉDIA) ----
        st.markdown("### 📊 Indicadores de mercado (R$/m²)")
        i1, i2, i3 = st.columns(3)
        i1.metric("Transações", f"{len(df):,}")
        i2.metric("Média R$/m² construção", formata_moeda(media_c))
        i3.metric("Média R$/m² terreno",
                  formata_moeda(media_t) if not np.isnan(media_t) else "—")
        j1, j2 = st.columns(2)
        j1.metric("% Modernizados", f'{(df["Status"] == "Modernizado").mean() * 100:.0f}%')
        j2.metric("Faixa construção (mín–máx)/m²",
                  f"{formata_moeda(min_c)} — {formata_moeda(max_c)}")

        # 9.1 mapa
        df_geo = df.dropna(subset=["Latitude", "Longitude"])
        df_geo = df_geo[df_geo["Latitude"].between(-90, 90) & df_geo["Longitude"].between(-180, 180)]

        # modo de busca: True quando o usuário escolheu um distrito (sem logradouro)
        modo_distrito = (not rua) and (distrito_alvo != "Selecione...")

        if not df_geo.empty:
            st.markdown("### 🗺️ Distribuição Geográfica")

            feature_dist = None
            if modo_distrito and DISTRITOS_GEO is not None:
                feature_dist = DISTRITOS_GEO.get(str(distrito_alvo))

            # centro do mapa: ponto do endereço (raio) OU centroide do distrito OU média dos pontos
            if lat_c and lon_c:
                centro = [lat_c, lon_c]
            elif feature_dist is not None:
                centro = centroide_distrito(feature_dist) or \
                         [df_geo["Latitude"].mean(), df_geo["Longitude"].mean()]
            else:
                centro = [df_geo["Latitude"].mean(), df_geo["Longitude"].mean()]

            zoom = 13 if modo_distrito else 15
            m = folium.Map(location=centro, zoom_start=zoom, tiles="CartoDB positron")

            # modo raio: marcador alvo + círculo
            if lat_c and lon_c:
                folium.Marker([lat_c, lon_c], tooltip="Endereço Alvo",
                              icon=folium.Icon(color="red", icon="star")).add_to(m)
                folium.Circle([lat_c, lon_c], radius=raio, color="#1f77b4",
                              fill=True, fill_opacity=0.05).add_to(m)

            # modo distrito: contorno do polígono
            if feature_dist is not None:
                folium.GeoJson(
                    feature_dist,
                    name=str(distrito_alvo),
                    style_function=lambda _: {
                        "color": "#d62728", "weight": 3,
                        "fill": True, "fillColor": "#d62728", "fillOpacity": 0.07,
                    },
                    tooltip=str(distrito_alvo),
                ).add_to(m)
                try:
                    m.fit_bounds(folium.GeoJson(feature_dist).get_bounds())
                except Exception:
                    pass

            # camada de ZONEAMENTO (opcional, só no modo distrito) — buscada por bbox
            if mostrar_zoneamento and feature_dist is not None:
                bb = bbox_de_feature(feature_dist)
                if bb is not None:
                    with st.spinner("Carregando zoneamento do distrito..."):
                        zjson = buscar_zoneamento_bbox(*bb)
                    feats_z = (zjson or {}).get("features", [])
                    if feats_z:
                        siglas_presentes = {}
                        for fz in feats_z:
                            sig = str(fz.get("properties", {}).get(CAMPO_ZONA, "—"))
                            cor = _cor_zona(sig)
                            siglas_presentes[sig] = cor
                            folium.GeoJson(
                                fz,
                                style_function=lambda _f, _c=cor: {
                                    "color": _c, "weight": 1,
                                    "fill": True, "fillColor": _c, "fillOpacity": 0.25,
                                },
                                tooltip=sig,
                            ).add_to(m)
                        # legenda das zonas presentes
                        itens = "".join(
                            f'<div style="margin:2px 0"><span style="display:inline-block;'
                            f'width:12px;height:12px;background:{c};margin-right:6px;'
                            f'border:1px solid #666"></span>{s}</div>'
                            for s, c in sorted(siglas_presentes.items())
                        )
                        legenda = (
                            '<div style="position:fixed;bottom:30px;right:12px;z-index:9999;'
                            'background:white;padding:8px 10px;border:1px solid #999;'
                            'border-radius:6px;font-size:11px;max-height:240px;'
                            'overflow:auto;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
                            '<b>Zoneamento (LPUOS 2016)</b>' + itens + '</div>'
                        )
                        m.get_root().html.add_child(folium.Element(legenda))
                    else:
                        st.info("ℹ️ Zoneamento indisponível para esta área no momento "
                                "(serviço do GeoSampa não respondeu ou não há zonas no recorte).")

            cores = {"Modernizado": "green", "Antigo": "blue"}
            camada = MarkerCluster().add_to(m) if HAS_CLUSTER else m

            # guarda de desempenho: acima do limite, plota uma amostra no MAPA
            # (métricas, gráfico e tabela continuam usando TODAS as transações)
            MAX_PINS_MAPA = 15000
            df_map = df_geo
            if len(df_geo) > MAX_PINS_MAPA:
                df_map = df_geo.sample(MAX_PINS_MAPA, random_state=1)
                st.caption(f"🗺️ O mapa mostra {MAX_PINS_MAPA:,} pinos de "
                           f"{len(df_geo):,} (amostra, para não travar o navegador). "
                           f"As métricas, o gráfico e a tabela usam todas as transações.")

            for _, r in df_map.iterrows():
                popup = (f"<b>R$/m²:</b> {formata_moeda(r['Preco_m2'])}<br>"
                         f"<b>Valor:</b> {formata_moeda(r[COL_VAL])}<br>"
                         f"<b>Área:</b> {r[COL_AREA]:.0f} m²<br>"
                         f"<b>Status:</b> {r['Status']}")
                folium.CircleMarker(
                    [r["Latitude"], r["Longitude"]], radius=5,
                    color=cores.get(r["Status"], "gray"),
                    fill=True, fill_opacity=0.85,
                    popup=folium.Popup(popup, max_width=260),
                ).add_to(camada)

            # --- Camada de mapa de calor (opcional) ---
            if heatmap_modo != "Desligado" and HAS_HEATMAP:
                if heatmap_modo == "Densidade de transações":
                    # cada transação pesa igual -> regiões com mais negócios ficam quentes
                    pontos = df_geo[["Latitude", "Longitude"]].values.tolist()
                    if pontos:
                        HeatMap(pontos, radius=18, blur=22, min_opacity=0.3,
                                name="Densidade").add_to(m)
                else:
                    # Preço/m² de terreno -> ponderado pelo valor (regiões caras ficam quentes)
                    h = df_geo.dropna(subset=["Preco_m2_Terreno"]).copy()
                    h = h[h["Preco_m2_Terreno"] > 0]
                    if not h.empty:
                        # remove extremos do peso (p5–p95) para a escala de cor não saturar
                        lo, hi = h["Preco_m2_Terreno"].quantile([0.05, 0.95])
                        if hi <= lo:
                            hi = h["Preco_m2_Terreno"].max()
                            lo = h["Preco_m2_Terreno"].min()
                        peso = ((h["Preco_m2_Terreno"].clip(lo, hi) - lo) / (hi - lo)) \
                            if hi > lo else 1.0
                        h = h.assign(_peso=peso)
                        pontos = h[["Latitude", "Longitude", "_peso"]].values.tolist()
                        HeatMap(pontos, radius=20, blur=25, min_opacity=0.25,
                                name="Preço/m² terreno").add_to(m)
                        st.caption("🔥 Mapa de calor por **preço/m² de terreno**: tons quentes "
                                   "indicam terreno mais caro na região. Apartamentos têm área "
                                   "de terreno fracionada e podem distorcer — filtre por "
                                   "'Residenciais' para uma leitura mais limpa.")

            render_map(m)
        else:
            st.info("ℹ️ Sem coordenadas válidas nesta amostra (modo textual) — mapa "
                    "indisponível, mas o histórico abaixo é válido.")

        # 9.2 GRÁFICO PRINCIPAL: Preço/m² Modernizado x Antigo por faixa de área construída
        st.markdown("### 📈 Preço/m² construído — Modernizado × Antigo por faixa de área")

        bins = [0, 300, 400, 500, 600, 700, 800, np.inf]
        labels = ["<300", "300–400", "400–500", "500–600", "600–700", "700–800", ">800"]
        df["Faixa_Area"] = pd.cut(df[COL_AREA], bins=bins, labels=labels, right=False)

        ag = (
            df.dropna(subset=["Faixa_Area"])
            .groupby(["Faixa_Area", "Status"], observed=True)
            .agg(preco_m2=("Preco_m2", "mean"), n=("Preco_m2", "size"))
            .reset_index()
        )

        if ag.empty:
            st.info("Sem dados suficientes nesta região para montar o gráfico.")
        else:
            chart = (
                alt.Chart(ag)
                .mark_bar()
                .encode(
                    x=alt.X("Faixa_Area:N", sort=labels,
                            title="Faixa de área construída (m²)"),
                    xOffset=alt.XOffset("Status:N"),
                    y=alt.Y("preco_m2:Q", title="Média do preço/m² (R$)"),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(domain=["Antigo", "Modernizado"],
                                        range=["#9aa0a6", "#1a9850"]),
                        title="",
                    ),
                    tooltip=[
                        alt.Tooltip("Faixa_Area:N", title="Faixa"),
                        alt.Tooltip("Status:N", title="Tipo"),
                        alt.Tooltip("preco_m2:Q", title="Média R$/m²", format=",.0f"),
                        alt.Tooltip("n:Q", title="Nº transações"),
                    ],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Ver nº de transações por faixa (cuidado com amostras pequenas)"):
                tabela = (ag.pivot_table(index="Faixa_Area", columns="Status",
                                         values="n", observed=True)
                          .reindex(labels).fillna(0).astype(int))
                st.dataframe(tabela, use_container_width=True)

        # 9.2b GRÁFICO: evolução do R$/m² médio por ano da transação (Modernizado × Antigo)
        st.markdown("### 📉 Evolução do R$/m² médio por ano da transação")
        serie = df.dropna(subset=["Ano_Transacao", "Preco_m2"]).copy()
        serie["Ano"] = serie["Ano_Transacao"].astype(int)
        evol = (
            serie.groupby(["Ano", "Status"], observed=True)
            .agg(preco_m2=("Preco_m2", "mean"), n=("Preco_m2", "size"))
            .reset_index()
        )
        # evita linhas tremidas por anos com pouquíssimas transações
        evol = evol[evol["n"] >= 3]

        if evol.empty or evol["Ano"].nunique() < 2:
            st.info("Dados insuficientes para traçar a evolução por ano neste recorte "
                    "(é preciso pelo menos 2 anos com amostra suficiente).")
        else:
            linha = (
                alt.Chart(evol)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Ano:O", title="Ano da transação"),
                    y=alt.Y("preco_m2:Q", title="Média do preço/m² construído (R$)"),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(domain=["Antigo", "Modernizado"],
                                        range=["#9aa0a6", "#1a9850"]),
                        title="",
                    ),
                    tooltip=[
                        alt.Tooltip("Ano:O", title="Ano"),
                        alt.Tooltip("Status:N", title="Tipo"),
                        alt.Tooltip("preco_m2:Q", title="Média R$/m²", format=",.0f"),
                        alt.Tooltip("n:Q", title="Nº transações"),
                    ],
                )
                .properties(height=340)
            )
            st.altair_chart(linha, use_container_width=True)

        # 9.3 tabela de comparáveis
        st.markdown("### 📋 Transações Comparáveis")
        df_show = df.drop(columns=["_lat", "_lon", "Faixa_Area"], errors="ignore").copy()
        if "dist_metros" in df_show.columns:
            df_show = df_show.sort_values("dist_metros")
        else:
            df_show = df_show.sort_values("Preco_m2")
        st.dataframe(df_show, use_container_width=True)

else:
    st.info("Informe um logradouro (busca por raio/contingência textual) ou selecione um "
            "distrito na barra lateral para iniciar a análise.")
