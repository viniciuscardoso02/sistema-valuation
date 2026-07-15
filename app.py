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
import re
import unicodedata
import requests
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import folium
import altair as alt

# --- Renderizador de mapa ---
# Usamos folium_static (HTML Leaflet puro): mapa totalmente interativo — zoom,
# clique em cluster, popups — SEM o canal de retorno do st_folium, que causava
# segmentation fault no Streamlit Cloud. st_folium fica só como reserva.
try:
    from streamlit_folium import folium_static

    def render_map(m):
        try:
            folium_static(m, width=1100, height=480)
        except TypeError:
            folium_static(m)

except Exception:
    from streamlit_folium import st_folium

    def render_map(m):
        try:
            st_folium(m, height=480, use_container_width=True, returned_objects=[])
        except TypeError:
            st_folium(m, width=1100, height=480, returned_objects=[])

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


def geocodificar_endereco(rua, num, glob_path):
    """Acha as coordenadas de um endereço em 2 etapas:
    (1) na PRÓPRIA base — casa a rua (e o número, se houver) e usa a mediana das
        coordenadas dos imóveis correspondentes; rápido, grátis, robusto.
    (2) se a base não tiver, recorre ao LocationIQ (chave em st.secrets).
    Retorna (lat, lon, fonte, rotulo) ou (None, None, None, None) se falhar.
    'fonte' é 'base' ou 'locationiq'; 'rotulo' é o texto do endereço encontrado."""
    lat_e = coord_sql("Latitude")
    lon_e = coord_sql("Longitude")

    # --- Etapa 1: buscar na própria base ---
    try:
        palavras = extrair_palavras_chave_rua(rua)
        if palavras:
            cond_rua = " AND ".join(
                [f"UPPER(\"{COL_LOGR}\") LIKE '%{sql_str(p)}%'" for p in palavras]
            )
            # se número informado, tenta casar o número exato primeiro
            for cond_num in ([f"AND TRY_CAST(REGEXP_REPLACE(CAST(\"{COL_NUM}\" AS VARCHAR),"
                              f"'[^0-9]','','g') AS INTEGER) = {int(re.sub(r'[^0-9]','', str(num)))}"]
                             if num and re.sub(r'[^0-9]', '', str(num)) else [""]) + [""]:
                q = f"""
                SELECT MEDIAN({lat_e}) AS lat, MEDIAN({lon_e}) AS lon, COUNT(*) AS n
                FROM read_parquet('{glob_path}', union_by_name=true)
                WHERE ({cond_rua}) {cond_num}
                  AND {lat_e} IS NOT NULL AND {lon_e} IS NOT NULL
                """
                r = duckdb.query(q).df()
                if not r.empty and r["n"].iloc[0] > 0 and pd.notna(r["lat"].iloc[0]):
                    rotulo = f"{rua}" + (f", {num}" if num and cond_num else "")
                    return (float(r["lat"].iloc[0]), float(r["lon"].iloc[0]),
                            "base", rotulo)
    except Exception:
        pass

    # --- Etapa 2: LocationIQ (fallback) ---
    try:
        chave = st.secrets.get("LOCATIONIQ_KEY", None)
        if chave:
            endereco = f"{rua}, {num}, São Paulo, SP, Brasil" if num \
                else f"{rua}, São Paulo, SP, Brasil"
            resp = requests.get(
                "https://us1.locationiq.com/v1/search",
                params={"key": chave, "q": endereco, "format": "json", "limit": 1,
                        "countrycodes": "br"},
                timeout=8,
            )
            if resp.status_code == 200:
                dados = resp.json()
                if dados:
                    return (float(dados[0]["lat"]), float(dados[0]["lon"]),
                            "locationiq", dados[0].get("display_name", endereco).split(",")[0])
    except Exception:
        pass

    return (None, None, None, None)


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


@st.cache_data(show_spinner=False)
def carregar_lista_zonas(glob_path):
    """Lista as zonas (LPUOS) presentes na base, para o filtro de exclusão.
    Retorna [] se a coluna Zona ainda não existir na base."""
    if "Zona" not in get_available_columns(glob_path):
        return []
    try:
        q = ('SELECT DISTINCT "Zona" AS z '
             f"FROM read_parquet('{glob_path}', union_by_name=true) "
             'WHERE "Zona" IS NOT NULL')
        df_z = duckdb.query(q).df()
        return sorted(df_z["z"].astype(str).unique())
    except Exception:
        return []


zonas_disp = carregar_lista_zonas(PARQUET_GLOB)


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


# --- GeoJSON das quadras (para valorização por quarteirão) ---
GEOJSON_QUADRAS = os.path.join(APP_DIR, "quadras_sp.geojson")
if not os.path.exists(GEOJSON_QUADRAS):
    GEOJSON_QUADRAS = "quadras_sp.geojson"


@st.cache_data(show_spinner=False)
def carregar_geojson_quadras(caminho):
    """Carrega o GeoJSON das quadras uma vez e indexa por código de quadra
    (setor+quadra, 6 dígitos). Retorna dict {codigo: geometry_dict} ou {}."""
    try:
        import json
        with open(caminho, "r", encoding="utf-8") as fh:
            fc = json.load(fh)
        por_codigo = {}
        for feat in fc.get("features", []):
            cod = feat.get("properties", {}).get("Quadra")
            if cod:
                por_codigo[str(cod)] = feat.get("geometry")
        return por_codigo
    except Exception:
        return {}


QUADRAS_GEO = carregar_geojson_quadras(GEOJSON_QUADRAS)


# --- Base de alvarás (Aprova Digital), indexada por SQL ---
ALVARAS_PATH = os.path.join(APP_DIR, "alvaras_final.parquet")
if not os.path.exists(ALVARAS_PATH):
    ALVARAS_PATH = "alvaras_final.parquet"


@st.cache_data(show_spinner=False)
def carregar_alvaras(caminho):
    """Carrega a base de alvarás (uma linha por SQL×alvará) e devolve dois objetos:
    (1) o DataFrame completo, e (2) um dict {SQL: DataFrame dos alvarás daquele SQL}
    para consulta rápida no relatório do imóvel. Retorna (None, {}) se faltar."""
    try:
        df = pd.read_parquet(caminho)
        df["SQL"] = df["SQL"].astype(str)
        # ano como número (para ordenar); mantém 'nan' como faltante
        df["Ano_Alvara_num"] = pd.to_numeric(df["Ano_Alvara"], errors="coerce")
        por_sql = {sql: grupo for sql, grupo in df.groupby("SQL")}
        return df, por_sql
    except Exception:
        return None, {}


ALVARAS_DF, ALVARAS_POR_SQL = carregar_alvaras(ALVARAS_PATH)


# --- Base de anúncios (Matú Imóveis), separada da base de transações ---
ANUNCIOS_PATH = os.path.join(APP_DIR, "anuncios_matu.parquet")
if not os.path.exists(ANUNCIOS_PATH):
    ANUNCIOS_PATH = "anuncios_matu.parquet"

# Como os anúncios seguem o MESMO filtro "Uso do Imóvel" das transações,
# mapeamos os subtipos da imobiliária para os dois grupos do app.
GRUPO_ANUNCIO = {
    "Residenciais": {"Casa", "Casa de Condomínio"},
    "Apartamentos": {"Apartamento", "Cobertura", "Duplex"},
}

# limites do município (descarta coordenadas claramente erradas)
BBOX_SP = (-23.83, -23.36, -46.83, -46.36)   # lat_min, lat_max, lon_min, lon_max


@st.cache_data(show_spinner=False)
def carregar_anuncios(caminho):
    """Carrega os anúncios (preço PEDIDO). Base independente da de transações:
    serve para comparar oferta x fechado, nunca para entrar na média do ITBI.
    Coordenadas vêm do CEP (precisão de logradouro). Retorna None se faltar."""
    try:
        df = pd.read_parquet(caminho)
        for c in ("valor", "preco_m2_pedido", "lat", "lon", "area_construida",
                  "area_terreno", "dorm", "banh", "suite", "vaga"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # zera coordenadas fora do município (não desenha ponto errado)
        if {"lat", "lon"}.issubset(df.columns):
            dentro = (df["lat"].between(BBOX_SP[0], BBOX_SP[1]) &
                      df["lon"].between(BBOX_SP[2], BBOX_SP[3]))
            df.loc[~dentro, ["lat", "lon"]] = np.nan
        return df
    except Exception:
        return None


ANUNCIOS_DF = carregar_anuncios(ANUNCIOS_PATH)


def _dist_m(lat1, lon1, lat2, lon2):
    """Distância aproximada em metros (equirretangular; suficiente p/ raios curtos)."""
    import math
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6371000 * math.hypot(x, y)


def _ponto_em_geom(lat, lon, geom):
    """Ray casting: o ponto está dentro do Polygon/MultiPolygon do GeoJSON?"""
    def _em_anel(anel):
        dentro = False
        n = len(anel)
        for i in range(n):
            x1, y1 = anel[i][0], anel[i][1]
            x2, y2 = anel[(i + 1) % n][0], anel[(i + 1) % n][1]
            if (y1 > lat) != (y2 > lat):
                xin = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
                if lon < xin:
                    dentro = not dentro
        return dentro
    try:
        t = geom.get("type")
        coords = geom.get("coordinates", [])
        poligonos = [coords] if t == "Polygon" else (coords if t == "MultiPolygon" else [])
        for poly in poligonos:
            if not poly:
                continue
            if _em_anel(poly[0]):                       # anel externo
                if any(_em_anel(buraco) for buraco in poly[1:]):
                    continue                            # caiu num buraco
                return True
        return False
    except Exception:
        return False


def anuncios_da_regiao(anuncios, tipo, negocios, centro=None, raio_m=None, geom_dist=None):
    """Filtra anúncios por uso (mesmo botão das transações), negócio e região.
    Região = raio ao redor do centro OU polígono do distrito. Só anúncios com
    coordenada entram no recorte espacial."""
    if anuncios is None or anuncios.empty:
        return anuncios.iloc[0:0] if anuncios is not None else None
    d = anuncios[anuncios["subtipo"].isin(GRUPO_ANUNCIO.get(tipo, set()))]
    if negocios:
        d = d[d["negocio"].isin(negocios)]
    d = d[d["lat"].notna() & d["lon"].notna()]
    if d.empty:
        return d
    if geom_dist is not None:
        mask = d.apply(lambda r: _ponto_em_geom(r["lat"], r["lon"], geom_dist), axis=1)
        return d[mask]
    if centro is not None and raio_m:
        mask = d.apply(lambda r: _dist_m(centro[0], centro[1], r["lat"], r["lon"]) <= raio_m,
                       axis=1)
        return d[mask]
    return d


@st.cache_data(show_spinner=False)
def carregar_reformados(glob_path, piso_m2=5000, teto_m2=50000,
                        ano_min=None, ano_max=None, uso_sql="", zonas_excl=None):
    """Carrega transações de imóveis reformados (Modernizada=true) com coordenada,
    área e valor válidos, já com R$/m² calculado. É o 'valor de saída' (preço-alvo
    pós-reforma) da análise de retrofit.

    Respeita os MESMOS filtros da barra lateral aplicados ao resto do app, para a
    referência ser coerente com o recorte que o usuário está vendo:
      - período (ano_min..ano_max)
      - uso (uso_sql: cláusula já pronta de Residenciais/Apartamentos)
      - zoneamento excluído (zonas_excl: lista de zonas a remover)

    A base ITBI é VALOR DECLARADO — ~72% dos reformados têm valor subdeclarado
    (herança, doação) abaixo de R$1.000/m², que contaminam a mediana; por isso a
    FAIXA DE MERCADO [piso, teto], digitável. Cacheado por combinação de filtros;
    filtro por raio é feito em memória."""
    lat_e = coord_sql("Latitude")
    lon_e = coord_sql("Longitude")
    val_e = coord_sql(COL_VAL)
    area_e = coord_sql(COL_AREA)
    ano_e = coord_sql("Ano_Transacao")

    cond_ano = ""
    if ano_min is not None and ano_max is not None:
        cond_ano = f" AND {ano_e} BETWEEN {int(ano_min)} AND {int(ano_max)}"

    cond_zona = ""
    if zonas_excl:
        lista = ", ".join("'" + str(z).replace("'", "''") + "'" for z in zonas_excl)
        cond_zona = f" AND (Zona IS NULL OR Zona NOT IN ({lista}))"

    q = f"""
    WITH b AS (
        SELECT {lat_e} AS lat, {lon_e} AS lon,
               {val_e} AS valor, {area_e} AS area
        FROM read_parquet('{glob_path}', union_by_name=true)
        WHERE LOWER(CAST(Modernizada AS VARCHAR)) = 'true'
        {uso_sql}
        {cond_ano}
        {cond_zona}
    )
    SELECT lat, lon, valor / area AS preco_m2
    FROM b
    WHERE lat IS NOT NULL AND lon IS NOT NULL
      AND area > 0 AND valor > 0
      AND (valor / area) BETWEEN {float(piso_m2)} AND {float(teto_m2)}
    """
    try:
        return run_query(q)
    except Exception:
        return pd.DataFrame(columns=["lat", "lon", "preco_m2"])


def _venda_m2_vizinhos(anuncio_lat, anuncio_lon, reformados_df,
                       raio_ini=500, raio_max=2000, passo=250, min_casos=5):
    """Mediana do R$/m² CONSTRUÍDO dos imóveis reformados ao redor do anúncio.
    Expande o raio de `raio_ini` até `raio_max` enquanto não juntar `min_casos`.
    Retorna (venda_m2, raio_usado, n_casos) ou (None, None, 0)."""
    if reformados_df is None or reformados_df.empty:
        return None, None, 0
    dlat = np.radians(reformados_df["lat"].values - anuncio_lat)
    dlon = np.radians(reformados_df["lon"].values - anuncio_lon)
    latm = np.radians((reformados_df["lat"].values + anuncio_lat) / 2)
    dist = 6371000 * np.hypot(dlon * np.cos(latm), dlat)

    raio = raio_ini
    while raio <= raio_max:
        sel = reformados_df[dist <= raio]
        if len(sel) >= min_casos:
            return sel["preco_m2"].median(), raio, len(sel)
        raio += passo
    sel = reformados_df[dist <= raio_max]
    if len(sel) == 0:
        return None, None, 0
    return sel["preco_m2"].median(), raio_max, len(sel)


def classificar_retrofit(anuncio_lat, anuncio_lon, valor_pedido, area_construida,
                         area_terreno, reformados_df, custo_obra_m2, desconto=0.15,
                         area_projetada=0, valor_saida_fixo=None,
                         raio_ini=500, raio_max=2000, passo=250, min_casos=5,
                         corte_verde=0.15):
    """Viabilidade de comprar um anúncio para retrofit. Conta em VALORES TOTAIS,
    porque compra e venda têm bases diferentes (terreno x construída):

        Compra        = preço pedido × (1 − desconto)
                        (exibida também como R$/m² de TERRENO)
        Área de venda = max(área construída atual, área projetada)
                        — nunca reduz: se a casa já é maior, mantém a atual
        Obra          = custo_obra_m2 × área de venda
                        (sobre a área FINAL; senão a ampliação sairia de graça)
        Venda         = venda_m2 × área de venda
                        venda_m2 = R$/m² CONSTRUÍDO, de um de dois modos:
                          · `valor_saida_fixo`: valor arbitrado pelo usuário
                          · senão: mediana dos reformados ao redor (raio expansível)

        Lucro %       = (Venda − Compra − Obra) ÷ (Compra + Obra)

    Cores (sobre o LUCRO):
        verde    = lucro >= corte_verde (padrão 15%)
        amarelo  = 0 <= lucro < corte_verde     (apertado)
        vermelho = lucro < 0                    (prejuízo)

    A área de terreno é apenas lente de leitura: se faltar, o imóvel continua
    sendo classificado (só não exibe o R$/m² de terreno)."""
    # requisitos mínimos da conta
    if valor_pedido is None or pd.isna(valor_pedido) or valor_pedido <= 0:
        return {"cor": "cinza", "motivo": "sem valor do anúncio"}
    if area_construida is None or pd.isna(area_construida) or area_construida <= 0:
        return {"cor": "cinza", "motivo": "sem área construída"}

    # área que será vendida: a projetada só vale se for MAIOR que a atual
    area_venda = float(area_construida)
    if area_projetada and area_projetada > area_venda:
        area_venda = float(area_projetada)
    ampliou = area_venda > float(area_construida)

    # preço de venda por m² construído
    if valor_saida_fixo is not None and valor_saida_fixo > 0:
        venda_m2, raio_usado, n_ref, modo = float(valor_saida_fixo), None, None, "manual"
    else:
        venda_m2, raio_usado, n_ref = _venda_m2_vizinhos(
            anuncio_lat, anuncio_lon, reformados_df,
            raio_ini=raio_ini, raio_max=raio_max, passo=passo, min_casos=min_casos)
        modo = "vizinhos"
        if venda_m2 is None or venda_m2 <= 0:
            return {"cor": "cinza",
                    "motivo": f"nenhum reformado em {raio_max} m"}

    compra = float(valor_pedido) * (1 - desconto)
    obra = float(custo_obra_m2) * area_venda
    receita = venda_m2 * area_venda
    custo_total = compra + obra
    if custo_total <= 0:
        return {"cor": "cinza", "motivo": "custo inválido"}

    lucro = receita - custo_total
    lucro_pct = lucro / custo_total

    if lucro_pct >= corte_verde:
        cor = "verde"
    elif lucro_pct >= 0:
        cor = "amarelo"
    else:
        cor = "vermelho"

    # lente de leitura: quanto se paga por m² de terreno (opcional)
    compra_m2_terreno = None
    if area_terreno is not None and not pd.isna(area_terreno) and area_terreno > 0:
        compra_m2_terreno = compra / float(area_terreno)

    return {
        "cor": cor,
        "compra": compra,
        "compra_m2_terreno": compra_m2_terreno,
        "obra": obra,
        "receita": receita,
        "custo_total": custo_total,
        "lucro": lucro,
        "lucro_pct": lucro_pct,
        "venda_m2": venda_m2,
        "area_venda": area_venda,
        "ampliou": ampliou,
        "raio_usado": raio_usado,
        "n_reformados": n_ref,
        "modo": modo,
    }


# --- Zoneamento (LPUOS 2016) buscado sob demanda por área, direto do GeoSampa ---
WFS_GEOSAMPA = "http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
CAMADA_ZONEAMENTO = "geoportal:zoneamento_2016_map1"
CAMPO_ZONA = "cd_zoneamento_perimetro"


# --- Parâmetros construtivos do Quadro 3 da LPUOS (Lei 16.402/2016) ---
# Fonte oficial: gestaourbana.prefeitura.sp.gov.br (Quadro 3 - Parâmetros de ocupação).
# Campos: CA básico, CA máximo, TO (<500m²), TO (>=500m²), gabarito (m).
# "NA (livre)" = gabarito não definido pela zona. obs = nota do próprio Quadro 3.
QUADRO3_PARAMETROS = {
    "ZEU":    {"ca_bas": "1", "ca_max": "4",   "to_ate500": "0,85", "to_500": "0,70", "gab": "NA (livre)", "obs": ""},
    "ZEUa":   {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,70", "to_500": "0,50", "gab": "28", "obs": ""},
    "ZEUP":   {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "28", "obs": "Atendido o art. 83 do PDE, recebe os parâmetros de ZEU (CAmáx 4)."},
    "ZEUPa":  {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,70", "to_500": "0,50", "gab": "28", "obs": "Atendido o art. 83 do PDE, recebe os parâmetros de ZEUa."},
    "ZEM":    {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "28", "obs": "CAmáx pode ser 4 nos casos do §1º do art. 8º."},
    "ZEMP":   {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "28", "obs": "CAmáx pode ser 4 nos casos do §2º do art. 8º."},
    "ZC":     {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "48", "obs": ""},
    "ZCa":    {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,70", "to_500": "0,70", "gab": "20", "obs": ""},
    "ZC-ZEIS":{"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "NA (livre)", "obs": ""},
    "ZCOR-1": {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZCOR-2": {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZCOR-3": {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZCORa":  {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZM":     {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "28", "obs": ""},
    "ZMa":    {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,70", "to_500": "0,50", "gab": "15", "obs": ""},
    "ZMIS":   {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,85", "to_500": "0,70", "gab": "28", "obs": ""},
    "ZMISa":  {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,70", "to_500": "0,50", "gab": "15", "obs": ""},
    "ZEIS-1": {"ca_bas": "1", "ca_max": "2,5", "to_ate500": "0,85", "to_500": "0,70", "gab": "NA (livre)", "obs": "CAmáx = 2 se o lote < 1.000 m²."},
    "ZEIS-2": {"ca_bas": "1", "ca_max": "4",   "to_ate500": "0,85", "to_500": "0,70", "gab": "NA (livre)", "obs": "CAmáx = 2 se o lote < 1.000 m²."},
    "ZEIS-3": {"ca_bas": "1", "ca_max": "4",   "to_ate500": "0,85", "to_500": "0,70", "gab": "NA (livre)", "obs": "CAmáx = 2 se o lote < 500 m²."},
    "ZEIS-4": {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,70", "to_500": "0,50", "gab": "NA (livre)", "obs": "CAmáx = 1 se o lote < 1.000 m²."},
    "ZEIS-5": {"ca_bas": "1", "ca_max": "4",   "to_ate500": "0,85", "to_500": "0,70", "gab": "NA (livre)", "obs": "CAmáx = 2 se o lote < 1.000 m²."},
    "ZDE-1":  {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,70", "to_500": "0,70", "gab": "28", "obs": ""},
    "ZDE-2":  {"ca_bas": "1", "ca_max": "2",   "to_ate500": "0,70", "to_500": "0,50", "gab": "28", "obs": ""},
    "ZPI-1":  {"ca_bas": "1", "ca_max": "1,5", "to_ate500": "0,70", "to_500": "0,70", "gab": "28", "obs": ""},
    "ZPI-2":  {"ca_bas": "1", "ca_max": "1,5", "to_ate500": "0,50", "to_500": "0,30", "gab": "28", "obs": ""},
    "ZPR":    {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZER-1":  {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZER-2":  {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZERa":   {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,50", "to_500": "0,50", "gab": "10", "obs": ""},
    "ZPDS":   {"ca_bas": "1", "ca_max": "1",   "to_ate500": "0,35", "to_500": "0,25", "gab": "20", "obs": ""},
    "ZPDSr":  {"ca_bas": "0,2", "ca_max": "0,2","to_ate500": "0,20", "to_500": "0,15", "gab": "10", "obs": ""},
    "ZEPAM":  {"ca_bas": "0,1", "ca_max": "0,1","to_ate500": "0,10", "to_500": "0,10", "gab": "10", "obs": ""},
}


def parametros_construtivos(sigla):
    """Parâmetros do Quadro 3 para a sigla. Correspondência exata e, se falhar,
    por prefixo mais longo (ZCOR-1 antes de ZC)."""
    if not sigla:
        return None
    s = str(sigla).upper().strip()
    if s in QUADRO3_PARAMETROS:
        return QUADRO3_PARAMETROS[s]
    for chave in sorted(QUADRO3_PARAMETROS, key=len, reverse=True):
        if s.startswith(chave):
            return QUADRO3_PARAMETROS[chave]
    return None


def nome_familia_zona(sigla):
    """Nome legível da família da zona a partir da sigla (LPUOS 2016)."""
    s = (sigla or "").upper()
    familias = [
        ("ZEIS", "Zona Especial de Interesse Social"),
        ("ZEPAM", "Zona Especial de Proteção Ambiental"),
        ("ZEPEC", "Zona Especial de Preservação Cultural"),
        ("ZEP", "Zona Especial de Preservação"),
        ("ZER", "Zona Exclusivamente Residencial"),
        ("ZEUP", "Eixo de Estruturação (Previsto)"),
        ("ZEU", "Eixo de Estruturação da Transformação Urbana"),
        ("ZEM", "Eixo de Estruturação (Metropolitano)"),
        ("ZC", "Zona de Centralidade"),
        ("ZM", "Zona Mista"),
        ("ZPI", "Zona Predominantemente Industrial"),
        ("ZDE", "Zona de Desenvolvimento Econômico"),
        ("ZPR", "Zona Predominantemente Residencial"),
        ("ZOE", "Zona de Ocupação Especial"),
        ("ZLT", "Zona de Lazer e Turismo"),
    ]
    for pref, nome in familias:
        if s.startswith(pref):
            return nome
    if "PRAÇA" in s or "CANTEIRO" in s:
        return "Praça / Canteiro / Área verde"
    return ""


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


# --- Valorização por quadra (quarteirão) ---
VALOR_MIN_TRANSACOES = 5       # mínimo de transações na quadra
VALOR_MIN_ANOS = 3             # mínimo de anos distintos na quadra


def valorizacao_por_quadra(df_pts):
    """Agrupa as transações pela coluna Quadra (setor+quadra) e calcula a
    valorização anual (% a.a.) de cada quarteirão via regressão log(preço/m²) ~ ano.
    Retorna dict {codigo_quadra: (valorizacao_aa, n)} só para quadras com dados
    suficientes (nº de transações e anos distintos)."""
    if "Quadra" not in df_pts.columns:
        return {}
    d = df_pts.dropna(subset=["Quadra", "Preco_m2", "Ano_Transacao"]).copy()
    d = d[d["Preco_m2"] > 0]
    if d.empty:
        return {}
    d["_ano"] = d["Ano_Transacao"].astype(int)

    resultado = {}
    for cod, grupo in d.groupby("Quadra"):
        if len(grupo) < VALOR_MIN_TRANSACOES:
            continue
        if grupo["_ano"].nunique() < VALOR_MIN_ANOS:
            continue
        x = grupo["_ano"].to_numpy(dtype=float)
        y = np.log(grupo["Preco_m2"].to_numpy(dtype=float))
        try:
            slope = np.polyfit(x, y, 1)[0]
        except Exception:
            continue
        val_aa = float(np.exp(slope) - 1.0)   # valorização anual composta
        resultado[str(cod)] = (val_aa, len(grupo))
    return resultado


def modernizacao_por_quadra(df_pts):
    """Conta imóveis modernizados por quadra (setor+quadra). Usa a base ANTES do
    filtro de status, para mostrar onde há modernização independentemente do
    filtro selecionado. Retorna dict {codigo_quadra: (n_modernizados, n_total)}
    apenas para quadras que têm ao menos 1 modernizado."""
    if "Quadra" not in df_pts.columns or "Status" not in df_pts.columns:
        return {}
    d = df_pts.dropna(subset=["Quadra"]).copy()
    if d.empty:
        return {}
    resultado = {}
    for cod, grupo in d.groupby("Quadra"):
        n_mod = int((grupo["Status"] == "Modernizado").sum())
        if n_mod >= 1:
            resultado[str(cod)] = (n_mod, len(grupo))
    return resultado


def alvaras_por_quadra(alvaras_df, familia="Reforma", quadras_visiveis=None):
    """Conta alvarás por quadra (setor+quadra = 6 primeiros dígitos do SQL do alvará).
    'familia' filtra por Reforma/Edificação Nova/Demolição, ou 'Todos' soma as três.
    Conta PROCESSOS únicos (um alvará multi-lote não infla a contagem por lote).
    Retorna dict {codigo_quadra: n_alvaras} só para quadras com ao menos 1 alvará."""
    if alvaras_df is None or "SQL" not in alvaras_df.columns:
        return {}
    d = alvaras_df.copy()
    if familia != "Todos":
        d = d[d["Familia"] == familia]
    if d.empty:
        return {}
    # quadra = 6 primeiros dígitos do SQL (já normalizado a 11 díg na base)
    d = d.assign(_quadra=d["SQL"].astype(str).str[:6])
    if quadras_visiveis is not None:
        d = d[d["_quadra"].isin(quadras_visiveis)]
    if d.empty:
        return {}
    resultado = {}
    for cod, grupo in d.groupby("_quadra"):
        # conta processos únicos (evita inflar por alvará que cobre vários lotes)
        n = grupo["Processo Aprova Digital"].nunique() \
            if "Processo Aprova Digital" in grupo.columns else len(grupo)
        if n >= 1:
            resultado[str(cod)] = int(n)
    return resultado


def _html_popup_alvaras(quadra, familia, grupo):
    """Monta o HTML do popup listando os alvarás de uma família num quarteirão.
    Recebe 'grupo' já filtrado (quadra+família) para não refiltrar a base a cada
    círculo. Uma linha por processo único, com Rua, Mês/Ano, Fase, Segmento, SQL e
    Nº do processo. Ordenado do mais recente para o mais antigo, rola se for longo."""
    d = grupo
    if d.empty:
        return f"<b>Quarteirão {quadra} — {familia}</b><br>Sem alvarás."
    if "Processo Aprova Digital" in d.columns:
        d = d.drop_duplicates("Processo Aprova Digital")
    d = d.assign(
        _ano=pd.to_numeric(d["Ano_Alvara"], errors="coerce"),
        _mes=pd.to_numeric(d.get("Mes_Alvara"), errors="coerce"),
    )
    d = d.sort_values(["_ano", "_mes"], ascending=False, na_position="last")

    def _competencia(r):
        # monta "mm/aaaa" tratando o .0 que vem do parquet salvo como texto
        ano = int(r["_ano"]) if pd.notna(r["_ano"]) else None
        mes = int(r["_mes"]) if pd.notna(r["_mes"]) else None
        if ano and mes:
            return f"{mes:02d}/{ano}"
        if ano:
            return f"{ano}"
        return "s/ data"

    linhas = []
    for _, r in d.iterrows():
        comp = _competencia(r)
        rua = r.get("Rua", "")
        rua = rua if (isinstance(rua, str) and rua and rua.lower() != "nan") else ""
        num = r.get("Numero", "")
        num = num if (isinstance(num, str) and num and num.lower() != "nan"
                      and num not in ("0", "0.0")) else ""
        # monta "Rua, Número" conforme o que houver
        if rua and num:
            endereco = f"{rua}, {num}"
        elif rua:
            endereco = rua
        else:
            endereco = "s/ endereço"
        linhas.append(
            f"<li style='margin-bottom:5px'>"
            f"<b>{endereco}</b><br>"
            f"{comp} · {r.get('Fase','—')} · {r.get('Segmento','—')}<br>"
            f"<span style='color:#555'>SQL {r.get('SQL','—')} · "
            f"Proc. {r.get('Processo Aprova Digital','—')}</span></li>"
        )
    return (
        f"<div style='font-size:13px'>"
        f"<b>Quarteirão {quadra} — {familia}</b> ({len(d)})"
        f"<ul style='max-height:200px;overflow-y:auto;padding-left:16px;"
        f"margin:6px 0 0 0'>" + "".join(linhas) + "</ul></div>"
    )


# --- Links de busca em portais de imóveis (distrito + tipo + faixa de preço) ---
# Zona oficial de cada distrito (fonte: GeoSampa, campo nm_regiao_05).
DISTRITO_ZONA = {
    "AGUA RASA": "Leste", "ALTO DE PINHEIROS": "Oeste", "ANHANGUERA": "Norte",
    "ARICANDUVA": "Leste", "ARTUR ALVIM": "Leste", "BARRA FUNDA": "Oeste",
    "BELA VISTA": "Centro", "BELEM": "Leste", "BOM RETIRO": "Centro", "BRAS": "Leste",
    "BRASILANDIA": "Norte", "BUTANTA": "Oeste", "CACHOEIRINHA": "Norte",
    "CAMBUCI": "Centro", "CAMPO BELO": "Sul", "CAMPO GRANDE": "Sul", "CAMPO LIMPO": "Sul",
    "CANGAIBA": "Leste", "CAPAO REDONDO": "Sul", "CARRAO": "Leste", "CASA VERDE": "Norte",
    "CIDADE ADEMAR": "Sul", "CIDADE DUTRA": "Sul", "CIDADE LIDER": "Leste",
    "CIDADE TIRADENTES": "Leste", "CONSOLACAO": "Centro", "CURSINO": "Sul",
    "ERMELINO MATARAZZO": "Leste", "FREGUESIA DO O": "Norte", "GRAJAU": "Sul",
    "GUAIANASES": "Leste", "IGUATEMI": "Leste", "IPIRANGA": "Sul", "ITAIM BIBI": "Oeste",
    "ITAIM PAULISTA": "Leste", "ITAQUERA": "Leste", "JABAQUARA": "Sul", "JACANA": "Norte",
    "JAGUARA": "Oeste", "JAGUARE": "Oeste", "JARAGUA": "Norte", "JARDIM ANGELA": "Sul",
    "JARDIM HELENA": "Leste", "JARDIM PAULISTA": "Oeste", "JARDIM SAO LUIS": "Sul",
    "JOSE BONIFACIO": "Leste", "LAJEADO": "Leste", "LAPA": "Oeste", "LIBERDADE": "Centro",
    "LIMAO": "Norte", "MANDAQUI": "Norte", "MARSILAC": "Sul", "MOEMA": "Sul",
    "MOOCA": "Leste", "MORUMBI": "Oeste", "PARELHEIROS": "Sul", "PARI": "Leste",
    "PARQUE DO CARMO": "Leste", "PEDREIRA": "Sul", "PENHA": "Leste", "PERDIZES": "Oeste",
    "PERUS": "Norte", "PINHEIROS": "Oeste", "PIRITUBA": "Norte", "PONTE RASA": "Leste",
    "RAPOSO TAVARES": "Oeste", "REPUBLICA": "Centro", "RIO PEQUENO": "Oeste",
    "SACOMA": "Sul", "SANTA CECILIA": "Centro", "SANTANA": "Norte", "SANTO AMARO": "Sul",
    "SAO DOMINGOS": "Norte", "SAO LUCAS": "Leste", "SAO MATEUS": "Leste",
    "SAO MIGUEL": "Leste", "SAO RAFAEL": "Leste", "SAPOPEMBA": "Leste", "SAUDE": "Sul",
    "SE": "Centro", "SOCORRO": "Sul", "TATUAPE": "Leste", "TREMEMBE": "Norte",
    "TUCURUVI": "Norte", "VILA ANDRADE": "Sul", "VILA CURUCA": "Leste",
    "VILA FORMOSA": "Leste", "VILA GUILHERME": "Norte", "VILA JACUI": "Leste",
    "VILA LEOPOLDINA": "Oeste", "VILA MARIA": "Norte", "VILA MARIANA": "Sul",
    "VILA MATILDE": "Leste", "VILA MEDEIROS": "Norte", "VILA PRUDENTE": "Leste",
    "VILA SONIA": "Oeste",
}


def _slug(texto):
    """Converte 'Alto de Pinheiros' -> 'alto-de-pinheiros' (para URLs de portais)."""
    t = remover_acentos(texto).lower().strip()
    return "-".join(t.split())


def links_portais(distrito, tipo, preco_min=None, preco_max=None):
    """Monta URLs de busca para QuintoAndar, Viva Real e ZAP a partir do distrito,
    do tipo de imóvel e (quando houver) da faixa de preço estimada. Os portais
    mudam de estrutura ao longo do tempo; estes links usam o padrão de busca por
    bairro de São Paulo e devem levar à região certa — o usuário refina no site.
    Retorna lista de dicts {nome, url}."""
    from urllib.parse import quote

    bairro_slug = _slug(distrito)
    # normaliza o tipo para cada portal (casa x apartamento)
    eh_apto = (tipo == "Apartamentos")
    links = []

    # --- Viva Real ---
    vr_tipo = "apartamento_residencial" if eh_apto else "casa_residencial"
    vr = (f"https://www.vivareal.com.br/venda/sao-paulo/sao-paulo/bairros/"
          f"{bairro_slug}/{vr_tipo}/")
    params_vr = []
    if preco_min:
        params_vr.append(f"preco-desde={int(preco_min)}")
    if preco_max:
        params_vr.append(f"preco-ate={int(preco_max)}")
    if params_vr:
        vr += "?" + "&".join(params_vr)
    links.append({"nome": "Viva Real", "url": vr})

    # --- ZAP Imóveis (mesma família da Viva Real, estrutura parecida) ---
    # descobre a zona oficial do distrito (nm_regiao_05 do GeoSampa) e monta o
    # trecho de zona da URL. "Centro" no ZAP não usa o prefixo "zona-".
    zona_nome = DISTRITO_ZONA.get(remover_acentos(distrito), "")
    if zona_nome == "Centro":
        zona_slug = "centro"
    elif zona_nome:
        zona_slug = f"zona-{_slug(zona_nome)}"
    else:
        zona_slug = ""  # distrito não mapeado: omite a zona (link mais amplo, não quebra)

    zap_tipo = "apartamento_residencial" if eh_apto else "casa_residencial"
    if zona_slug:
        zap = (f"https://www.zapimoveis.com.br/venda/{zap_tipo}/"
               f"sp+sao-paulo+{zona_slug}+{bairro_slug}/")
    else:
        zap = (f"https://www.zapimoveis.com.br/venda/{zap_tipo}/"
               f"sp+sao-paulo+{bairro_slug}/")
    params_zap = []
    if preco_min:
        params_zap.append(f"precoMinimo={int(preco_min)}")
    if preco_max:
        params_zap.append(f"precoMaximo={int(preco_max)}")
    if params_zap:
        zap += "?" + "&".join(params_zap)
    links.append({"nome": "ZAP Imóveis", "url": zap})

    # --- QuintoAndar (usa busca por texto; filtros na própria página) ---
    qa_tipo = "apartamentos" if eh_apto else "casas"
    qa = (f"https://www.quintoandar.com.br/comprar/imovel/"
          f"{bairro_slug}-sao-paulo-sp-brasil/{qa_tipo}")
    links.append({"nome": "QuintoAndar", "url": qa})

    return links


# --- Pontos de interesse (POIs) via OpenStreetMap / Overpass API ---
# Cada categoria define: rótulo, cor, ícone (folium/glyphicon) e os filtros OSM.
POI_CATEGORIAS = {
    "educacao": {
        "label": "Educação", "cor": "blue", "icone": "education",
        "filtros": ['["amenity"~"school|university|college"]'],
    },
    "verde": {
        "label": "Áreas verdes", "cor": "green", "icone": "tree-conifer",
        "filtros": ['["leisure"="park"]', '["leisure"="garden"]'],
    },
    "saude": {
        "label": "Saúde", "cor": "red", "icone": "plus-sign",
        "filtros": ['["amenity"~"hospital|clinic|doctors"]'],
    },
    "comercio": {
        "label": "Comércio/serviços", "cor": "orange", "icone": "shopping-cart",
        "filtros": ['["shop"="mall"]', '["shop"="supermarket"]',
                    '["amenity"="marketplace"]'],
    },
    "cultura": {
        "label": "Cultura/lazer", "cor": "purple", "icone": "star",
        "filtros": ['["tourism"~"museum|gallery"]',
                    '["amenity"~"theatre|cinema|arts_centre"]'],
    },
}


@st.cache_data(show_spinner=False, ttl=3600)
def buscar_pois_bbox(min_lon, min_lat, max_lon, max_lat, categorias_key):
    """Consulta a Overpass API pelos POIs das categorias dentro do bbox.
    Retorna dict {categoria: [ (lat, lon, nome), ... ]} em caso de sucesso,
    ou {"_erro": "mensagem"} em caso de falha (para diagnóstico na tela).
    Tenta múltiplos servidores espelho, pois o endpoint público vive instável."""
    import requests
    resultados = {c: [] for c in categorias_key}
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    partes = []
    for cat in categorias_key:
        for filtro in POI_CATEGORIAS[cat]["filtros"]:
            partes.append(f'node{filtro}({bbox});')
            partes.append(f'way{filtro}({bbox});')
    query = f"[out:json][timeout:60];({''.join(partes)});out center tags;"

    # servidores espelho da Overpass — se um falha, tenta o próximo
    espelhos = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    elementos = None
    ultimo_erro = ""
    for url in espelhos:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": "h2i-valuation/1.0"}, timeout=90)
            if r.status_code != 200:
                ultimo_erro = f"HTTP {r.status_code} em {url.split('/')[2]}"
                continue
            elementos = r.json().get("elements", [])
            break  # sucesso
        except requests.exceptions.Timeout:
            ultimo_erro = f"timeout em {url.split('/')[2]}"
        except ValueError:
            ultimo_erro = f"resposta não-JSON de {url.split('/')[2]} (servidor sobrecarregado)"
        except Exception as e:
            ultimo_erro = f"{type(e).__name__} em {url.split('/')[2]}"

    if elementos is None:
        return {"_erro": ultimo_erro or "todos os servidores falharam"}

    for el in elementos:
        tags = el.get("tags", {})
        nome = tags.get("name", "(sem nome)")
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        for cat in categorias_key:
            achou = False
            if cat == "educacao" and tags.get("amenity") in ("school", "university", "college"):
                achou = True
            elif cat == "verde" and tags.get("leisure") in ("park", "garden"):
                achou = True
            elif cat == "saude" and tags.get("amenity") in ("hospital", "clinic", "doctors"):
                achou = True
            elif cat == "comercio" and (tags.get("shop") in ("mall", "supermarket")
                                        or tags.get("amenity") == "marketplace"):
                achou = True
            elif cat == "cultura" and (tags.get("tourism") in ("museum", "gallery")
                                       or tags.get("amenity") in ("theatre", "cinema", "arts_centre")):
                achou = True
            if achou:
                resultados[cat].append((lat, lon, nome))
                break
    return resultados


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
          "Preço/m² de terreno: regiões mais caras ficam quentes."),
)

st.sidebar.markdown("---")
st.sidebar.header("🧱 Camadas por Quarteirão")
st.sidebar.caption("Podem ser combinadas: a **cor** mostra valorização e a "
                   "**borda** destaca modernização.")
mostrar_valorizacao = st.sidebar.checkbox(
    "Valorização (% a.a.) — cor do quarteirão", value=False,
    help="Colore cada quarteirão pela tendência de preço/m² ao ano "
         "(verde = subindo, vermelho = caindo).",
)
mostrar_modernizacao = st.sidebar.checkbox(
    "Modernização — borda destacada", value=False,
    help="Destaca com borda os quarteirões que tiveram imóveis modernizados. "
         "Independe do filtro de status.",
)
mostrar_alvaras = st.sidebar.checkbox(
    "Alvarás de obra — círculo no quarteirão", value=False,
    help="Marca cada quarteirão com círculos por família de alvará (Aprova Digital). "
         "Reforma = verde, Edificação Nova = roxo, Demolição = magenta. "
         "O tamanho reflete a quantidade. Independe do filtro de status.",
)
alvara_familias = st.sidebar.multiselect(
    "↳ Famílias a mostrar",
    ["Reforma", "Edificação Nova", "Demolição"],
    default=["Reforma", "Edificação Nova", "Demolição"],
    help="As três aparecem ao mesmo tempo, cada uma com sua cor. "
         "Desmarque alguma para ocultá-la.",
    disabled=not mostrar_alvaras,
)

st.sidebar.markdown("---")
st.sidebar.header("📣 Anúncios (Matú Imóveis)")
if ANUNCIOS_DF is None:
    st.sidebar.caption("Base de anúncios não encontrada no repositório.")
    mostrar_anuncios = False
    anuncio_negocios = []
    anuncios_heatmap = "Desligado"
    retrofit_ligado = False
    retrofit_etiquetas = True
    retrofit_desconto = 0.15
    retrofit_obra = 4000
    retrofit_piso = 5000
    retrofit_teto = 50000
    retrofit_modo = "Imóveis reformados ao redor"
    retrofit_venda_m2 = None
    retrofit_area_proj = 0
else:
    st.sidebar.caption("Preço **pedido** (oferta). Base separada das transações — "
                       "não entra na média do ITBI. Segue o filtro *Uso do Imóvel*.")
    mostrar_anuncios = st.sidebar.checkbox(
        "Mostrar anúncios no mapa", value=False,
        help="Marca cada anúncio com um pino. Venda = preto, Aluguel = azul-petróleo. "
             "A localização vem do CEP (precisão de logradouro).",
    )
    anuncio_negocios = st.sidebar.multiselect(
        "↳ Negócio",
        ["Venda", "Aluguel"],
        default=["Venda"],
        help="Venda e aluguel aparecem com cores diferentes.",
        disabled=not mostrar_anuncios,
    )
    anuncios_heatmap = st.sidebar.radio(
        "Camada de calor dos anúncios",
        ["Desligado", "R$/m² pedido", "Valor total pedido", "Densidade de anúncios"],
        help="Sobreponível ao mapa de calor das transações. Tons de roxo/magenta "
             "(distintos do azul→vermelho das transações), para comparar onde o "
             "preço pedido e o transacionado se concentram. Usa anúncios de VENDA.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Análise de retrofit")
    st.sidebar.caption("Pinta cada anúncio de **verde/amarelo/vermelho** conforme "
                       "a viabilidade de comprar, reformar e revender.")
    retrofit_ligado = st.sidebar.checkbox(
        "Classificar anúncios para retrofit", value=False,
        help="Compra = preço pedido − desconto. Custo total = compra + obra. "
             "Compara com o preço de venda (dos reformados ao redor ou o que você definir).",
    )
    retrofit_etiquetas = st.sidebar.checkbox(
        "↳ Mostrar % de lucro no mapa", value=True, disabled=not retrofit_ligado,
        help="Etiqueta fixa ao lado dos pinos verdes e amarelos. Desligue para "
             "limpar o mapa — a cor do pino continua indicando a classificação, e "
             "o lucro segue no popup e na tabela abaixo do mapa.",
    )
    retrofit_desconto = st.sidebar.slider(
        "Desconto de negociação (%)", 0, 40, 15, disabled=not retrofit_ligado,
        help="Abatimento sobre o preço pedido, assumido na compra.",
    ) / 100.0
    retrofit_obra = st.sidebar.number_input(
        "Custo de obra (R$/m²)", min_value=0, max_value=50000, value=4000, step=500,
        disabled=not retrofit_ligado,
        help="Custo estimado da reforma por m², somado ao preço de compra.",
    )

    retrofit_modo = st.sidebar.radio(
        "Preço de venda (saída) vem de:",
        ["Imóveis reformados ao redor", "Valor que eu definir"],
        disabled=not retrofit_ligado,
        help="Reformados ao redor: usa a mediana do R$/m² dos imóveis reformados "
             "perto de cada anúncio (respeita período/uso/zona). "
             "Valor definido: usa um R$/m² fixo que você arbitra, igual para todos.",
    )

    retrofit_area_proj = st.sidebar.number_input(
        "Área projetada (m²)", min_value=0, max_value=20000, value=0, step=50,
        disabled=not retrofit_ligado,
        help="Área construída que você pretende entregar após a obra. Deixe 0 para "
             "usar a área atual de cada imóvel. Se o imóvel já for MAIOR que a "
             "projetada, mantém a área dele (nunca reduz). A obra e a venda são "
             "calculadas sobre esta área.",
    )

    if retrofit_modo == "Valor que eu definir":
        retrofit_venda_m2 = st.sidebar.number_input(
            "Preço de venda pós-reforma (R$/m²)", min_value=0, max_value=200000,
            value=15000, step=500, disabled=not retrofit_ligado,
            help="Por quanto você acredita que consegue revender o m² reformado "
                 "nesta região. Aplicado a todos os anúncios.",
        )
        # faixa de mercado não é usada neste modo (não consulta reformados)
        retrofit_piso, retrofit_teto = 5000, 50000
    else:
        retrofit_venda_m2 = None
        st.sidebar.caption("Faixa de mercado dos reformados (remove valores subdeclarados "
                           "da base ITBI, como heranças e doações):")
        fx1, fx2 = st.sidebar.columns(2)
        retrofit_piso = fx1.number_input("Piso R$/m²", min_value=0, max_value=100000,
                                         value=5000, step=500, disabled=not retrofit_ligado)
        retrofit_teto = fx2.number_input("Teto R$/m²", min_value=1000, max_value=200000,
                                         value=50000, step=1000, disabled=not retrofit_ligado)

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Zoneamento")
mostrar_zoneamento = st.sidebar.toggle(
    "Mostrar zoneamento (LPUOS 2016)", value=False,
    help=("Desenha as zonas de uso do solo (ZER, ZM, ZEIS, ZEU...) sobre o mapa "
          "do distrito. As zonas são buscadas na hora do GeoSampa, só para a "
          "área visível."),
)
zonas_excluidas = st.sidebar.multiselect(
    "Excluir zonas do cálculo",
    zonas_disp,
    default=[],
    help=("Remove imóveis dessas zonas de TODO o relatório (média de preço, "
          "faixa de valor, mapa). Útil para tirar zonas comerciais que distorcem "
          "a média residencial — ex.: ZCOR-1 nas avenidas do Alto de Pinheiros."),
    disabled=(not zonas_disp),
)
if zonas_excluidas:
    st.sidebar.caption(f"↳ Excluindo: {', '.join(zonas_excluidas)}")

st.sidebar.markdown("---")
st.sidebar.header("📍 Pontos de Interesse")
pois_selecionados = st.sidebar.multiselect(
    "Mostrar no mapa (OpenStreetMap)",
    options=list(POI_CATEGORIAS.keys()),
    format_func=lambda k: POI_CATEGORIAS[k]["label"],
    default=[],
    help=("Escolas, parques, saúde, comércio e cultura da região. "
          "Buscados na hora do OpenStreetMap, só para a área visível."),
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
                # 7.1 geocodifica: primeiro na própria base, depois LocationIQ
                lat_c = lon_c = None
                fonte_geo = rotulo_geo = None
                try:
                    lat_c, lon_c, fonte_geo, rotulo_geo = geocodificar_endereco(
                        rua, num, PARQUET_GLOB)
                except Exception:
                    lat_c = lon_c = None

                tem_geo = has("Latitude") and has("Longitude")

                # 7.2 busca por raio (geocodificou E base tem coordenadas)
                if lat_c is not None and lon_c is not None and tem_geo:
                    origem = "base de transações" if fonte_geo == "base" else "LocationIQ"
                    st.success(f"📍 Endereço Alvo Localizado: **{rotulo_geo}** "
                               f"_(via {origem})_")

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
                        if lat_c is None or lon_c is None:
                            st.warning("⚠️ Não foi possível localizar as coordenadas exatas "
                                       "deste endereço (nem na base, nem no geocodificador). "
                                       "Exibindo o histórico textual da rua.")
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

        # 8.4a exclusão de zonas (afeta todo o relatório: média, faixa, mapa)
        if zonas_excluidas and "Zona" in df.columns:
            antes = len(df)
            df = df[~df["Zona"].isin(zonas_excluidas)]
            removidas = antes - len(df)
            if removidas > 0:
                st.info(f"🗺️ {removidas} transação(ões) de {', '.join(zonas_excluidas)} "
                        f"excluída(s) do cálculo.")
            if df.empty:
                st.warning("Todas as transações do recorte estão nas zonas excluídas. "
                           "Remova alguma zona da exclusão para ver resultados.")
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

        # guarda a base COM classificação mas ANTES do filtro de status, para a
        # camada de "Modernização" contar modernizados independentemente do filtro
        df_sem_filtro_status = df.copy()

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

        # ---- ANÚNCIOS DE IMÓVEIS NA REGIÃO (links para portais) ----
        # só faz sentido quando há um distrito selecionado (modo distrito)
        if not rua and distrito_alvo != "Selecione...":
            # usa a faixa de valor estimada como filtro de preço, se existir
            pmin = pmax = None
            if faixas_estimadas:
                pmin = int(np.mean([f[1] for f in faixas_estimadas]))
                pmax = int(np.mean([f[2] for f in faixas_estimadas]))
            portais = links_portais(distrito_alvo, tipo, pmin, pmax)

            st.markdown("### 🔎 Ver anúncios à venda na região")
            cols_links = st.columns(len(portais))
            for col, p in zip(cols_links, portais):
                col.link_button(p["nome"], p["url"], use_container_width=True)
            filtro_txt = f"{tipo.lower()}"
            if pmin and pmax:
                filtro_txt += f" · {formata_moeda(pmin)} a {formata_moeda(pmax)}"
            st.caption(f"Busca aberta em {distrito_alvo} ({filtro_txt}). Os portais "
                       "atualizam suas URLs periodicamente; se um filtro não vier "
                       "aplicado, refine na própria página do portal.")

        # ---- HISTÓRICO DE ALVARÁS (Aprova Digital) ----
        if ALVARAS_DF is not None and COL_SQL in df.columns:
            # normaliza o SQL dos imóveis do recorte para 11 dígitos
            def _sql11(v):
                d = re.sub(r"\D", "", str(v))
                if len(d) > 11:
                    return d[:11]        # trunca extras
                return d.zfill(11)       # completa à esquerda
            sql_norm = df[COL_SQL].astype(str).apply(_sql11)
            sqls_recorte = set(sql_norm) & set(ALVARAS_POR_SQL.keys())

            if sqls_recorte:
                # junta todos os alvarás dos imóveis do recorte
                alv = ALVARAS_DF[ALVARAS_DF["SQL"].isin(sqls_recorte)].copy()
                # conta alvarás ÚNICOS (um alvará pode cobrir vários lotes)
                n_processos = alv["Processo Aprova Digital"].nunique()
                n_imoveis_com = len(sqls_recorte)
                n_reformas = alv[alv["Familia"] == "Reforma"]["Processo Aprova Digital"].nunique()
                n_novas = alv[alv["Familia"] == "Edificação Nova"]["Processo Aprova Digital"].nunique()
                n_demol = alv[alv["Familia"] == "Demolição"]["Processo Aprova Digital"].nunique()

                st.markdown("### 🏗️ Alvarás de obra na região (Aprova Digital)")
                ca1, ca2, ca3, ca4 = st.columns(4)
                ca1.metric("Imóveis com alvará", n_imoveis_com)
                ca2.metric("Reformas", n_reformas)
                ca3.metric("Construções novas", n_novas)
                ca4.metric("Demolições", n_demol)

                # lista os imóveis reformados (mais relevante para retrofit)
                reformados = alv[alv["Familia"] == "Reforma"].copy()
                if not reformados.empty:
                    reformados["_ano"] = pd.to_numeric(reformados["Ano_Alvara"], errors="coerce")
                    reformados = reformados.sort_values("_ano", ascending=False)
                    with st.expander(f"🔧 Ver {n_reformas} imóvel(is) com alvará de REFORMA"):
                        for _, r in reformados.drop_duplicates("Processo Aprova Digital").iterrows():
                            ano = int(r["_ano"]) if pd.notna(r["_ano"]) else "s/ data"
                            st.write(f"• **SQL {r['SQL']}** · {ano} · {r['Fase']} · {r['Segmento']}")

                st.caption("Fonte: alvarás do Aprova Digital (PMSP), 2021–2026, cruzados por SQL. "
                           "Um alvará pode cobrir vários lotes; a contagem é de processos únicos. "
                           "Alvará de reforma é sinal forte de retrofit.")
            else:
                st.markdown("### 🏗️ Alvarás de obra na região (Aprova Digital)")
                st.caption("Nenhum alvará de obra (2021–2026) encontrado para os imóveis "
                           "deste recorte.")

        # ---- ANÚNCIOS (PREÇO PEDIDO) E SPREAD CONTRA O TRANSACIONADO ----
        if ANUNCIOS_DF is not None:
            _modo_dist = (not rua) and (distrito_alvo != "Selecione...")
            _feat = DISTRITOS_GEO.get(str(distrito_alvo)) \
                if (_modo_dist and DISTRITOS_GEO is not None) else None
            _geom = _feat.get("geometry") if _feat else None
            _centro = [lat_c, lon_c] if (lat_c and lon_c) else None

            an = anuncios_da_regiao(
                ANUNCIOS_DF, tipo, ["Venda", "Aluguel"],
                centro=(None if _modo_dist else _centro),
                raio_m=(None if _modo_dist else raio),
                geom_dist=_geom,
            )

            st.markdown("### 📣 Anúncios na região (preço pedido)")
            if an is None or an.empty:
                st.caption("Nenhum anúncio da base (Matú Imóveis) nesta região "
                           "para o uso selecionado.")
            else:
                venda = an[(an["negocio"] == "Venda") & an["preco_m2_pedido"].notna()]
                med_ped = venda["preco_m2_pedido"].median() if not venda.empty else None
                med_tra = df["Preco_m2"].median() if "Preco_m2" in df.columns else None

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Anúncios (venda)", f"{int((an['negocio']=='Venda').sum())}")
                a2.metric("Mediana R$/m² pedido",
                          formata_moeda(med_ped) if med_ped else "—")
                a3.metric("Mediana R$/m² transacionado",
                          formata_moeda(med_tra) if med_tra else "—")
                if med_ped and med_tra and med_tra > 0:
                    spread = (med_ped / med_tra - 1) * 100
                    a4.metric("Spread pedido × transacionado", f"{spread:+.0f}%")
                else:
                    a4.metric("Spread pedido × transacionado", "—")

                with st.expander(f"Ver {len(an)} anúncio(s) desta região"):
                    _v = an.sort_values("preco_m2_pedido", ascending=False, na_position="last")
                    for _, a in _v.iterrows():
                        num = a.get("numero")
                        num_txt = ""
                        if pd.notna(num) and str(num) not in ("", "nan"):
                            flag = "" if a.get("numero_confiavel") in (True, "True") else "?"
                            num_txt = f", {int(float(num))}{flag}"
                        pm2 = a.get("preco_m2_pedido")
                        pm2_txt = f" · {formata_moeda(pm2)}/m²" if pd.notna(pm2) else ""
                        st.write(f"• **{a.get('logradouro','s/ endereço')}{num_txt}** "
                                 f"({a.get('subtipo','—')}, {a.get('negocio','')}) — "
                                 f"{formata_moeda(a.get('valor'))}{pm2_txt}")

                st.caption(
                    "Fonte: lista da Matú Imóveis, extraída em 10/07/2026. **Preço pedido**, "
                    "não fechado. O ITBI registra o **valor declarado**, normalmente abaixo do "
                    "preço real; o anúncio é o teto pretendido. Assim, o spread **superestima** "
                    "o desconto real e serve para comparar regiões entre si, não como "
                    "estimativa de desconto obtenível. Localização pelo CEP (precisão de rua)."
                )

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
                            props_z = fz.get("properties", {})
                            sig = str(props_z.get(CAMPO_ZONA, "—"))
                            cor = _cor_zona(sig)
                            siglas_presentes[sig] = cor

                            # monta o popup com as informações disponíveis da zona
                            descr = props_z.get("tx_zoneamento_perimetro")
                            obs = props_z.get("tx_observacao_perimetro")
                            lei = props_z.get("cd_numero_legislacao_zoneamento")
                            ano_lei = props_z.get("an_legislacao_zoneamento")
                            nome_zona = nome_familia_zona(sig)

                            linhas = [f"<b style='font-size:13px'>{sig}</b>"]
                            if nome_zona:
                                linhas.append(f"<span style='color:#555'>{nome_zona}</span>")
                            if descr and str(descr).strip() and str(descr) != sig:
                                linhas.append(f"<b>Descrição:</b> {descr}")
                            if obs and str(obs).strip() and str(obs).lower() != "none":
                                linhas.append(f"<b>Obs.:</b> {obs}")

                            # parâmetros construtivos do Quadro 3 (LPUOS 2016)
                            par = parametros_construtivos(sig)
                            if par:
                                linhas.append(
                                    "<hr style='margin:5px 0'>"
                                    "<b>Parâmetros construtivos (Quadro 3)</b>"
                                    f"<br>• Coef. aproveitamento: básico <b>{par['ca_bas']}</b>"
                                    f" · máximo <b>{par['ca_max']}</b>"
                                    f"<br>• Taxa de ocupação: {par['to_ate500']} (lote &lt;500m²)"
                                    f" · {par['to_500']} (≥500m²)"
                                    f"<br>• Gabarito de altura: <b>{par['gab']}</b> m"
                                )
                                if par.get("obs"):
                                    linhas.append(
                                        f"<span style='color:#777;font-size:10px'>{par['obs']}</span>"
                                    )
                                linhas.append(
                                    "<span style='color:#999;font-size:10px'>"
                                    "Fonte: Quadro 3 da Lei 16.402/2016 (LPUOS). "
                                    "Confirme exceções na lei.</span>"
                                )

                            if lei:
                                leg = f"Lei {lei}"
                                if ano_lei:
                                    leg += f"/{ano_lei}"
                                linhas.append(f"<span style='color:#777;font-size:11px'>{leg}</span>")
                            popup_z = "<br>".join(linhas)

                            folium.GeoJson(
                                fz,
                                style_function=lambda _f, _c=cor: {
                                    "color": _c, "weight": 1,
                                    "fill": True, "fillColor": _c, "fillOpacity": 0.25,
                                },
                                highlight_function=lambda _f: {
                                    "weight": 3, "fillOpacity": 0.45, "color": "#111",
                                },
                                tooltip=sig,
                                popup=folium.Popup(popup_z, max_width=280),
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
                            '<b>Zoneamento (LPUOS 2016)</b>'
                            '<div style="color:#777;margin:2px 0 4px">clique numa zona p/ detalhes</div>'
                            + itens + '</div>'
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
                elif heatmap_modo == "Preço/m² de terreno":
                    # ponderado pelo valor (regiões caras ficam quentes)
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

            # --- Camada de calor dos ANÚNCIOS (roxo/magenta), sobreponível ---
            if anuncios_heatmap != "Desligado" and HAS_HEATMAP and ANUNCIOS_DF is not None:
                geom_d = feature_dist.get("geometry") if (modo_distrito and feature_dist) else None
                an_heat = anuncios_da_regiao(
                    ANUNCIOS_DF, tipo, ["Venda"],
                    centro=(None if modo_distrito else centro),
                    raio_m=(None if modo_distrito else raio),
                    geom_dist=geom_d,
                )
                # gradiente roxo->magenta, distinto do azul->vermelho das transações
                grad_anuncios = {0.0: "#3f007d", 0.5: "#807dba",
                                 0.8: "#dd3497", 1.0: "#fa9fb5"}
                if an_heat is None or an_heat.empty:
                    st.caption("🟣 Sem anúncios de venda com coordenada nesta região para o "
                               "mapa de calor.")
                elif anuncios_heatmap == "Densidade de anúncios":
                    pontos = an_heat[["lat", "lon"]].dropna().values.tolist()
                    if pontos:
                        HeatMap(pontos, radius=20, blur=24, min_opacity=0.35,
                                gradient=grad_anuncios, name="Densidade anúncios").add_to(m)
                        st.caption("🟣 Calor **roxo/magenta = densidade de anúncios** (onde há "
                                   "mais imóveis à venda). Sobreponível ao calor das transações "
                                   "(azul→vermelho) para comparar oferta × negócios fechados.")
                elif anuncios_heatmap == "R$/m² pedido":
                    h = an_heat.dropna(subset=["lat", "lon", "preco_m2_pedido"]).copy()
                    h = h[h["preco_m2_pedido"] > 0]
                    if not h.empty:
                        lo, hi = h["preco_m2_pedido"].quantile([0.05, 0.95])
                        if hi <= lo:
                            lo, hi = h["preco_m2_pedido"].min(), h["preco_m2_pedido"].max()
                        peso = ((h["preco_m2_pedido"].clip(lo, hi) - lo) / (hi - lo)) \
                            if hi > lo else 1.0
                        h = h.assign(_peso=peso)
                        pontos = h[["lat", "lon", "_peso"]].values.tolist()
                        HeatMap(pontos, radius=22, blur=26, min_opacity=0.3,
                                gradient=grad_anuncios, name="R$/m² pedido").add_to(m)
                        st.caption("🟣 Calor **roxo/magenta = R$/m² pedido** (tons quentes = "
                                   "anúncios mais caros). Compare com o calor das transações "
                                   "(azul→vermelho): se as manchas coincidem, o preço pedido "
                                   "acompanha o transacionado; se a mancha roxa está mais quente "
                                   "onde a de transação é fria, há oferta acima do mercado ali.")
                elif anuncios_heatmap == "Valor total pedido":
                    h = an_heat.dropna(subset=["lat", "lon", "valor"]).copy()
                    h = h[h["valor"] > 0]
                    if not h.empty:
                        lo, hi = h["valor"].quantile([0.05, 0.95])
                        if hi <= lo:
                            lo, hi = h["valor"].min(), h["valor"].max()
                        peso = ((h["valor"].clip(lo, hi) - lo) / (hi - lo)) \
                            if hi > lo else 1.0
                        h = h.assign(_peso=peso)
                        pontos = h[["lat", "lon", "_peso"]].values.tolist()
                        HeatMap(pontos, radius=22, blur=26, min_opacity=0.3,
                                gradient=grad_anuncios, name="Valor total pedido").add_to(m)
                        st.caption("🟣 Calor **roxo/magenta = valor total pedido** (preço cheio "
                                   "do anúncio, não por m²). ⚠️ Como vários anúncios dividem o "
                                   "mesmo CEP, a mancha quente pode indicar imóveis caros **ou** "
                                   "muitos anúncios no mesmo ponto — leia junto com a densidade.")

            # --- Camadas por quarteirão: valorização (cor) + modernização (borda) ---
            # As duas são combináveis: cada quarteirão é desenhado UMA vez, com a
            # cor de preenchimento vinda da valorização e a borda vinda da modernização.
            if (mostrar_valorizacao or mostrar_modernizacao):
                if not QUADRAS_GEO:
                    st.caption("ℹ️ Arquivo de polígonos das quadras (quadras_sp.geojson) "
                               "não encontrado no repositório.")
                else:
                    # (a) valorização por quadra (só se marcada)
                    val_quadras = valorizacao_por_quadra(df_geo) if mostrar_valorizacao else {}
                    lim_val = None
                    if val_quadras:
                        _vals = np.array([v[0] for v in val_quadras.values()])
                        lim_val = max(abs(np.quantile(_vals, 0.10)),
                                      abs(np.quantile(_vals, 0.90)))
                        lim_val = lim_val if lim_val > 0 else (abs(_vals).max() or 0.01)

                    def _cor_val(v):
                        # vermelho (desvaloriza) -> cinza (estável) -> verde (valoriza)
                        t = max(-1.0, min(1.0, v / lim_val))
                        if t >= 0:
                            r = int(158 + (26 - 158) * t)
                            g = int(158 + (152 - 158) * t)
                            b = int(158 + (80 - 158) * t)
                        else:
                            r = int(158 + (215 - 158) * (-t))
                            g = int(158 + (48 - 158) * (-t))
                            b = int(158 + (39 - 158) * (-t))
                        return f"#{r:02x}{g:02x}{b:02x}"

                    # (b) modernização por quadra (só se marcada) — base ANTES do filtro de status
                    mod_quadras = {}
                    if mostrar_modernizacao:
                        codigos_visiveis = set(df_geo["Quadra"].dropna().astype(str)) \
                            if "Quadra" in df_geo.columns else set()
                        base_mod = df_sem_filtro_status
                        if codigos_visiveis and "Quadra" in base_mod.columns:
                            base_mod = base_mod[base_mod["Quadra"].astype(str).isin(codigos_visiveis)]
                        mod_quadras = modernizacao_por_quadra(base_mod)
                    lim_mod = None
                    if mod_quadras:
                        _counts = np.array([v[0] for v in mod_quadras.values()])
                        lim_mod = max(1.0, float(np.quantile(_counts, 0.90)))

                    # (c) conjunto de quadras a desenhar = união das duas métricas
                    codigos = set(val_quadras.keys()) | set(mod_quadras.keys())
                    desenhadas = 0
                    for cod in codigos:
                        geom = QUADRAS_GEO.get(cod)
                        if geom is None:
                            continue

                        # preenchimento pela valorização (cinza neutro se sem valor)
                        if cod in val_quadras:
                            val_aa, n_val = val_quadras[cod]
                            fill_cor = _cor_val(val_aa)
                            fill_op = 0.6
                        else:
                            val_aa, n_val = None, None
                            fill_cor = "#9e9e9e"
                            fill_op = 0.12  # quase transparente quando só há borda

                        # borda pela modernização (azul destacado; espessura ~ contagem)
                        if cod in mod_quadras:
                            n_mod, n_tot_mod = mod_quadras[cod]
                            intens = min(1.0, n_mod / lim_mod)
                            borda_cor = "#08519c"                 # azul forte
                            borda_peso = 1.5 + 3.5 * intens        # 1.5 a 5 px
                        else:
                            n_mod, n_tot_mod = None, None
                            borda_cor = "#00000030"                # borda neutra discreta
                            borda_peso = 0.4

                        # popup combinando o que houver
                        linhas_popup = [f"<b>Quarteirão:</b> {cod}"]
                        if val_aa is not None:
                            linhas_popup.append(f"<b>Valorização:</b> {val_aa*100:+.1f}% a.a. "
                                                f"({n_val} transações)")
                        if n_mod is not None:
                            pct = (n_mod / n_tot_mod * 100) if n_tot_mod else 0
                            linhas_popup.append(f"<b>Modernizados:</b> {n_mod} de {n_tot_mod} "
                                                f"({pct:.0f}%)")
                        popup_html = "<br>".join(linhas_popup)

                        folium.GeoJson(
                            {"type": "Feature", "geometry": geom, "properties": {}},
                            style_function=lambda _f, _fc=fill_cor, _fo=fill_op,
                                                   _bc=borda_cor, _bp=borda_peso: {
                                "color": _bc, "weight": _bp,
                                "fill": True, "fillColor": _fc, "fillOpacity": _fo,
                            },
                            popup=folium.Popup(popup_html, max_width=230),
                        ).add_to(m)
                        desenhadas += 1

                    # legenda/resumo conforme o que está ativo
                    partes_cap = []
                    if mostrar_valorizacao:
                        if val_quadras:
                            med = np.median([v[0] for v in val_quadras.values()]) * 100
                            partes_cap.append(f"**cor** = valorização (verde sobe, vermelho cai; "
                                              f"mediana {med:+.1f}% a.a.)")
                        else:
                            partes_cap.append("**cor** = valorização (sem quadras com dados "
                                              "suficientes neste recorte)")
                    if mostrar_modernizacao:
                        if mod_quadras:
                            total_mod = int(sum(v[0] for v in mod_quadras.values()))
                            partes_cap.append(f"**borda azul** = modernização "
                                              f"({total_mod} imóveis modernizados; borda mais "
                                              f"grossa = mais retrofit)")
                        else:
                            partes_cap.append("**borda azul** = modernização (nenhum modernizado "
                                              "neste recorte)")
                    if partes_cap:
                        st.caption("🧱 Camadas por quarteirão — " + " · ".join(partes_cap) + ".")

            # --- Camada de alvarás por quarteirão (círculos por família) ---
            if mostrar_alvaras:
                if ALVARAS_DF is None:
                    st.caption("ℹ️ Base de alvarás (alvaras_final.parquet) não encontrada "
                               "no repositório.")
                elif not QUADRAS_GEO:
                    st.caption("ℹ️ Arquivo de polígonos das quadras não encontrado.")
                elif not alvara_familias:
                    st.caption("ℹ️ Selecione ao menos uma família de alvará para exibir.")
                else:
                    # cores por família (reforma verde, nova roxo, demolição magenta)
                    CORES_FAM = {"Reforma": "#238b45",
                                 "Edificação Nova": "#6a51a3",
                                 "Demolição": "#c51b8a"}

                    # quadras visíveis no recorte
                    quadras_visiveis = set(df_geo["Quadra"].dropna().astype(str)) \
                        if "Quadra" in df_geo.columns else None

                    # conta cada família selecionada por quadra
                    contagens = {}   # familia -> {quadra: n}
                    for fam in alvara_familias:
                        contagens[fam] = alvaras_por_quadra(
                            ALVARAS_DF, familia=fam, quadras_visiveis=quadras_visiveis)

                    # vmax global (comparabilidade de tamanho entre famílias)
                    todos_vals = [n for c in contagens.values() for n in c.values()]
                    if not todos_vals:
                        st.caption("ℹ️ Nenhum alvará das famílias selecionadas nos "
                                   "quarteirões deste recorte.")
                    else:
                        vmax = max(todos_vals)

                        def _centroide(geom):
                            """Centroide simples (média dos vértices) de Polygon/MultiPolygon."""
                            try:
                                t = geom.get("type")
                                coords = geom.get("coordinates", [])
                                pts = []
                                if t == "Polygon":
                                    pts = coords[0]
                                elif t == "MultiPolygon":
                                    pts = coords[0][0]
                                if not pts:
                                    return None
                                lons = [p[0] for p in pts]
                                lats = [p[1] for p in pts]
                                return (sum(lats) / len(lats), sum(lons) / len(lons))
                            except Exception:
                                return None

                        # reúne, por quadra, a lista de (familia, n) a desenhar
                        por_quadra = {}   # quadra -> [(familia, n), ...]
                        for fam, cont in contagens.items():
                            for cod, n in cont.items():
                                por_quadra.setdefault(cod, []).append((fam, n))

                        # pré-computa os popups (uma passada na base, não uma por círculo).
                        # filtra a base só às quadras/famílias que serão desenhadas.
                        quadras_desenhar = set(por_quadra.keys())
                        base_pop = ALVARAS_DF[
                            ALVARAS_DF["Familia"].isin(alvara_familias)
                        ].copy()
                        base_pop["_quadra"] = base_pop["SQL"].astype(str).str[:6]
                        base_pop = base_pop[base_pop["_quadra"].isin(quadras_desenhar)]
                        popups = {}   # (quadra, familia) -> html
                        for (qd, fm), grp in base_pop.groupby(["_quadra", "Familia"]):
                            popups[(qd, fm)] = _html_popup_alvaras(qd, fm, grp)

                        desenhados = set()
                        for cod, itens in por_quadra.items():
                            geom = QUADRAS_GEO.get(cod)
                            if geom is None:
                                continue
                            centro = _centroide(geom)
                            if centro is None:
                                continue
                            # desenha do MAIOR para o menor, para o menor ficar visível por cima
                            for fam, n in sorted(itens, key=lambda x: x[1], reverse=True):
                                cor = CORES_FAM.get(fam, "#238b45")
                                raio = 4 + 14 * (n / vmax if vmax else 0)
                                folium.CircleMarker(
                                    location=centro, radius=raio,
                                    color=cor, weight=1,
                                    fill=True, fill_color=cor, fill_opacity=0.45,
                                    popup=folium.Popup(
                                        popups.get((cod, fam),
                                                   f"<b>Quarteirão {cod} — {fam}</b>"),
                                        max_width=280),
                                    tooltip=f"{n} × {fam}",
                                ).add_to(m)
                            desenhados.add(cod)

                        # legenda com totais por família
                        partes = []
                        nomes_cor = {"Reforma": "verde", "Edificação Nova": "roxo",
                                     "Demolição": "magenta"}
                        for fam in alvara_familias:
                            tot = sum(contagens[fam].values())
                            if tot:
                                partes.append(f"{fam} ({nomes_cor.get(fam,'')}): {tot}")
                        st.caption(f"🏗️ **Alvarás por quarteirão** — círculo maior = mais "
                                   f"alvarás · {len(desenhados)} quarteirões · "
                                   + " · ".join(partes)
                                   + ". Independe do filtro de status.")

            # --- Camada de anúncios (preço pedido), base separada ---
            anuncios_regiao = None
            linhas_tabela = []   # alimenta a tabela abaixo do mapa
            if (mostrar_anuncios or retrofit_ligado) and ANUNCIOS_DF is not None:
                geom_d = feature_dist.get("geometry") if (modo_distrito and feature_dist) else None
                # se retrofit está ligado, garante que Venda entra (a análise é de venda)
                negocios_mostrar = list(anuncio_negocios)
                if retrofit_ligado and "Venda" not in negocios_mostrar:
                    negocios_mostrar = negocios_mostrar + ["Venda"]
                anuncios_regiao = anuncios_da_regiao(
                    ANUNCIOS_DF, tipo, negocios_mostrar,
                    centro=(None if modo_distrito else centro),
                    raio_m=(None if modo_distrito else raio),
                    geom_dist=geom_d,
                )
                if anuncios_regiao is None or anuncios_regiao.empty:
                    st.caption("ℹ️ Nenhum anúncio da Matú nesta região com os filtros atuais.")
                else:
                    CORES_NEG = {"Venda": "black", "Aluguel": "cadetblue"}
                    # cores dos pinos por classificação de retrofit (folium.Icon)
                    CORES_RETROFIT = {"verde": "green", "amarelo": "orange",
                                      "vermelho": "red", "cinza": "lightgray"}
                    # carrega reformados só no modo "vizinhos" (no manual não são usados)
                    reformados_df = None
                    usa_manual = (retrofit_modo == "Valor que eu definir"
                                  and retrofit_venda_m2)
                    if retrofit_ligado and not usa_manual:
                        reformados_df = carregar_reformados(
                            PARQUET_GLOB, piso_m2=retrofit_piso, teto_m2=retrofit_teto,
                            ano_min=ano_min, ano_max=ano_max,
                            uso_sql=condicao_extra,
                            zonas_excl=tuple(zonas_excluidas) if zonas_excluidas else None)
                    contagem_retrofit = {"verde": 0, "amarelo": 0, "vermelho": 0, "cinza": 0}

                    for _, a in anuncios_regiao.iterrows():
                        pm2 = a.get("preco_m2_pedido")
                        pm2_txt = formata_moeda(pm2) if pd.notna(pm2) else "—"
                        num = a.get("numero")
                        num_txt = ""
                        if pd.notna(num) and str(num) not in ("", "nan"):
                            flag = "" if a.get("numero_confiavel") in (True, "True") else "?"
                            num_txt = f", {int(float(num))}{flag}"
                        quartos = " · ".join(
                            f"{int(a[c])} {n}" for c, n in
                            (("dorm", "dorm"), ("banh", "banh"), ("suite", "suíte"), ("vaga", "vaga"))
                            if pd.notna(a.get(c))
                        )
                        html = (
                            f"<div style='font-size:13px'>"
                            f"<b>{a.get('logradouro','s/ endereço')}{num_txt}</b><br>"
                            f"{a.get('subtipo','—')} · <b>{a.get('negocio','')}</b><br>"
                            f"{formata_moeda(a.get('valor'))}"
                            + (f" · <b>{pm2_txt}/m²</b>" if a.get("negocio") == "Venda" else "")
                            + f"<br><span style='color:#555'>{quartos}</span><br>"
                            f"<span style='color:#555'>constr. {a.get('area_construida','—')} m² · "
                            f"terreno {a.get('area_terreno','—')} m²</span>"
                        )

                        # classificação de retrofit (só venda)
                        cls = {}
                        cor_pino = CORES_NEG.get(a["negocio"], "gray")
                        tooltip = f"{a.get('negocio')} · {pm2_txt}/m²"
                        etiqueta_lucro = None   # etiqueta fixa no mapa (verde/amarelo)
                        if retrofit_ligado and a["negocio"] == "Venda":
                            cls = classificar_retrofit(
                                a["lat"], a["lon"],
                                valor_pedido=a.get("valor"),
                                area_construida=a.get("area_construida"),
                                area_terreno=a.get("area_terreno"),
                                reformados_df=reformados_df,
                                custo_obra_m2=retrofit_obra,
                                desconto=retrofit_desconto,
                                area_projetada=retrofit_area_proj,
                                valor_saida_fixo=(retrofit_venda_m2 if usa_manual else None))
                            cor = cls.get("cor", "cinza")
                            contagem_retrofit[cor] = contagem_retrofit.get(cor, 0) + 1
                            cor_pino = CORES_RETROFIT.get(cor, "lightgray")
                            if cor == "cinza":
                                html += ("<br><b>Retrofit:</b> sem dados suficientes "
                                         f"({cls.get('motivo','—')})")
                                tooltip = f"Retrofit: indefinido · {pm2_txt}/m²"
                            else:
                                nome_cor = {"verde": "🟢 VERDE", "amarelo": "🟡 AMARELO",
                                            "vermelho": "🔴 VERMELHO"}[cor]
                                # origem do preço de venda
                                if cls["modo"] == "manual":
                                    origem_venda = "definida por você"
                                else:
                                    origem_venda = (f"{cls['n_reformados']} reformados "
                                                    f"em {cls['raio_usado']}m")
                                # lente de leitura: R$/m² de terreno
                                linha_terreno = ""
                                if cls.get("compra_m2_terreno"):
                                    linha_terreno = (
                                        f" <span style='color:#555'>"
                                        f"({formata_moeda(cls['compra_m2_terreno'])}/m² terreno)"
                                        f"</span>")
                                # área usada na obra/venda
                                if cls["ampliou"]:
                                    linha_area = (
                                        f"<span style='color:#555'>área de venda: "
                                        f"{cls['area_venda']:.0f} m² "
                                        f"(projetada; atual {float(a['area_construida']):.0f})"
                                        f"</span><br>")
                                else:
                                    linha_area = (
                                        f"<span style='color:#555'>área de venda: "
                                        f"{cls['area_venda']:.0f} m² (atual)</span><br>")
                                html += (
                                    "<hr style='margin:4px 0'>"
                                    f"<b>Retrofit: {nome_cor}</b><br>"
                                    + linha_area
                                    + f"compra ({int(retrofit_desconto*100)}% desc.): "
                                    f"{formata_moeda(cls['compra'])}{linha_terreno}<br>"
                                    f"+ obra ({formata_moeda(retrofit_obra)}/m² × "
                                    f"{cls['area_venda']:.0f}): {formata_moeda(cls['obra'])}<br>"
                                    f"= custo total: <b>{formata_moeda(cls['custo_total'])}</b><br>"
                                    f"venda ({formata_moeda(cls['venda_m2'])}/m² constr., "
                                    f"{origem_venda}): {formata_moeda(cls['receita'])}<br>"
                                    "<hr style='margin:4px 0'>"
                                    f"<b>Lucro: {formata_moeda(cls['lucro'])} "
                                    f"({cls['lucro_pct']*100:+.1f}%)</b>"
                                )
                                tooltip = f"{nome_cor} · lucro {cls['lucro_pct']*100:+.1f}%"
                                # etiqueta fixa no mapa: só verdes e amarelas
                                if cor in ("verde", "amarelo") and retrofit_etiquetas:
                                    etiqueta_lucro = f"{cls['lucro_pct']*100:+.0f}%"
                        html += "</div>"

                        # --- alimenta a tabela abaixo do mapa ---
                        cls_t = cls if (retrofit_ligado and a["negocio"] == "Venda") else {}
                        linhas_tabela.append({
                            "Situação": {"verde": "🟢 Vale", "amarelo": "🟡 Apertado",
                                         "vermelho": "🔴 Não vale",
                                         "cinza": "⚪ Sem dados"}.get(cls_t.get("cor"), "—"),
                            "Lucro %": (round(cls_t["lucro_pct"] * 100, 1)
                                        if cls_t.get("lucro_pct") is not None else None),
                            "Lucro (R$)": cls_t.get("lucro"),
                            "Obra (R$)": cls_t.get("obra"),
                            "Custo total (R$)": cls_t.get("custo_total"),
                            "Endereço": (f"{a.get('logradouro','s/ endereço')}"
                                         f"{', ' + str(int(float(num))) if pd.notna(num) else ''}"),
                            "Tipo": a.get("subtipo"),
                            "Negócio": a.get("negocio"),
                            "Pedido (R$)": a.get("valor"),
                            "R$/m² pedido": (round(pm2) if pd.notna(pm2) else None),
                            "Compra c/ desc. (R$)": cls_t.get("compra"),
                            "Compra R$/m² terreno": (round(cls_t["compra_m2_terreno"])
                                                     if cls_t.get("compra_m2_terreno") else None),
                            "Venda estimada (R$)": cls_t.get("receita"),
                            "Venda R$/m² constr.": (round(cls_t["venda_m2"])
                                                    if cls_t.get("venda_m2") else None),
                            "Área de venda (m²)": cls_t.get("area_venda"),
                            "Área constr. (m²)": a.get("area_construida"),
                            "Área terreno (m²)": a.get("area_terreno"),
                            "Dorm": a.get("dorm"), "Banh": a.get("banh"),
                            "Suíte": a.get("suite"), "Vaga": a.get("vaga"),
                            "Bairro": a.get("bairro"), "CEP": a.get("cep"),
                            "Base da venda": ("definida" if cls_t.get("modo") == "manual"
                                              else (f"{cls_t['n_reformados']} ref. em "
                                                    f"{cls_t['raio_usado']}m"
                                                    if cls_t.get("n_reformados") else None)),
                        })

                        # Um marcador aceita só UM tooltip: quando há etiqueta de lucro,
                        # ela vira o tooltip permanente (fica visível sem clicar).
                        # Os detalhes seguem no popup (clique).
                        if etiqueta_lucro:
                            fundo = "#1b7837" if cor == "verde" else "#b8860b"
                            tip = folium.Tooltip(
                                f"<span style='background:{fundo};color:#fff;"
                                f"padding:1px 5px;border-radius:3px;font-weight:700;"
                                f"font-size:11px;white-space:nowrap'>"
                                f"{etiqueta_lucro}</span>",
                                permanent=True, direction="right", offset=(8, 0))
                        else:
                            tip = tooltip
                        folium.Marker(
                            location=[a["lat"], a["lon"]],
                            icon=folium.Icon(color=cor_pino, icon="home", prefix="fa"),
                            popup=folium.Popup(html, max_width=280),
                            tooltip=tip,
                        ).add_to(m)

                    n_v = int((anuncios_regiao["negocio"] == "Venda").sum())
                    n_a = int((anuncios_regiao["negocio"] == "Aluguel").sum())
                    if retrofit_ligado:
                        if usa_manual:
                            origem = (f"venda definida por você em "
                                      f"**{formata_moeda(retrofit_venda_m2)}/m²** construído")
                        else:
                            origem = ("R$/m² construído dos **reformados ao redor** "
                                      "(respeita período, uso e zonas excluídas)")
                        area_txt = (f"área de venda: **{retrofit_area_proj} m² projetados** "
                                    f"(ou a atual, se maior)"
                                    if retrofit_area_proj else
                                    "área de venda: a **atual** de cada imóvel")
                        st.caption(
                            f"🎯 **Retrofit** — {contagem_retrofit['verde']} 🟢 lucro ≥15% · "
                            f"{contagem_retrofit['amarelo']} 🟡 lucro 0–15% · "
                            f"{contagem_retrofit['vermelho']} 🔴 prejuízo · "
                            f"{contagem_retrofit['cinza']} ⚪ sem dados. "
                            f"**Lucro = (venda − compra − obra) ÷ (compra + obra)**. "
                            f"Compra = pedido −{int(retrofit_desconto*100)}% (mostrada também "
                            f"por m² de terreno); obra = {formata_moeda(retrofit_obra)}/m² × "
                            f"área de venda; venda = {origem}. {area_txt.capitalize()}. "
                            + ("A etiqueta no mapa mostra o lucro % (só verdes e amarelas); "
                               if retrofit_etiquetas else "")
                            + "Clique no pino para a conta completa; a tabela abaixo do "
                              "mapa lista tudo e pode ser ordenada.")
                    else:
                        st.caption(f"📣 **Anúncios (Matú)** — {len(anuncios_regiao)} nesta região "
                                   f"({n_v} venda, {n_a} aluguel). Pino preto = venda, "
                                   f"azul-petróleo = aluguel. Preço **pedido**; posição pelo CEP "
                                   f"(precisão de rua). Nº com “?” é de baixa confiança.")

            # --- Pontos de interesse (OpenStreetMap), opcional ---
            contagem_pois = {}
            if pois_selecionados:
                # bbox: no modo distrito usa o polígono; no raio, o alvo ± raio;
                # senão, a extensão dos comparáveis.
                poi_bb = None
                if feature_dist is not None:
                    poi_bb = bbox_de_feature(feature_dist)
                elif lat_c and lon_c:
                    d = raio / 111000.0  # ~graus por metro
                    poi_bb = (lon_c - d, lat_c - d, lon_c + d, lat_c + d)
                elif not df_geo.empty:
                    poi_bb = (df_geo["Longitude"].min(), df_geo["Latitude"].min(),
                              df_geo["Longitude"].max(), df_geo["Latitude"].max())

                if poi_bb is not None:
                    with st.spinner("Buscando pontos de interesse (OpenStreetMap)..."):
                        pois = buscar_pois_bbox(*poi_bb, tuple(sorted(pois_selecionados)))
                    if pois and "_erro" not in pois:
                        grupo_poi = folium.FeatureGroup(name="Pontos de interesse")
                        total_pois = 0
                        for cat, lista in pois.items():
                            meta = POI_CATEGORIAS[cat]
                            contagem_pois[cat] = len(lista)
                            total_pois += len(lista)
                            for lat, lon, nome in lista:
                                folium.Marker(
                                    [lat, lon],
                                    tooltip=f"{meta['label']}: {nome}",
                                    icon=folium.Icon(color=meta["cor"], icon=meta["icone"]),
                                ).add_to(grupo_poi)
                        grupo_poi.add_to(m)
                        if total_pois == 0:
                            st.caption("ℹ️ Nenhum ponto de interesse encontrado nesta área "
                                       "para as categorias selecionadas.")
                    else:
                        motivo = pois.get("_erro", "sem resposta") if pois else "sem resposta"
                        st.caption(f"ℹ️ Pontos de interesse indisponíveis no momento "
                                   f"({motivo}). O serviço público do OpenStreetMap costuma "
                                   f"oscilar — tente novamente em alguns segundos.")

            render_map(m)

            # --- Tabela dos anúncios da região (ordenável) ---
            if linhas_tabela:
                tab = pd.DataFrame(linhas_tabela)
                st.markdown("#### 📋 Anúncios da região (Matú)")
                if retrofit_ligado:
                    # ordem de leitura: primeiro o que interessa decidir
                    ordem = ["Situação", "Lucro %", "Lucro (R$)", "Endereço", "Tipo",
                             "Negócio", "Pedido (R$)", "R$/m² pedido",
                             "Compra c/ desc. (R$)", "Compra R$/m² terreno",
                             "Obra (R$)", "Custo total (R$)", "Venda estimada (R$)",
                             "Venda R$/m² constr.", "Área de venda (m²)",
                             "Área constr. (m²)", "Área terreno (m²)",
                             "Dorm", "Banh", "Suíte", "Vaga", "Bairro", "CEP",
                             "Base da venda"]
                    tab = tab.sort_values("Lucro %", ascending=False, na_position="last")
                    dica = ("Clique no cabeçalho de **Lucro %** ou **Obra (R$)** para "
                            "ordenar. Já vem ordenada por lucro, do melhor para o pior.")
                else:
                    ordem = ["Endereço", "Tipo", "Negócio", "Pedido (R$)", "R$/m² pedido",
                             "Área constr. (m²)", "Área terreno (m²)",
                             "Dorm", "Banh", "Suíte", "Vaga", "Bairro", "CEP"]
                    dica = ("Clique no cabeçalho de qualquer coluna para ordenar. "
                            "Ligue a **análise de retrofit** na barra lateral para ver "
                            "lucro, custo de obra e classificação.")
                tab = tab[[c for c in ordem if c in tab.columns]]

                # column_config formata sem quebrar a ordenação por clique no cabeçalho
                cfg = {}
                for c in ("Lucro (R$)", "Obra (R$)", "Custo total (R$)", "Pedido (R$)",
                          "Compra c/ desc. (R$)", "Venda estimada (R$)"):
                    if c in tab.columns:
                        cfg[c] = st.column_config.NumberColumn(c, format="R$ %.0f")
                for c in ("R$/m² pedido", "Compra R$/m² terreno", "Venda R$/m² constr."):
                    if c in tab.columns:
                        cfg[c] = st.column_config.NumberColumn(c, format="R$ %.0f")
                for c in ("Área de venda (m²)", "Área constr. (m²)", "Área terreno (m²)"):
                    if c in tab.columns:
                        cfg[c] = st.column_config.NumberColumn(c, format="%.0f m²")
                if "Lucro %" in tab.columns:
                    cfg["Lucro %"] = st.column_config.NumberColumn(
                        "Lucro %", format="%.1f%%",
                        help="(venda − compra − obra) ÷ (compra + obra)")

                st.dataframe(tab, use_container_width=True, hide_index=True,
                             height=380, column_config=cfg)
                st.caption(dica + "  Os valores seguem os parâmetros da barra lateral "
                                  "(desconto, obra, área projetada, preço de venda).")
                st.download_button(
                    "⬇️ Baixar esta tabela (CSV)",
                    tab.to_csv(index=False).encode("utf-8-sig"),
                    file_name="anuncios_regiao.csv", mime="text/csv")

            # contagem de POIs abaixo do mapa (indicador da região)
            if contagem_pois:
                st.markdown("#### 📍 Infraestrutura da região")
                cols_poi = st.columns(len(contagem_pois))
                for col, (cat, qtd) in zip(cols_poi, contagem_pois.items()):
                    col.metric(POI_CATEGORIAS[cat]["label"], qtd)
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
