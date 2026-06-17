import streamlit as st
import pandas as pd
import duckdb
from datetime import date

st.set_page_config(page_title="Valuation ITBI SP", layout="wide")
st.title("🏢 Ferramenta de Análise Home 2 Invest")

@st.cache_data
def obter_filtros_dinamicos():
    query = """
    SELECT DISTINCT "Descrição do uso (IPTU)", Bairro 
    FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true) 
    """
    df_filtros = duckdb.query(query).df()
    
    tipos = ["Todos"] + sorted([t for t in df_filtros['Descrição do uso (IPTU)'].unique() if t and str(t) != '-'])
    bairros = ["Todos"] + sorted([b for b in df_filtros['Bairro'].unique() if b and str(b) != '-'])
    
    return tipos, bairros

tipos_disp, bairros_disp = obter_filtros_dinamicos()

# --- BARRA LATERAL ---
st.sidebar.header("Filtros de Busca")

rua = st.sidebar.text_input("Nome da Rua / Logradouro")
bairro_selecionado = st.sidebar.selectbox("Bairro", bairros_disp)
tipo_selecionado = st.sidebar.selectbox("Tipo de Imóvel", tipos_disp)
mod_filtro = st.sidebar.selectbox("Estado de Conservação", ["Ambos", "Apenas Modernizadas", "Apenas Antigas"])

# --- LÓGICA DE BUSCA ---
if rua or bairro_selecionado != "Todos":
    with st.spinner("Cruzando dados históricos e atributos físicos da região..."):
        
        condicoes = []
        if rua:
            condicoes.append(f"LOWER(\"Nome do Logradouro\") LIKE '%{rua.lower()}%'")
        if bairro_selecionado != "Todos":
            b_safe = bairro_selecionado.replace("'", "''")
            condicoes.append(f"Bairro = '{b_safe}'")
            
        if tipo_selecionado != "Todos":
            condicoes.append(f"\"Descrição do uso (IPTU)\" = '{tipo_selecionado}'")
            
        if mod_filtro == "Apenas Modernizadas":
            condicoes.append("Modernizada = true")
        elif mod_filtro == "Apenas Antigas":
            condicoes.append("Modernizada = false")
            
        clausula_where = " AND ".join(condicoes)
        
        query_final = f"SELECT * FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true) WHERE {clausula_where}"
        
        try:
            dados_filtrados = duckdb.query(query_final).df()
            
            st.subheader(f"📊 Resultados: {len(dados_filtrados)} transações encontradas")
            
            if len(dados_filtrados) > 0:
                coluna_valor = 'Valor de Transação (declarado pelo contribuinte)'
                col_area_terreno = 'Área do Terreno (m2)'
                col_area_const = 'Área Construída (m2)'
                
                v_num = pd.to_numeric(dados_filtrados[coluna_valor], errors='coerce')
                a_terreno = pd.to_numeric(dados_filtrados[col_area_terreno], errors='coerce')
                a_const = pd.to_numeric(dados_filtrados[col_area_const], errors='coerce')
                
                df_calc = pd.DataFrame({'Valor': v_num, 'Area_T': a_terreno, 'Area_C': a_const})
                
                media_absoluta = df_calc['Valor'].mean()
                
                df_terreno_valido = df_calc[df_calc['Area_T'] > 0]
                media_m2_terreno = (df_terreno_valido['Valor'] / df_terreno_valido['Area_T']).mean() if not df_terreno_valido.empty else 0
                
                df_const_valido = df_calc[df_calc['Area_C'] > 0]
                media_m2_const = (df_const_valido['Valor'] / df_const_valido['Area_C']).mean() if not df_const_valido.empty else 0
                
                # --- CÁLCULO DE IDADE MÉDIA (NOVIDADE DO GEOSAMPA) ---
                idade_media_texto = "Sem dados"
                if 'Ano_Construcao_Geo' in dados_filtrados.columns:
                    anos_construcao = pd.to_numeric(dados_filtrados['Ano_Construcao_Geo'], errors='coerce')
                    anos_validos = anos_construcao[anos_construcao > 1500] # Ignora zeros ou erros antigos
                    
                    if not anos_validos.empty:
                        ano_atual = date.today().year
                        idade_media = ano_atual - anos_validos.mean()
                        idade_media_texto = f"{int(idade_media)} anos"

                # --- EXIBIÇÃO EM 4 COLUNAS VISUAIS ---
                col1, col2, col3, col4 = st.columns(4)
                
                def formata_moeda(valor):
                    return f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                
                col1.metric("Valor Médio (Absoluto)", formata_moeda(media_absoluta))
                
                if media_m2_terreno > 0:
                    col2.metric("Média / m² (Terreno)", formata_moeda(media_m2_terreno))
                else:
                    col2.metric("Média / m² (Terreno)", "-")
                    
                if media_m2_const > 0:
                    col3.metric("Média / m² (Construída)", formata_moeda(media_m2_const))
                else:
                    col3.metric("Média / m² (Construída)", "-")
                    
                col4.metric("Idade Média Oficial", idade_media_texto)
                
                st.markdown("---") 
                
                # --- TABELA DE DADOS ---
                df_visual = dados_filtrados.copy()
                
                def formatar_tabela(valor):
                    try:
                        v = float(valor)
                        if pd.isna(v): return "-"
                        return formata_moeda(v)
                    except:
                        return valor
                        
                df_visual[coluna_valor] = df_visual[coluna_valor].apply(formatar_tabela)
                st.dataframe(df_visual, use_container_width=True)
            else:
                st.warning("Nenhuma transação atende a todos os filtros simultaneamente.")
                
        except Exception as e:
            st.error(f"Erro ao processar consulta: {e}")
else:
    st.info("👈 Digite uma rua ou selecione um bairro no menu lateral para iniciar o Valuation.")
