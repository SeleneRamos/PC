"""
EDA Predios — Cercado de Lima  |  Ene–Mar 2025
Ejecutar: streamlit run eda_predios.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paleta ──────────────────────────────────────────────────────────────────
BG     = "#0F1117"
PANEL  = "#1A1D27"
ACCENT = "#4F8EF7"
GOLD   = "#F4B942"
RED    = "#E05C5C"
GREEN  = "#4FC78A"
PURPLE = "#9B72CF"
GREY   = "#8A8FA8"
WHITE  = "#E8EAF0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GREY, "axes.labelcolor": WHITE,
    "xtick.color": GREY, "ytick.color": GREY,
    "text.color": WHITE, "grid.color": "#2A2D3A",
    "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Streamlit config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Predios Cercado de Lima",
    page_icon="🏘️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      h1 { color: #4F8EF7; }
      h2 { color: #E8EAF0; border-bottom: 1px solid #2A2D3A; padding-bottom: 4px; }
      .stPlotlyChart, .stpyplot { background: #0F1117; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏘️ Análisis Exploratorio de Datos")
st.markdown(
    "**Predios — Cercado de Lima · Ene–Mar 2025** &nbsp;|&nbsp; "
    "Fuente: Portal Nacional de Datos Abiertos del Perú"
)

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [
        "num_registro", "fecha_adquisicion", "fecha_declaracion", "num_persona",
        "tipo_propietario", "pct_propiedad", "num_predio", "cod_uso_predio",
        "uso_predio", "area_terreno", "area_comun_terreno", "area_construida",
        "area_comun_construida", "area_total_construida", "pisos",
        "anio_construccion", "mayor_anio_construccion", "material_predio",
        "valor_terreno", "valor_construccion_dep", "valor_obras_comp",
        "autovaluo", "afecto_imp_predial",
    ]
    df["log_autovaluo"]         = np.log1p(df["autovaluo"])
    df["log_valor_terreno"]     = np.log1p(df["valor_terreno"])
    df["log_valor_construccion"]= np.log1p(df["valor_construccion_dep"])
    df["anio_adquisicion"]      = df["fecha_adquisicion"].dt.year
    df["decada"]                = (df["anio_construccion"].dropna() // 10 * 10).astype("Int64")
    return df


# ── Sidebar: carga o ruta ─────────────────────────────────────────────────────
st.sidebar.header("📂 Datos")
uploaded = st.sidebar.file_uploader(
    "Sube el archivo XLSX", type=["xlsx"],
    help="Predios_Cercado_de_Lima_Ene-Mar_2025.xlsx",
)
DEFAULT_PATH = "Predios_Cercado_de_Lima_Ene-Mar_2025_4__1_.xlsx"

if uploaded is not None:
    df = load_data(uploaded)
else:
    try:
        df = load_data(DEFAULT_PATH)
        st.sidebar.info(f"Usando archivo local: `{DEFAULT_PATH}`")
    except FileNotFoundError:
        st.error(
            "⚠️ No se encontró el archivo. Súbelo desde el sidebar o coloca "
            f"`{DEFAULT_PATH}` en el mismo directorio que este script."
        )
        st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total predios",       f"{len(df):,}")
c2.metric("Autovalúo mediano",   f"S/ {df['autovaluo'].median():,.0f}")
c3.metric("Autovalúo promedio",  f"S/ {df['autovaluo'].mean():,.0f}")
c4.metric("% Afectos IP",        f"{(df['afecto_imp_predial']=='S').mean()*100:.1f}%")
c5.metric("Valores nulos (año)", f"{df['anio_construccion'].isna().sum():,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def new_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(PANEL)
    return fig, ax

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Distribución autovalúo: original vs log
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("① Distribución del Autovalúo — escala original vs log₁₊ₓ")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "El autovalúo sigue una distribución de **cola pesada** (lognormal): "
        "la mayoría de predios se concentra en valores bajos, pero unos pocos "
        "predios comerciales o de muchos pisos disparan la media muy por encima "
        "de la mediana. La transformación **log₁₊ₓ** corrige la asimetría y "
        "mejora el desempeño de modelos de regresión."
    )

fig1, ax1a = plt.subplots(figsize=(12, 4.5), facecolor=BG)
ax1a.set_facecolor(PANEL)
ax1b = ax1a.twiny()

ax1a.hist(df["log_autovaluo"], bins=120, color=ACCENT, alpha=0.85, edgecolor="none")
ax1a.set_xlabel("log₁₊ₓ (Autovalúo)", color=WHITE, fontsize=10)
ax1a.set_ylabel("Frecuencia", color=WHITE, fontsize=10)
ax1a.tick_params(colors=GREY, labelsize=8)
ax1a.grid(axis="y", alpha=0.3)

vals_clip = df["autovaluo"].clip(upper=df["autovaluo"].quantile(0.995))
kde_x = np.linspace(vals_clip.min(), vals_clip.max(), 400)
kde   = stats.gaussian_kde(vals_clip, bw_method=0.1)
ax1b.plot(kde_x, kde(kde_x), color=GOLD, lw=2.2, alpha=0.9)
ax1b.set_xlabel("Autovalúo original (S/)", color=GOLD, fontsize=9)
ax1b.tick_params(axis="x", colors=GOLD, labelsize=7)
ax1b.set_ylim(bottom=0)
ax1b.yaxis.set_visible(False)
ax1b.spines["top"].set_visible(True); ax1b.spines["top"].set_color(GOLD)
ax1b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

med = df["autovaluo"].median()
ax1a.axvline(np.log1p(med), color=GREEN, lw=1.8, ls="--")
ax1a.text(np.log1p(med) + 0.08, ax1a.get_ylim()[1] * 0.82,
          f"Mediana\nS/ {med:,.0f}", color=GREEN, fontsize=8.5)

q75 = df["autovaluo"].quantile(0.75)
ax1a.axvline(np.log1p(q75), color=PURPLE, lw=1.5, ls=":")
ax1a.text(np.log1p(q75) + 0.08, ax1a.get_ylim()[1] * 0.60,
          f"P75\nS/ {q75:,.0f}", color=PURPLE, fontsize=8)

ax1a.set_title("Distribución del Autovalúo  ·  barras = log | curva = original",
               color=WHITE, fontsize=11, pad=10)
show(fig1)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2 — CASCADA
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("② Cascada — Descomposición del Autovalúo Promedio por Componente")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "El autovalúo es la suma de tres componentes. "
        "El **valor del terreno** suele ser el componente dominante en Cercado de Lima, "
        "zona con alta densidad y precio de suelo elevado. "
        "El valor de construcción depreciado refleja la antigüedad del parque inmobiliario."
    )

fig2, ax2 = plt.subplots(figsize=(9, 5), facecolor=BG)
ax2.set_facecolor(PANEL)

medias = [
    df["valor_terreno"].mean(),
    df["valor_construccion_dep"].mean(),
    df["valor_obras_comp"].mean(),
]
auto_mean = df["autovaluo"].mean()
labels_c  = ["Valor\nTerreno", "Valor\nConstrucción\ndep.", "Obras\nComp."]
colors_c  = [ACCENT, PURPLE, GREEN]

# barra total
ax2.bar(0, auto_mean, color="none", edgecolor=GOLD, linewidth=2.2, width=0.55, zorder=3)
ax2.text(0, auto_mean * 1.01, f"S/ {auto_mean:,.0f}",
         ha="center", va="bottom", color=GOLD, fontsize=9.5, fontweight="bold")

running = 0
for i, (h, c, lbl) in enumerate(zip(medias, colors_c, labels_c)):
    ax2.bar(i + 1, h, bottom=running, color=c, width=0.55, alpha=0.9, zorder=3)
    ax2.text(i + 1, running + h / 2,
             f"S/ {h:,.0f}\n({h/auto_mean*100:.1f}%)",
             ha="center", va="center", color=WHITE, fontsize=9, fontweight="bold")
    running += h

ax2.set_xticks(range(4))
ax2.set_xticklabels(["Autovalúo\nTotal"] + labels_c, color=WHITE, fontsize=9)
ax2.set_ylabel("S/ Promedio", color=WHITE, fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
ax2.grid(axis="y", alpha=0.3)
ax2.set_title("Cascada — Descomposición del Autovalúo Promedio", color=WHITE, fontsize=11, pad=10)
show(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 3 — Boxplot autovalúo × uso de predio
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("③ Boxplot: Autovalúo por Uso de Predio — Top 8 Categorías")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "Los locales de servicios y galerías comerciales tienen autovalúos medianos "
        "superiores a las viviendas, aunque con mayor dispersión. "
        "Los stands de mercado son los más homogéneos y baratos. "
        "Esta variable será un predictor categórico clave en el modelo."
    )

short_map = {
    "Vivienda": "Vivienda",
    "Puesto (o stand) en galería comercial": "Stand galería",
    "Otros usos comerciales no especificados": "Otros comercial",
    "Comercial - no identificado": "Comercial N/I",
    "Cochera": "Cochera",
    "Local de servicios (empresarial o profesional)": "Local servicios",
    "Almacén o depósito": "Almacén/Depósito",
    "Puesto (o stand) en mercado o campo ferial": "Stand mercado",
}
top8   = df["uso_predio"].value_counts().head(8).index.tolist()
df_box = df[df["uso_predio"].isin(top8)].copy()
df_box["uso_short"] = df_box["uso_predio"].map(short_map)
orden_short = (
    df_box.groupby("uso_short")["autovaluo"]
    .median().sort_values(ascending=False).index.tolist()
)

pal = sns.color_palette([ACCENT, GOLD, GREEN, PURPLE, RED, "#50C8C8", "#E87ED2", "#F48442"], 8)
fig3, ax3 = plt.subplots(figsize=(12, 5), facecolor=BG)
ax3.set_facecolor(PANEL)
sns.boxplot(data=df_box, x="log_autovaluo", y="uso_short", order=orden_short,
            palette=pal, ax=ax3, linewidth=0.8, width=0.6,
            flierprops=dict(marker=".", markersize=2, alpha=0.3, color=GREY))
ax3.set_xlabel("log₁₊ₓ (Autovalúo S/)", color=WHITE, fontsize=10)
ax3.set_ylabel("", color=WHITE)
ax3.tick_params(axis="y", labelsize=9, colors=WHITE)
ax3.grid(axis="x", alpha=0.3)
ax3.set_title("Autovalúo por Uso de Predio — Top 8 Categorías (escala log)",
              color=WHITE, fontsize=11, pad=10)
show(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 4 — Autovalúo mediano por década + IQR
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("④ Autovalúo Mediano por Década de Construcción + Banda IQR")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "Los predios de construcción más **reciente** (2000s–2020s) tienen autovalúos "
        "medianos más elevados, lo que tiene sentido por materiales modernos y mejor "
        "conservación. La banda IQR se amplía en décadas recientes: mayor **heterogeneidad** "
        "entre predios nuevos (desde estudios hasta edificios de lujo)."
    )

df_dec   = df.dropna(subset=["decada"]).copy()
dec_stat = (
    df_dec.groupby("decada")["autovaluo"]
    .agg(mediana="median",
         p25=lambda x: x.quantile(0.25),
         p75=lambda x: x.quantile(0.75),
         n="count")
    .reset_index()
)
dec_stat = dec_stat[dec_stat["decada"] >= 1900]

fig4, ax4 = plt.subplots(figsize=(12, 4.5), facecolor=BG)
ax4.set_facecolor(PANEL)
ax4.fill_between(dec_stat["decada"], dec_stat["p25"], dec_stat["p75"],
                 alpha=0.22, color=ACCENT, label="IQR (P25–P75)")
ax4.plot(dec_stat["decada"], dec_stat["mediana"],
         color=ACCENT, lw=2.5, marker="o", markersize=5, label="Mediana")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}s"))
ax4.set_xlabel("Década de construcción", color=WHITE, fontsize=10)
ax4.set_ylabel("Autovalúo (S/)", color=WHITE, fontsize=10)
ax4.legend(fontsize=9, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
ax4.grid(alpha=0.3)
ax4.set_title("Autovalúo Mediano por Década de Construcción + Banda IQR",
              color=WHITE, fontsize=11, pad=10)

ax4b = ax4.twinx()
ax4b.bar(dec_stat["decada"], dec_stat["n"], width=8, alpha=0.15, color=GOLD, zorder=0)
ax4b.set_ylabel("Nº predios", color=GOLD, fontsize=9)
ax4b.tick_params(axis="y", colors=GOLD, labelsize=7)
ax4b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
show(fig4)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 5 — Scatter log–log: terreno vs construcción × material
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("⑤ Scatter log–log: Valor Terreno vs Valor Construcción · por Material")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "Relación positiva esperada: predios de mayor valor de terreno también tienen "
        "mayor valor de construcción. **Concreto y Ladrillo** dominan el cuadrante de "
        "valores altos. **Adobe** se concentra en valores bajos en ambas dimensiones, "
        "lo que sugiere predios antiguos y deteriorados."
    )

n_sample = st.slider("Muestra para scatter (filas)", 5_000, 30_000, 15_000, 5_000)
df_sc  = df[(df["valor_terreno"] > 0) & (df["valor_construccion_dep"] > 0)].copy()
sample = df_sc.sample(n=min(n_sample, len(df_sc)), random_state=42)

mat_col = {"Ladrillo": ACCENT, "Concreto": GREEN, "Adobe": GOLD,
           "Madera u otros": RED, "No declarado": GREY}

fig5, ax5 = plt.subplots(figsize=(12, 5), facecolor=BG)
ax5.set_facecolor(PANEL)
for mat, color in mat_col.items():
    sub = sample[sample["material_predio"] == mat]
    ax5.scatter(sub["log_valor_terreno"], sub["log_valor_construccion"],
                s=5, alpha=0.35, color=color, label=mat, linewidths=0, rasterized=True)

m, b, r, *_ = stats.linregress(sample["log_valor_terreno"], sample["log_valor_construccion"])
xr = np.linspace(sample["log_valor_terreno"].min(), sample["log_valor_terreno"].max(), 200)
ax5.plot(xr, m * xr + b, color=WHITE, lw=1.8, ls="--", alpha=0.8, label=f"OLS  r={r:.2f}")

ax5.set_xlabel("log₁₊ₓ (Valor Terreno)", color=WHITE, fontsize=10)
ax5.set_ylabel("log₁₊ₓ (Valor Construcción dep.)", color=WHITE, fontsize=10)
ax5.set_title("Scatter log–log: Valor Terreno vs Construcción  ·  por Material",
              color=WHITE, fontsize=11, pad=10)
ax5.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE,
           markerscale=2.5, scatterpoints=1)
ax5.grid(alpha=0.25)
show(fig5)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 6 — Heatmap % afecto IP: material × tipo propietario
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("⑥ Heatmap: % Afecto al Impuesto Predial — Material × Tipo Propietario")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "El porcentaje de predios afectos al IP es casi universal (≥90%) para "
        "**Propietario Único** y **Condómino**, independiente del material. "
        "Los **Poseedores** muestran tasas más bajas, posiblemente por situación "
        "jurídica no resuelta. Esta variable podría usarse como target de clasificación."
    )

pivot = df.pivot_table(
    values="afecto_imp_predial", index="material_predio", columns="tipo_propietario",
    aggfunc=lambda x: (x == "S").mean() * 100,
)
col_order = ["Propietario Único", "Condómino", "Poseedor", "Responsable", "Concesionario"]
pivot = pivot.reindex(columns=col_order, fill_value=np.nan)

fig6, ax6 = plt.subplots(figsize=(10, 4), facecolor=BG)
ax6.set_facecolor(PANEL)
sns.heatmap(pivot, ax=ax6, cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.5, linecolor=BG, mask=pivot.isnull(),
            annot_kws={"size": 10, "color": "white", "fontweight": "bold"},
            cbar_kws={"shrink": 0.8, "label": "% Afecto IP"})
ax6.set_title("% Afecto al Impuesto Predial:  Material × Tipo Propietario",
              color=WHITE, fontsize=11, pad=10)
ax6.set_xlabel("Tipo Propietario", color=WHITE, fontsize=9)
ax6.set_ylabel("Material", color=WHITE, fontsize=9)
ax6.tick_params(colors=WHITE, labelsize=9)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
ax6.collections[0].colorbar.ax.tick_params(colors=WHITE, labelsize=8)
ax6.collections[0].colorbar.ax.yaxis.label.set_color(WHITE)
show(fig6)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 7 — Correlaciones con log(autovalúo)
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("⑦ Correlaciones con Autovalúo — Escala Original vs Log")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "En escala **log** las correlaciones son más altas y lineales, confirmando que "
        "la transformación es necesaria para el modelado. "
        "**Valor terreno** y **valor construcción** lideran la correlación. "
        "El **año de construcción** tiene correlación positiva: predios más nuevos valen más."
    )

num_cols = [
    "area_terreno", "area_construida", "area_total_construida", "pisos",
    "anio_construccion", "pct_propiedad", "valor_terreno",
    "valor_construccion_dep", "valor_obras_comp",
]
labels_s = [
    "Área terreno", "Área construida", "Área total", "Pisos",
    "Año constr.", "Pct prop.", "Val. terreno", "Val. constr.", "Obras comp.",
]

corr_orig = [df[c].corr(df["autovaluo"])     for c in num_cols]
corr_log  = [df[c].corr(df["log_autovaluo"]) for c in num_cols]

y_pos = np.arange(len(num_cols))
bh    = 0.35

fig7, ax7 = plt.subplots(figsize=(12, 5), facecolor=BG)
ax7.set_facecolor(PANEL)
ax7.barh(y_pos + bh / 2, corr_log,  bh, color=ACCENT, alpha=0.88, label="vs log(autovalúo)")
ax7.barh(y_pos - bh / 2, corr_orig, bh, color=GOLD,  alpha=0.88, label="vs autovalúo original")

for i, (co, cl) in enumerate(zip(corr_orig, corr_log)):
    ax7.text(max(cl, 0) + 0.01, i + bh / 2, f"{cl:.2f}", va="center", fontsize=7.5, color=ACCENT)
    ax7.text(max(co, 0) + 0.01, i - bh / 2, f"{co:.2f}", va="center", fontsize=7.5, color=GOLD)

ax7.axvline(0, color=GREY, lw=0.8)
ax7.set_yticks(y_pos); ax7.set_yticklabels(labels_s, color=WHITE, fontsize=9)
ax7.set_xlabel("Correlación de Pearson", color=WHITE, fontsize=10)
ax7.set_title("Correlaciones con Autovalúo  ·  original vs log",
              color=WHITE, fontsize=11, pad=10)
ax7.legend(fontsize=9, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
ax7.grid(axis="x", alpha=0.3)
ax7.set_xlim(-0.35, 1.1)
show(fig7)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 8 — Adquisiciones por año + autovalúo mediano del cohorte
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("⑧ Adquisiciones por Año (2000–2024) + Autovalúo Mediano del Cohorte")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "Pico de transacciones alrededor de **2005–2010**, período de expansión del "
        "crédito hipotecario en Perú. Los cohortes de adquisición más recientes "
        "presentan autovalúos medianos más altos, coherente con la apreciación "
        "acumulada del mercado inmobiliario limeño."
    )

col_a, col_b = st.columns(2)
yr_min = int(col_a.number_input("Año inicio", 2000, 2023, 2000))
yr_max = int(col_b.number_input("Año fin",    2001, 2024, 2024))

df_aq  = df[(df["anio_adquisicion"] >= yr_min) & (df["anio_adquisicion"] <= yr_max)]
aq_st  = (
    df_aq.groupby("anio_adquisicion")
    .agg(n=("num_predio", "count"), mediana=("autovaluo", "median"))
    .reset_index()
)

fig8, ax8 = plt.subplots(figsize=(12, 4.5), facecolor=BG)
ax8.set_facecolor(PANEL)
ax8.bar(aq_st["anio_adquisicion"], aq_st["n"],
        color=PURPLE, alpha=0.75, width=0.75, label="Nº predios adquiridos")
ax8b = ax8.twinx()
ax8b.plot(aq_st["anio_adquisicion"], aq_st["mediana"],
          color=GOLD, lw=2.5, marker="o", markersize=5, label="Autovalúo mediano")
ax8b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
ax8b.set_ylabel("Autovalúo mediano (S/)", color=GOLD, fontsize=9)
ax8b.tick_params(axis="y", colors=GOLD, labelsize=8)
ax8.set_xlabel("Año de adquisición", color=WHITE, fontsize=10)
ax8.set_ylabel("Nº predios", color=PURPLE, fontsize=9)
ax8.tick_params(axis="y", colors=PURPLE, labelsize=8)
ax8.tick_params(axis="x", labelsize=9, rotation=45)
ax8.set_title(f"Adquisiciones {yr_min}–{yr_max}  ·  Autovalúo Mediano del Cohorte",
              color=WHITE, fontsize=11, pad=10)
ax8.grid(axis="y", alpha=0.25)
h1, l1 = ax8.get_legend_handles_labels()
h2, l2 = ax8b.get_legend_handles_labels()
ax8.legend(h1 + h2, l1 + l2, fontsize=8.5, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
show(fig8)


# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICO 9 — Heatmap autovalúo mediano: material × pisos
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("⑨ Autovalúo Mediano (miles S/) — Material × Número de Pisos")

with st.expander("ℹ️ Interpretación", expanded=False):
    st.markdown(
        "Interacción clave para el modelado: **Concreto × muchos pisos** alcanza "
        "los autovalúos más altos. **Adobe** se mantiene bajo sin importar la altura, "
        "lo que puede indicar conversión informal de materiales. "
        "Esta combinación es un excelente **feature de interacción** para los modelos."
    )

max_pisos = st.slider("Máximo de pisos a mostrar", 5, 20, 10)
top_p   = list(range(1, max_pisos + 1))
df_hp   = df[df["pisos"].isin(top_p)].copy()
pivot2  = (
    df_hp.pivot_table(values="autovaluo", index="material_predio",
                      columns="pisos", aggfunc="median") / 1e3
)
mat_ord = ["Concreto", "Ladrillo", "Adobe", "Madera u otros", "No declarado"]
pivot2  = pivot2.reindex(mat_ord)

fig9, ax9 = plt.subplots(figsize=(14, 4), facecolor=BG)
ax9.set_facecolor(PANEL)
sns.heatmap(pivot2, ax=ax9, cmap="magma", annot=True, fmt=".0f",
            linewidths=0.6, linecolor=BG, mask=pivot2.isnull(),
            annot_kws={"size": 9, "color": "white", "fontweight": "bold"},
            cbar_kws={"shrink": 0.6, "label": "Autovalúo mediano (miles S/)"})
ax9.set_title("Autovalúo Mediano (miles S/)  ·  Material × Número de Pisos",
              color=WHITE, fontsize=11, pad=10)
ax9.set_xlabel("Número de pisos", color=WHITE, fontsize=10)
ax9.set_ylabel("Material del predio", color=WHITE, fontsize=10)
ax9.tick_params(colors=WHITE, labelsize=9)
ax9.collections[0].colorbar.ax.tick_params(colors=WHITE, labelsize=8)
ax9.collections[0].colorbar.ax.yaxis.label.set_color(WHITE)
show(fig9)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Proyecto Integrador de Ciencia de Datos · Estadística Informática · 2025  |  "
    "Datos: Portal Nacional de Datos Abiertos del Perú"
)
