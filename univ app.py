import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_1samp

# Styling visual
sns.set(style="whitegrid")
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ===============================
# 1. 📥 BACA DATA
# ===============================
st.title("📊 Analisis Universitas Dunia")

try:
    df = pd.read_csv('univ_data.csv')
    st.success("Data berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

st.write("📊 Data Analisis Awal:")
st.dataframe(df.head())

st.write("🔢 Kolom yang Tersedia:", df.columns.tolist())

# ===============================
# 2. 🧼 BERSIHKAN DAN UBAH TIPE DATA
# ===============================
df.columns = df.columns.str.strip()

text_columns = ['Institution', 'Country', 'Study', 'Bagian', 'Schp']
numeric_columns = [col for col in df.columns if col not in text_columns and col != '#']

cols_to_convert = ['Academic', 'Employer', 'Citations', 'H', 'IRN', 'Score']
for col in cols_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

#st.write("🧾 Tipe Data Setelah Dibersihkan:")
#st.write(df.dtypes)
st.dataframe(df.head())

# ===============================
# 3. 🏅 PENGURUTAN DATA
# ===============================
if 'Academic' in df.columns:
    df_sorted = df.sort_values(by='Academic', ascending=False).reset_index(drop=True)
    top_10 = df_sorted.head(10)
    bottom_10 = df_sorted.tail(10)
    middle_index = len(df_sorted) // 2
    middle_10 = df_sorted.iloc[middle_index-5:middle_index+5]
else:
    st.warning("Kolom 'Academic' tidak ditemukan untuk pengurutan.")
    st.stop()

# ===============================
# 4. 📌 TAMPILKAN RINGKASAN
# ===============================
def safe_display(df_subset, cols, title):
    available_cols = [col for col in cols if col in df_subset.columns]
    if available_cols:
        st.subheader(title)
        st.dataframe(df_subset[available_cols])
    else:
        st.warning(f"Kolom tidak ditemukan: {cols}")

safe_display(top_10, ['Institution', 'Country', 'Study', 'AR Rank'], "🏆 Top 10 Universitas:")
safe_display(middle_10, ['Institution', 'Study', 'AR Rank'], "🎯 Middle 10 Universitas:")
safe_display(bottom_10, ['Institution', 'Study', 'AR Rank'], "🔻 Bottom 10 Universitas:")

# ===============================
# 5. 📊 VISUALISASI GRAFIK BAR
# ===============================
def plot_metrics(df, title):
    if df.empty:
        st.warning("⚠️ Tidak ada data untuk ditampilkan.")
        return

    metrics = ['Academic', 'Employer', 'Citations', 'H', 'IRN', 'Score']
    available_metrics = [col for col in metrics if col in df.columns]

    if not available_metrics:
        st.warning("⚠️ Tidak ada metrik yang tersedia untuk diplot.")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = df.set_index('Institution')[available_metrics]
        df_plot.plot(kind='bar', ax=ax)
        ax.set_title(f"{title} Universities by Metric")
        ax.set_ylabel("Nilai")
        ax.set_xticklabels(df['Institution'], rotation=45, ha='right')

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi error saat membuat grafik: {e}")

    
    for metric in metrics:
        if metric in df_subset.columns:
            sorted_df = df_subset.sort_values(by=metric, ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=metric, y='Institution', data=sorted_df, palette='viridis')
            plt.title(f'{title_prefix} 10 Universitas berdasarkan {metric}')
            plt.xlabel(metric)
            plt.ylabel('Institution')
            for i, (value, name) in enumerate(zip(sorted_df[metric], sorted_df['Institution'])):
                if not pd.isna(value):
                    plt.text(value, i, f'{value:.2f}', va='center', ha='left', fontsize=9)
            st.pyplot(plt.gcf())
            plt.clf()
            plt.close()

plot_metrics(top_10, '🏆 Top')
plot_metrics(middle_10, '🎯 Middle')
plot_metrics(bottom_10, '🔻 Bottom')

# ===============================
# 6. 📈 ANALISIS STATISTIK
# ===============================
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("📊 Statistik Deskriptif:")
st.dataframe(df[numeric_columns].describe())

st.subheader("🏅 Nilai Tertinggi dan Terendah per Metrik:")
for metric in numeric_columns:
    if df[metric].notna().any():
        top_uni = df.loc[df[metric].idxmax(), 'Institution']
        bottom_uni = df.loc[df[metric].idxmin(), 'Institution']
        st.write(f"{metric}: tertinggi = {top_uni} ({df[metric].max():.2f}), terendah = {bottom_uni} ({df[metric].min():.2f})")

# ===============================
# Korelasi antar metrik
# ===============================
st.subheader("📌 Korelasi antar Metrik")
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔍 Korelasi antar metrik")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()
plt.close()

# Interpretasi
def interpret_correlation_visual(value):
    if value == 1:
        return "🟢 Sempurna (positif)"
    elif value >= 0.7:
        return "🟢 Kuat (positif)"
    elif value >= 0.3:
        return "🟡 Sedang (positif)"
    elif value >= 0.1:
        return "🟠 Lemah (positif)"
    elif value > -0.1:
        return "⚪ Tidak ada korelasi"
    elif value > -0.3:
        return "🔵 Lemah (negatif)"
    elif value > -0.7:
        return "🔵 Sedang (negatif)"
    else:
        return "🔴 Kuat (negatif)"

st.subheader("📊 Interpretasi Visual Korelasi:")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        value = corr_matrix.iloc[i, j]
        st.write(f"{col1} ↔ {col2}: {value:.2f} {interpret_correlation_visual(value)}")


# ===============================
# ===============================
# 📌 FILTER BERDASARKAN NEGARA
# ===============================
st.sidebar.header("🔍 Filter Data")

if 'Country' in df.columns:
    negara_list = df['Country'].dropna().unique().tolist()
    negara_terpilih = st.sidebar.multiselect("Pilih Negara:", sorted(negara_list), default=negara_list[:3])

    if negara_terpilih:
        df_filtered = df[df['Country'].isin(negara_terpilih)]
    else:
        df_filtered = df.copy()
        st.warning("⚠️ Tidak ada negara yang dipilih. Menampilkan semua data.")

    st.subheader("📌 Data Setelah Difilter:")
    st.dataframe(df_filtered[['Institution', 'Country', 'Study', 'Academic', 'Employer', 'Score']])
else:
    st.error("Kolom 'Country' tidak ditemukan dalam data.")
    df_filtered = df.copy()


# ===============================
# 📈 TREN TAHUNAN (JIKA ADA KOLOM TAHUN)
# ===============================
if 'Year' in df_filtered.columns:
    st.subheader("📈 Tren Skor Rata-rata per Tahun")

    df_tren = df_filtered[['Year', 'Score']].dropna()
    if not df_tren.empty:
        df_tren_grouped = df_tren.groupby('Year')['Score'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x='Year', y='Score', data=df_tren_grouped, marker='o')
        plt.title("Rata-rata Score per Tahun")
        plt.xlabel("Tahun")
        plt.ylabel("Score")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Data tren tahun tidak tersedia.")
else:
    st.info("📅 Kolom 'Year' tidak tersedia dalam dataset.")


# ===============================
# 🎓 PERBANDINGAN DUA UNIVERSITAS
# ===============================
st.sidebar.subheader("🎓 Bandingkan Dua Universitas")

if 'Institution' in df_filtered.columns:
    daftar_univ = df_filtered['Institution'].dropna().unique().tolist()

    if len(daftar_univ) >= 2:
        univ_1 = st.sidebar.selectbox("Universitas Pertama", sorted(daftar_univ))
        univ_2 = st.sidebar.selectbox("Universitas Kedua", sorted(daftar_univ), index=1)

        df_compare = df_filtered[df_filtered['Institution'].isin([univ_1, univ_2])]
        metrics_to_plot = ['Academic', 'Employer', 'Citations', 'H', 'IRN', 'Score']

        try:
            df_plot = df_compare.set_index('Institution')[metrics_to_plot].T
            st.subheader(f"📊 Perbandingan Antara {univ_1} dan {univ_2}")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_plot.plot(kind='bar', ax=ax)
            plt.title(f"Perbandingan Metrik")
            plt.ylabel("Nilai")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal menampilkan perbandingan: {e}")
    else:
        st.warning("⚠️ Minimal harus ada dua universitas untuk dibandingkan.")
else:
    st.error("Kolom 'Institution' tidak tersedia.")

