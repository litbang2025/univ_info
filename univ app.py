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
# Bersihkan nama kolom dari spasi ekstra
df.columns = df.columns.str.strip()

text_columns = ['Institution', 'Country', 'Study', 'Bagian', 'Schp']
numeric_columns = [col for col in df.columns if col not in text_columns and col != '#']

# Konversi kolom numerik
cols_to_convert = ['Academic', 'Employer', 'Citations', 'H', 'IRN', 'Score']
for col in cols_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

st.write("🧾 Tipe Data Setelah Dibersihkan:")
st.write(df.dtypes)
st.dataframe(df.head())

# ===============================
# 3. 🏅 PENGURUTAN DATA
# ===============================
df_sorted = df.sort_values(by='Academic', ascending=False).reset_index(drop=True)
top_10 = df_sorted.head(10)
bottom_10 = df_sorted.tail(10)
middle_index = len(df_sorted) // 2
middle_10 = df_sorted.iloc[middle_index-5:middle_index+5]

# ===============================
# 4. 📌 TAMPILKAN RINGKASAN
# ===============================
st.subheader("🏆 Top 10 Universitas:")
st.dataframe(top_10[['Institution', 'Country', 'Study', 'AR Rank']])

st.subheader("🎯 Middle 10 Universitas:")
st.dataframe(middle_10[['Institution', 'Study', 'AR Rank']])

st.subheader("🔻 Bottom 10 Universitas:")
st.dataframe(bottom_10[['Institution', 'Study', 'AR Rank']])

# ===============================
# 5. 📊 VISUALISASI GRAFIK BAR
# ===============================
def plot_metrics(df_subset, title_prefix):
    metrics = ['Academic', 'Employer', 'Citations', 'H', 'IRN', 'Score']
    
    for metric in metrics:
        if metric in df_subset.columns:
            sorted_df = df_subset.sort_values(by=metric, ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=metric, y='Institution', data=sorted_df, palette='viridis')
            plt.title(f'{title_prefix} 10 Universitas berdasarkan {metric}')
            plt.xlabel(metric)
            plt.ylabel('Institution')
            for i, (value, name) in enumerate(zip(sorted_df[metric], sorted_df['Institution'])):
                plt.text(value, i, f'{value:.2f}', va='center', ha='left', fontsize=9)
            st.pyplot(plt)

plot_metrics(top_10, '🏆 Top')
plot_metrics(middle_10, '🎯 Middle')
plot_metrics(bottom_10, '🔻 Bottom')

# ===============================
# 6. 📈 ANALISIS STATISTIK
# ===============================
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("📊 Statistik Deskriptif:")
st.dataframe(df[numeric_columns].describe())

# Nilai tertinggi dan terendah per metrik
st.subheader("🏅 Nilai Tertinggi dan Terendah per Metrik:")
for metric in numeric_columns:
    if df[metric].notna().any():
        top_uni = df.loc[df[metric].idxmax(), 'Institution']
        bottom_uni = df.loc[df[metric].idxmin(), 'Institution']
        st.write(f"{metric}: tertinggi = {top_uni} ({df[metric].max():.2f}), terendah = {bottom_uni} ({df[metric].min():.2f})")

# Korelasi antar metrik
st.subheader("📌 Korelasi antar Metrik")
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔍 Korelasi antar metrik")
plt.tight_layout()
st.pyplot(plt)

# Interpretasi visual korelasi
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
