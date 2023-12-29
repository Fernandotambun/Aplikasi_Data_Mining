import streamlit as st
from web_function import load_data
import matplotlib.pyplot as plt
import seaborn as sns

def app(df, x, y):
    st.title("Selamat Datang di Aplikasi Prediksi Kredit")

    # Deskripsi singkat tentang aplikasi
    st.markdown("""
    Aplikasi ini dirancang untuk memprediksi status kredit berdasarkan beberapa faktor kunci.
    Anda dapat menggunakan fitur-fitur seperti prediksi, visualisasi, dan mengunggah data kredit Anda sendiri untuk dianalisis.
    """)

    # Cek apakah data tersedia
    if df is not None:

        # Visualisasi data berdasarkan status kredit
        st.subheader("Visualisasi Data Berdasarkan Status Kredit")
        target_counts = y.value_counts()

        # Bar chart
        fig, ax = plt.subplots()
        target_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Jumlah Sampel Berdasarkan Status Kredit')
        ax.set_xlabel('Status Kredit')
        ax.set_ylabel('Jumlah Sampel')
        st.pyplot(fig)
        st.markdown("""
        Grafik di atas menunjukkan sebaran jumlah sampel berdasarkan status kredit. Ini membantu memahami seimbang atau tidaknya dataset tergantung pada jumlah sampel dalam setiap kategori status kredit.
        """)

        # Pie chart untuk proporsi status kredit
        st.subheader("Pie Chart Proporsi Status Kredit")
        fig, ax = plt.subplots()
        ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('Proporsi Status Kredit')
        st.pyplot(fig)
        st.markdown("""
        Pie chart di atas memperlihatkan proporsi masing-masing kategori pada variabel target 'Status'. Ini membantu memahami sebaran kelas atau status kredit dalam dataset.
        """)
    else:
        st.warning("Please upload a CSV file to get started.")
