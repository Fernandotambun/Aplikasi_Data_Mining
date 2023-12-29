import streamlit as st
import time
from web_function import predict

def app(df, x, y):
    
    if df is None or df.empty:
        st.warning("Please upload a CSV file to get started.")
        return

    st.title("Halaman Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        Kredit = st.number_input("Kredit")
    with col1:
        Sewa = st.number_input("Sewa")
    with col2:
        Tenor = st.number_input("Tenor")
    with col2:
        Tgk = st.number_input("Tunggakan")

    features = [Kredit, Sewa, Tenor, Tgk]

    if st.button("Predict"):
        progress_bar = st.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.02)  # Simulasi proses prediksi
            progress_bar.progress(percent_complete + 1)
        
        # Hilangkan elemen setelah proses prediksi selesai
        progress_bar.empty()

        prediction, score = predict(x, y, features)
        st.info("Prediksi Sukses")

        if prediction == 0:
            st.warning("Belum Lunas")
        else:
            st.success("Lunas")

        st.write("Model yang digunakan memiliki tingkat akurasi : ", (score * 100), "%")
