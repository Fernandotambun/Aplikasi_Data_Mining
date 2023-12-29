import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
import streamlit as st

from web_function import train_model

def app(df, x, y):
    if df is None or df.empty:
        st.warning("Please upload a CSV file to get started.")
        return
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi")

    st.write("Visualisasi data dapat membantu kita untuk memahami data yang kita miliki. Beberapa visualisasi yang dapat kita lakukan adalah sebagai berikut:")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x, y)

        # Buat prediksi menggunakan model
        y_pred = model.predict(x)

        # Dapatkan matriks kebingungan
        cm = confusion_matrix(y, y_pred)

        # Tampilkan matriks kebingungan menggunakan ConfusionMatrixDisplay
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cmd.plot(cmap='viridis', values_format='d', ax=plt.gca())
        
        st.pyplot()

        # Tambahkan metrik klasifikasi lanjutan ke dalam tabel
        st.subheader("Classification Metrics")
        metrics_data = {
            "Metric": ["Sensitivity (Recall)", "Specificity", "Precision", "Negative Predictive Value", "Accuracy"],
            "Value": [recall_score(y, y_pred), cm[0, 0] / (cm[0, 0] + cm[0, 1]), precision_score(y, y_pred),
                        cm[0, 0] / (cm[0, 0] + cm[1, 0]), accuracy_score(y, y_pred)]
        }
        metrics_table = pd.DataFrame(metrics_data)
        st.table(metrics_table)

        show_variable_explanation = st.checkbox("Show Variable Explanation")

        if show_variable_explanation:
            # Tambahkan penjelasan pada website dengan pembatas
            st.markdown("""
                - **Sensitivity (Recall):** Proporsi kasus positif yang berhasil diidentifikasi oleh model.
                - **Specificity:** Proporsi kasus negatif yang berhasil diidentifikasi oleh model.
                - **Precision:** Proporsi kasus positif yang diidentifikasi dengan benar dari semua kasus positif yang diidentifikasi oleh model.
                - **Negative Predictive Value:** Proporsi kasus negatif yang diidentifikasi dengan benar dari semua kasus negatif yang diidentifikasi oleh model.
                - **Accuracy:** Akurasi keseluruhan model.
            """)

            # Tambahkan penjelasan untuk TP, FN, FP, TN dengan nilai
            st.markdown("""
                - **True Positive (TP):** Jumlah kasus positif yang benar-benar diidentifikasi oleh model. (Nilai: {tp})
                - **False Negative (FN):** Jumlah kasus positif yang sebenarnya tetapi diidentifikasi sebagai negatif oleh model. (Nilai: {fn})
                - **False Positive (FP):** Jumlah kasus negatif yang sebenarnya tetapi diidentifikasi sebagai positif oleh model. (Nilai: {fp})
                - **True Negative (TN):** Jumlah kasus negatif yang benar-benar diidentifikasi oleh model. (Nilai: {tn})
            """.format(tp=cm[1, 1], fn=cm[1, 0], fp=cm[0, 1], tn=cm[0, 0]))

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=4, out_file=None, filled=True,rounded=True,
            feature_names=x.columns, class_names=['Belum Lunas','Lunas']
        )

        st.graphviz_chart(dot_data)
