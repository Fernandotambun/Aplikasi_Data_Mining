# main.py
import streamlit as st
from web_function import load_data, train_model, predict
from streamlit_option_menu import option_menu
from Tabs import home, predict, visualise

st.set_page_config(
    page_title="Aplikasi Prediksi Kredit",
    page_icon="🧠",
    initial_sidebar_state="auto",
)

Tabs = {
    "Home": home,
    "Prediction": predict,
    "Visualisation": visualise
}

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])



if uploaded_file is not None:
    # Load data and perform preprocessing
    df, x, y = load_data(uploaded_file)
    st.success("File uploaded successfully!")

    # Display original data
    st.subheader("Original Data:")
    st.write(df)
    
    # Display information about the data
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
else:
    # If no file is uploaded, show a message
    df, x, y = None, None, None
    st.info("Please upload a CSV file to get started.")

# Display checkbox for data upload rules
if st.checkbox("File Upload Rules 📋"):
    st.markdown("""
        - File must be a CSV file.
        - File must have columns with the following names:
            - Kredit
            - Sewa
            - Tenor
            - Tgk
        - File must have a column with the following name:
            - Status
    """)

# Display checkbox for viewing data before and after preprocessing
if st.checkbox("Show Preprocessed Data"):
    # Display preprocessed data
    st.subheader("Preprocessed Data:")
    st.write("Features (X):")
    st.write(x)
    st.write("Target variable (Y):")
    st.write(y)




selected = option_menu(
    menu_title=False,
    options=list(Tabs.keys()),
    icons=["house", "activity", "eye"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Condition to call the app function
if selected in ["Prediction", "Visualisation"]:
    Tabs[selected].app(df, x, y)
else:
    Tabs[selected].app(df, x, y)
