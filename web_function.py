import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.preprocessing import LabelEncoder

@st.cache_data()
def load_data(uploaded_file=None):
    """
    Load data from a CSV file or use a default file.

    Parameters:
    - uploaded_file (str): Path to the uploaded CSV file.

    Returns:
    - df (pd.DataFrame): Loaded DataFrame.
    - x (pd.DataFrame): Features.
    - y (pd.Series): Target variable.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Use a default file (DT_clean.csv)
        # df = pd.read_csv('DT_clean.csv')
        df = pd.DataFrame()

    # Select features (x) and target variable (y)
    x_columns = ["Kredit", "Sewa", "Tenor", "Tgk"]
    y_column = "Status"

    x = df[x_columns] if not df.empty else None

    # Transform target variable (y) to binary (1 or 0)
    le = LabelEncoder()
    y_transformed = le.fit_transform(df[y_column])
    y = pd.Series(y_transformed, name=y_column) if not df.empty else None

    return df, x, y

@st.cache_data()
def train_model(x,y):
    model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'        
            )
    model.fit(x,y)
    
    score = model.score(x,y)

    return model, score

def predict(x,y,features):
    model, score = train_model(x,y)

    prediction = model.predict(np.array(features).reshape(1,-1))

    return prediction, score

