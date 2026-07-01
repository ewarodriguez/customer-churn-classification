import streamlit as st
import pandas as pd
import pickle
import io
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler,RobustScaler

st.title("Customer Churn Prediction App")

# --- File uploader always visible ---
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    df_pre = pd.read_csv(uploaded_file)

    #Preprocessing 
    #OneHotEncoding(pd.get_dummies)
    # df_pre.set_index(['id','CustomerId','Surname','Churn'], inplace=True)
    if "Churn" in df_pre.columns:
        df_pre.set_index(['id', 'CustomerId', 'Surname', 'Churn'], inplace=True)
    else:
        df_pre.set_index(['id', 'CustomerId', 'Surname'], inplace=True)

    df = pd.get_dummies(data = df_pre, columns = list(df_pre.select_dtypes(include=['object'])))

    # Load the fitted RobustScaler instance safely
    BASE_DIR = r"C:\Users\DELL\Downloads\Personal_Projects\Classification_Customer_Churn\models"
    scaler_path = os.path.join(BASE_DIR, 'RobustScaler.pkl')
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # scaler = RobustScaler()
    df_test_scaled_idx = df.index
    df_test_scaled_cols = df.columns
    scaled_values2 = scaler.transform(df)

    # Rebuild the DataFrame while keeping the index
    df_test_scaled = pd.DataFrame(scaled_values2, index=df_test_scaled_idx, columns=df_test_scaled_cols)

    #Feature Scaling
    st.write("Input Preview:", df.head(100))

    st.write("Scaled Input Preview:", df_test_scaled.head(100))

    # --- Try loading model ---
    # try:
    #     model_path = os.path.join("models", "XgBoostClassifier_Model.pkl")  # adjust filename if needed
    #     with open(model_path, "rb") as f:
    #         model = pickle.load(f)

    # 1. Define the directory holding your models
    MODEL_DIR = "models"

    st.subheader("Model Selection & Inference")

    # 2. Safely fetch available models from the folder
    if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
        # Filter to only list actual files (e.g., .pkl or .json files)
        model_files = [
            f
            for f in os.listdir(MODEL_DIR)
            if os.path.isfile(os.path.join(MODEL_DIR, f))
            and "Model" in f
        ]
    else:
        model_files = []

    # 3. Present the dropdown or throw an error if the folder is empty
    if not model_files:
        st.error(f"No model files found in the '{MODEL_DIR}/' directory. Please add a model.")
        st.stop()

    # This generates the dropdown menu in the UI
    selected_model_name = st.selectbox("Choose a model for prediction:", model_files)

    # 4. Construct the full path and load the selected model
    model_path = os.path.join(MODEL_DIR, selected_model_name)

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Could not load the model: {e}")
        st.stop()

    # 5. Only show predictions and download option after clicking the button
    if st.button("Run Prediction", type="primary"):
        try:
            # Generate a list of all columns EXCEPT the ones you want to skip
            features_to_use = [col for col in df_test_scaled.columns if col not in ['id', 'Surname','CustomerId','Churn']]
            
            # Predict using the filtered list view
            preds = model.predict(df_test_scaled[features_to_use])
            # # Combine them efficiently
            # preds = model.predict(df)
            df["Customer Churn Prediction"] = preds

            st.write("Output Preview:", df.head())

            # --- Download button ---
            buffer = io.BytesIO()
            df.to_csv(buffer, index=True)
            buffer.seek(0)

            st.download_button(
                label="Download Customer Churn Predictions.csv",
                data=buffer,
                file_name="Customer Churn Predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not run predictions or generate download: {e}")
    # try:
    #     with open(model_path, "rb") as f:
    #         model = pickle.load(f)


    #     preds = model.predict(df)
    #     df["Customer Churn Prediction"] = preds                
    #     st.write("Output Preview:", df.head())


    #     # --- Download button ---
    #     buffer = io.BytesIO()
    #     df.to_csv(buffer, index=False)
    #     buffer.seek(0)

    #     st.download_button(
    #         label="Download Customer Churn Predictions.csv",
    #         data=buffer,
    #         file_name="Customer Churn Predictions.csv",
    #         mime="text/csv"
    #     )
    # except Exception as e:
    #     st.error(f"Could not load model or run predictions: {e}")
