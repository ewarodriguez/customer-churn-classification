import streamlit as st
import pandas as pd
import pickle
import io
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler

st.set_page_config(
    page_title="Customer Churn Prediction App",
    layout="wide" 
)
st.title("Customer Churn Prediction App 🔮")

# --- Styling Function to Force White Columns, Headers, & Index Fields ---
def style_white_dataframe(df_subset, is_scaled=False):
    """
    Safely shifts indices to regular columns so Streamlit cannot strip their white colors,
    handles string check mark conversions, and applies standard clean layouts.
    """
    display_df = df_subset.reset_index()
    
    if not is_scaled:
        for col in display_df.columns:
            if "Geography" in str(col) or "Gender" in str(col):
                display_df[col] = display_df[col].astype(str).replace({
                    '1': "✔️", '1.0': "✔️", 'True': "✔️",
                    '0': "", '0.0': "", 'False': ""
                })

    styler = display_df.style
    if is_scaled:
        styler = styler.format(precision=2)

    return (
        styler
        .hide(axis="index")
        .set_properties(**{
            'background-color': '#ffffff',
            'color': '#111215',
            'border': '1px solid #dcdde1',
            'text-align': 'center'
        })
        .set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#f1f2f6'),
                    ('color', '#111215'),
                    ('font-weight', 'bold'),
                    ('border', '1px solid #dcdde1'),
                    ('text-align', 'center')
                ]
            }
        ])
    )

# --- 🛠️ Sidebar Configuration Panel ---
st.sidebar.header("⚙️ Model Configuration")

MODEL_DIR = "models"
if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
    model_files = [ f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f)) and "Model" in f ]
else:
    model_files = []

if not model_files:
    st.sidebar.error(f"No model files found in the '{MODEL_DIR}/' directory. Please add a model.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Choose a model for prediction:", model_files)

model_path = os.path.join(MODEL_DIR, selected_model_name)
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.sidebar.error(f"Could not load the model: {e}")
    st.stop()

run_inference = st.sidebar.button("Run Prediction", type="primary", use_container_width=True)


# --- Main File Uploader Container ---
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    df_pre = pd.read_csv(uploaded_file)
    
    # Preprocessing
    if "Churn" in df_pre.columns:
        df_pre.set_index(['id', 'CustomerId', 'Surname', 'Churn'], inplace=True)
    else:
        df_pre.set_index(['id', 'CustomerId', 'Surname'], inplace=True)
        
    df = pd.get_dummies(data=df_pre, columns=list(df_pre.select_dtypes(include=['object'])))

    # Load the fitted RobustScaler instance safely
    BASE_DIR = r"C:\Users\DELL\Downloads\Personal_Projects\Classification_Customer_Churn\models"
    scaler_path = os.path.join(BASE_DIR, 'RobustScaler.pkl')
    try:
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
    except Exception as e:
        st.error(f"Could not load RobustScaler.pkl: {e}")
        st.stop()

    df_test_scaled_idx = df.index
    df_test_scaled_cols = df.columns
    scaled_values2 = scaler.transform(df)
    
    # Rebuild the DataFrame while keeping the index
    df_test_scaled = pd.DataFrame(scaled_values2, index=df_test_scaled_idx, columns=df_test_scaled_cols)

    # --- 📈 Top KPI Dashboard Summary Cards ---
    st.subheader("📊 Dataset Summary")
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric(label="Total Customer Records Loaded", value=f"{len(df):,}")
    with kpi_col2:
        st.metric(label="Historical Columns Extracted", value=f"{len(df.columns)}")

    # --- 🔀 Side-by-Side Data Exploration Previews ---
    st.write("---")
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        st.write("**Input Preview (Original Data)**")
        st.dataframe(style_white_dataframe(df.head(100)), use_container_width=True)

    with preview_col2:
        st.write("**Scaled Input Preview (Normalized Data)**")
        st.dataframe(style_white_dataframe(df_test_scaled.head(100), is_scaled=True), use_container_width=True)

    # --- Prediction Processing & Charts ---
    if run_inference:
        try:
            features_to_use = [col for col in df_test_scaled.columns if col not in ['id', 'Surname','CustomerId','Churn']]
            preds = model.predict(df_test_scaled[features_to_use])
            
            # Save predictions to global session state or df directly
            df["Customer Churn Prediction"] = preds
            st.session_state["predictions_complete"] = True
            st.session_state["df_with_predictions"] = df.copy()
            
            # --- Inference Dashboard Calculations ---
            total_predicted = len(preds)
            churned_count = int(sum(preds == 1))
            retained_count = int(sum(preds == 0))
            churn_rate = (churned_count / total_predicted) * 100

            st.write("---")
            st.subheader("🎯 Inference Analytics")
            
            dashboard_col, chart_col = st.columns(2) 
            
            with dashboard_col:
                st.metric(label="Predicted Churn Risk (Losing)", value=f"{churned_count:,} users")
                st.metric(label="Predicted Retained (Keeping)", value=f"{retained_count:,} users")
                st.metric(label="Overall Risk Percentage", value=f"{churn_rate:.1f}%")
                
            with chart_col:
                chart_df = pd.DataFrame({
                    "Retained (0)": [retained_count],
                    "Churn Risk (1)": [churned_count]
                })
                st.bar_chart(chart_df, color=["#2ecc71", "#e74c3c"])

            # --- Model Performance Report Table ---
            st.subheader("📋 Model Run Specifications")
            report_data = {
                "Metric Parameters": ["Selected Pipeline Model", "Batch Constraint Limit", "Input Vector Count", "Features Extracted"],
                "Details Summary": [str(selected_model_name), f"{total_predicted:,} Records", f"{len(features_to_use)} Features", ", ".join(features_to_use[:5]) + "..."]
            }
            st.dataframe(style_white_dataframe(pd.DataFrame(report_data).set_index("Metric Parameters")), use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction logic: {e}")

    # --- 🔍 Interactive Spreadsheet Explorer ---
    if "df_with_predictions" in st.session_state:
        st.write("---")
        st.subheader("📋 Interactive Spreadsheet Explorer")
        st.caption("💡 **Tip:** Click any column header to sort. Hover over the grid or select cells to access row search tools natively.")
        
        # Load prediction data snapshot
        display_output_df = st.session_state["df_with_predictions"].copy()
        
        # Truncate row limits cleanly
        display_output_df = display_output_df.head(100)

        # Prediction column conditional cell highlights
        def highlight_predictions(val):
            if val == 0:
                return 'background-color: #2ecc71; color: #ffffff; font-weight: bold; border: 1px solid #dcdde1;'
            elif val == 1:
                return 'background-color: #e74c3c; color: #ffffff; font-weight: bold; border: 1px solid #dcdde1;'
            return ''

        if len(display_output_df) > 0:
            styled_output_df = (
                style_white_dataframe(display_output_df)
                .map(highlight_predictions, subset=['Customer Churn Prediction'])
            )
            
            # Option 2 Implementation
            st.data_editor(
                styled_output_df, 
                use_container_width=True, 
                disabled=True  # Keeps fields read-only so users can't overwrite backend predictions
            )
        else:
            st.warning("No data found to display.")

# st.set_page_config(page_title="Customer Churn Prediction App")

# st.title("Customer Churn Prediction App")

# # --- File uploader always visible ---
# uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df_pre = pd.read_csv(uploaded_file)

#     #Preprocessing 
#     #OneHotEncoding(pd.get_dummies)
#     # df_pre.set_index(['id','CustomerId','Surname','Churn'], inplace=True)
#     if "Churn" in df_pre.columns:
#         df_pre.set_index(['id', 'CustomerId', 'Surname', 'Churn'], inplace=True)
#     else:
#         df_pre.set_index(['id', 'CustomerId', 'Surname'], inplace=True)

#     df = pd.get_dummies(data = df_pre, columns = list(df_pre.select_dtypes(include=['object'])))

#     # Load the fitted RobustScaler instance safely
#     BASE_DIR = r"C:\Users\DELL\Downloads\Personal_Projects\Classification_Customer_Churn\models"
#     scaler_path = os.path.join(BASE_DIR, 'RobustScaler.pkl')
#     with open(scaler_path, 'rb') as file:
#         scaler = pickle.load(file)

#     # scaler = RobustScaler()
#     df_test_scaled_idx = df.index
#     df_test_scaled_cols = df.columns
#     scaled_values2 = scaler.transform(df)

#     # Rebuild the DataFrame while keeping the index
#     df_test_scaled = pd.DataFrame(scaled_values2, index=df_test_scaled_idx, columns=df_test_scaled_cols)

#     #Feature Scaling
#     st.write("Input Preview:", df.head(100))

#     st.write("Scaled Input Preview:", df_test_scaled.head(100))

#     # --- Try loading model ---
#     # try:
#     #     model_path = os.path.join("models", "XgBoostClassifier_Model.pkl")  # adjust filename if needed
#     #     with open(model_path, "rb") as f:
#     #         model = pickle.load(f)

#     # 1. Define the directory holding your models
#     MODEL_DIR = "models"

#     st.subheader("Model Selection & Inference")

#     # 2. Safely fetch available models from the folder
#     if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
#         # Filter to only list actual files (e.g., .pkl or .json files)
#         model_files = [
#             f
#             for f in os.listdir(MODEL_DIR)
#             if os.path.isfile(os.path.join(MODEL_DIR, f))
#             and "Model" in f
#         ]
#     else:
#         model_files = []

#     # 3. Present the dropdown or throw an error if the folder is empty
#     if not model_files:
#         st.error(f"No model files found in the '{MODEL_DIR}/' directory. Please add a model.")
#         st.stop()

#     # This generates the dropdown menu in the UI
#     selected_model_name = st.selectbox("Choose a model for prediction:", model_files)

#     # 4. Construct the full path and load the selected model
#     model_path = os.path.join(MODEL_DIR, selected_model_name)

#     try:
#         with open(model_path, "rb") as f:
#             model = pickle.load(f)
#     except Exception as e:
#         st.error(f"Could not load the model: {e}")
#         st.stop()

#     # 5. Only show predictions and download option after clicking the button
#     if st.button("Run Prediction", type="primary"):
#         try:
#             # Generate a list of all columns EXCEPT the ones you want to skip
#             features_to_use = [col for col in df_test_scaled.columns if col not in ['id', 'Surname','CustomerId','Churn']]
            
#             # Predict using the filtered list view
#             preds = model.predict(df_test_scaled[features_to_use])
#             # # Combine them efficiently
#             # preds = model.predict(df)
#             df["Customer Churn Prediction"] = preds

#             st.write("Output Preview:", df.head())

#             # --- Download button ---
#             buffer = io.BytesIO()
#             df.to_csv(buffer, index=True)
#             buffer.seek(0)

#             st.download_button(
#                 label="Download Customer Churn Predictions.csv",
#                 data=buffer,
#                 file_name="Customer Churn Predictions.csv",
#                 mime="text/csv",
#             )
#         except Exception as e:
#             st.error(f"Could not run predictions or generate download: {e}")
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
