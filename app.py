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


# --- 🔄 Initialize State Trackers ---
if 'prediction_run' not in st.session_state:
    st.session_state['prediction_run'] = False

if 'slider_version' not in st.session_state:
    st.session_state['slider_version'] = 0

def force_slider_reset():
    st.session_state['slider_version'] += 1

# Callback function to safely reset state variables without forcing a file reload
def handle_clear_prediction():
    st.session_state['prediction_run'] = False

# --- 🚀 Cached Resource Loaders to Boost App Performance ---
@st.cache_resource(show_spinner="Loading selected model...")
def load_ml_model(model_folder, model_name):
    path = os.path.join(model_folder, model_name)
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading scaler transformations...")
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        return pickle.load(file)

# --- Styling Function to Force White Columns, Headers, & Index Fields ---
def style_white_dataframe(df_subset, is_scaled=False):
    """
    Optimized styling engine leveraging fast vectorized assignments 
    instead of slow index/column iteration loops.
    """
    display_df = df_subset.reset_index()
    
    if not is_scaled:
        # Find columns containing Geography or Gender
        target_cols = [col for col in display_df.columns if "Geography" in str(col) or "Gender" in str(col)]
        if target_cols:
            # High-speed vectorized string mapping
            display_df[target_cols] = display_df[target_cols].astype(str).replace({
                '1': "✔️", '1.0': "✔️", 'True': "✔️",'TRUE': "✔️", 
                '0': "", '0.0': "", 'False': "", 'FALSE': ""
            })

    styler = display_df.style
    # if is_scaled:
    #     styler = styler.format(precision=2)
    # 2. NUMERICAL TRANSFORMATION (Only runs on actual numbers)
    if is_scaled:
        # Isolate ONLY numeric columns to apply decimal rounding
        numeric_cols = display_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            styler = styler.format(precision=2, subset=numeric_cols)
                
        # Target non-numeric columns to map True/False to 1/0
        non_numeric_cols = display_df.select_dtypes(exclude=['number']).columns.tolist()
        for col in non_numeric_cols:
            display_df[col] = display_df[col].astype(str).replace({
                'True': '1', 'TRUE': '1', 'true': '1',
                'False': '0', 'FALSE': '0', 'false': '0'
            })

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
    model_files = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f)) and "Model" in f]
else:
    model_files = []

if not model_files:
    st.sidebar.error(f"No model files found in the '{MODEL_DIR}/' directory. Please add a model.")
    st.stop()

# Helper function to format the file name beautifully in the UI
def format_model_name(file_name):
    # Removes extension and replaces symbols with spaces (e.g., "Random_Forest_Model.pkl" -> "Random Forest Model")
    name_without_ext = os.path.splitext(file_name)[0]
    return name_without_ext.replace("_", " ").replace("-", " ")

# The selectbox now uses format_func to change the visual label while keeping the actual file name as the value
selected_model_name = st.sidebar.selectbox(
    "Choose a model for prediction:", 
    model_files,
    format_func=format_model_name
)

# Dynamic fallback path handling for system flexibility
APP_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
scaler_path = os.path.join(APP_DIR, 'models', 'RobustScaler.pkl')

try:
    model = load_ml_model(MODEL_DIR, selected_model_name)
except Exception as e:
    st.sidebar.error(f"Could not load the model: {e}")
    st.stop()


# --- 🔘 Interactive Prediction Run Controls ---
run_inference = st.sidebar.button("Run Prediction", type="primary", use_container_width=True)
if run_inference:
    st.session_state['prediction_run'] = True

# Display Clear Prediction button if a prediction has been executed
if st.session_state['prediction_run']:
    st.sidebar.button(
        "Clear Prediction", 
        type="secondary", 
        use_container_width=True, 
        on_click=handle_clear_prediction  # State updates instantly without breaking file upload context
    )

# --- Main File Uploader Container ---
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    df_pre = pd.read_csv(uploaded_file)
    
    # Preprocessing Multi-index Assignments
    if "Churn" in df_pre.columns:
        df_pre.set_index(['id', 'CustomerId', 'Surname', 'Churn'], inplace=True)
    else:
        df_pre.set_index(['id', 'CustomerId', 'Surname'], inplace=True)
        
    df = pd.get_dummies(data=df_pre, columns=list(df_pre.select_dtypes(include=['object'])))

    # Fetching scaled parameters dynamically
    try:
        scaler = load_scaler(scaler_path)
    except Exception as e:
        st.error(f"Could not load RobustScaler.pkl from default directory: {e}")
        st.stop()


    # Allowed values for a binary column
    binary_set = {0, 1, True, False, '0', '1', 'TRUE', 'FALSE', 'True', 'False'}

    # 1. DO NOT SCALE: Columns that are strictly binary
    no_scale_cols = [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset(binary_set)
    ]

    # 2. SCALE: Columns that are continuous/numeric (not binary)
    scale_cols = [
        col for col in df.columns
        if not set(df[col].dropna().unique()).issubset(binary_set)
    ]


    X_test_scaled_part = pd.DataFrame(
        scaler.transform(df[scale_cols]), 
        index=df.index, 
        columns=scale_cols
    )


    # 3. Concatenate the scaled parts with the unscaled parts
    df_test_scaled = pd.concat([X_test_scaled_part, df[no_scale_cols]], axis=1)

    # 4. Optional: Reorder columns to match the original layout
    df_test_scaled = df_test_scaled.reindex(columns=df.columns)



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




    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Model Settings")
        
        # 3. RUN MATH BEHIND THE SCENES: Automate Target Rate Calibration using your fixed training percentage
        try:
            features_to_use = [col for col in df_test_scaled.columns if col not in ['id', 'Surname','CustomerId','Churn']]
            live_probs = model.predict_proba(df_test_scaled[features_to_use])[:, 1]
            
            # Locked to your exact training baseline (34,921 / (130,113 + 34,921)) = ~21.16%
            target_churn_percentage = 0.2116 
            
            # Find the exact risk boundary that isolates the top 21.16% highest-risk rows
            percentile_needed = (1 - target_churn_percentage) * 100
            import numpy as np
            calculated_threshold = float(np.percentile(live_probs, percentile_needed))
        except Exception:
            calculated_threshold = 0.60

        # Preserve the variable name 'suggested_threshold' globally for downstream metrics
        suggested_threshold = calculated_threshold
        slider_default_value = max(0.10, min(0.90, round(suggested_threshold, 2)))

        # 4. Render the slider with a dynamic versioned key
        # Every time the button is clicked, the key changes (e.g., 'slider_v0' becomes 'slider_v1')
        custom_threshold = st.slider(
            label="Churn Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=slider_default_value,  
            key=f"slider_v{st.session_state['slider_version']}", 
            step=0.01,
            help=f"Adjust to manually shift metrics. The calculated value to match historical proportions is {suggested_threshold:.2f}."
        )
        
        # 5. THE RESET BUTTON: Calls the version incrementing function
        st.button("Reset to default", on_click=force_slider_reset)
        
        # 6. Simple visual status indicator at the bottom
        if abs(custom_threshold - suggested_threshold) < 0.015:
            st.caption("✨ Slider matches mathematically recommended threshold.")
        else:
            st.caption("🔒 Manually overriding threshold recommendation.")


    # --- Prediction Processing & Charts ---
    if run_inference:
        try:
            # Safely generate probabilities if they weren't saved globally earlier
            if 'live_probs' not in locals() or live_probs is None:
                features_to_use = [col for col in df_test_scaled.columns if col not in ['id', 'Surname','CustomerId','Churn']]
                live_probs = model.predict_proba(df_test_scaled[features_to_use])[:, 1]

            # Process predictions using the slider's active threshold value
            preds = (live_probs >= custom_threshold).astype('int')
            
            df["Customer Churn Prediction"] = preds
            st.session_state["predictions_complete"] = True
            st.session_state["df_with_predictions"] = df.copy()

            # --- Inference Dashboard Calculations ---
            total_predicted = len(preds)
            churned_count = int(sum(preds == 1))
            retained_count = int(sum(preds == 0))
            churn_rate = (churned_count / total_predicted) * 100

            # --- VISUAL SUGGESTION METRICS ---
            st.markdown("### 📊 Dashboard Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Current Selected Threshold", 
                    value=f"{custom_threshold:.2f}",
                    delta=f"Dynamic Rate Match: {suggested_threshold:.2f}",
                    delta_color="normal" if custom_threshold == round(suggested_threshold, 2) else "off"
                )
            with col2:
                st.metric(label="Total Flagged Churners", value=f"{churned_count:,}")
            with col3:
                st.metric(label="Current Churn Rate", value=f"{churn_rate:.2f}%")
                
            # Contextual alert banners for your dashboard operators
            target_check = round(suggested_threshold, 2)
            if custom_threshold < target_check:
                st.info(f"⚠️ Threshold is lower than the calculated baseline ({suggested_threshold:.2f}). Expect higher false positives.")
            elif custom_threshold > target_check:
                st.info(f"ℹ️ Threshold is higher than the calculated baseline ({suggested_threshold:.2f}). Filtering risk more conservatively.")
            else:
                st.success("🎯 Your slider perfectly matches the dynamic historical churn rate proportion!")

        # except Exception as e:
        #     st.error(f"Prediction Error: {e}")

            st.write("---")
            st.subheader("🎯 Customer Retention Overview:")
            
            dashboard_col, chart_col = st.columns(2) 
            
            with dashboard_col:
                st.metric(label="Predicted No. of Users to Churn (At Risk)", value=f"{churned_count:,} users")
                st.metric(label="Predicted No. of Users Likely to Stay (Safe)", value=f"{retained_count:,} users")
                st.metric(label="Overall Risk Percentage", value=f"{churn_rate:.1f}%")
                
            with chart_col:
                chart_df = pd.DataFrame({
                    "Likely to Churn (0)": [retained_count],
                    "Churn/At Risk (1)": [churned_count]
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
    # if "df_with_predictions" in st.session_state:
    #     st.write("---")
    #     st.subheader("📋 Interactive Spreadsheet Explorer")
    #     st.caption("💡 **Tip:** Click any column header to sort. Hover over the grid or select cells to access row search tools natively.")

        if "df_with_predictions" in st.session_state and st.session_state['prediction_run']:
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
                style_white_dataframe(display_output_df,is_scaled=True)
                .map(highlight_predictions, subset=['Customer Churn Prediction'])
            )
            
            # Interactive read-only viewport render 
            st.dataframe(
                styled_output_df, 
                use_container_width=True 
                # disabled=True
            )
        else:
            st.warning("No data found to display.")

# --- FIXED: ADDED DOWNLOAD BUTTON HERE (BELOW THE EXPLORER) ---
        # --- Download button below the explorer ---
        if len(st.session_state["df_with_predictions"]) > 0:
            st.write("") # Tiny spacer for layout breathing room
            
            # 1. Create a copy of the full dataset to safely modify for download
            download_df = st.session_state["df_with_predictions"].copy()
            
            # 2. Find all non-numeric columns (strings, objects, booleans)
            non_numeric_cols = download_df.select_dtypes(exclude=['number']).columns.tolist()
            
            # 3. Permanently map any True/False variations to 1/0 for the CSV structure
            for col in non_numeric_cols:
                download_df[col] = download_df[col].astype(str).replace({
                    'True': '1', 'TRUE': '1', 'true': '1', '1.0': '1',
                    'False': '0', 'FALSE': '0', 'false': '0', '0.0': '0'
                })

            # 4. Convert the mapped dataframe to CSV string data
            csv_data = download_df.to_csv(index=True, index_label="Index").encode('utf-8')
            
            # Fetch and format the current date (YYYY-MM-DD)
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            st.download_button(
                label="📥 Download Full Predictions (CSV)",
                data=csv_data,
                file_name=f"customer_churn_predictions_{current_date}.csv",
                mime="text/csv",
                key="download-predictions-csv",
                on_click="ignore"  # Keeps the action purely frontend to bypass script executions entirely
            )













