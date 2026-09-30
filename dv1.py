import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medicine Sales Prediction Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("💊 Medicine Sales Prediction & Inventory Analytics Dashboard")

st.markdown(
    "An interactive Machine Learning platform for forecasting "
    "pharmaceutical demand and optimizing inventory."
)

# ============================================================
# SIDEBAR - DATA UPLOAD
# ============================================================

st.sidebar.header("📁 Data Ingestion Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Pharmacy Sales CSV",
    type=["csv"]
)

# ============================================================
# LOAD DATA FUNCTION
# ============================================================

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        # Remove invalid dates
        df = df.dropna(subset=["Date"])

        # Create time-based features
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek

    return df


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # Load dataset
    df = load_data(uploaded_file)

    # Make a working copy
    df_filtered = df.copy()

    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================

    st.sidebar.header("🔍 Dynamic Data Filters")

    # --------------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------------

    if "Category" in df.columns:

        categories = sorted(
            df["Category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_categories = st.sidebar.multiselect(
            "Filter by Drug Category",
            options=categories,
            default=categories
        )

        df_filtered = df_filtered[
            df_filtered["Category"].astype(str).isin(
                selected_categories
            )
        ]

    # --------------------------------------------------------
    # DATE RANGE FILTER
    # --------------------------------------------------------

    if "Date" in df.columns and not df_filtered.empty:

        min_date = df_filtered["Date"].min().date()
        max_date = df_filtered["Date"].max().date()

        date_range = st.sidebar.date_input(
            "Select Sales Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:

            df_filtered = df_filtered[
                (df_filtered["Date"].dt.date >= date_range[0]) &
                (df_filtered["Date"].dt.date <= date_range[1])
            ]

    # ========================================================
    # MAIN NAVIGATION TABS
    # ========================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Dataset Overview",
            "📊 Exploratory Data Analysis",
            "🤖 Machine Learning Prediction",
            "📈 Model Performance Metrics",
            "📥 Download Predictions"
        ]
    )

    # ========================================================
    # TAB 1 - DATASET OVERVIEW
    # ========================================================

    with tab1:

        st.subheader("Dataset Preview & Structure")

        # ----------------------------------------------------
        # DATASET METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Records",
            f"{len(df_filtered):,}"
        )

        if "Category" in df_filtered.columns:
            col2.metric(
                "Total Drug Categories",
                f"{df_filtered['Category'].nunique():,}"
            )
        else:
            col2.metric(
                "Total Drug Categories",
                "N/A"
            )

        if "Quantity" in df_filtered.columns:
            col3.metric(
                "Total Quantity Sold",
                f"{df_filtered['Quantity'].sum():,.0f}"
            )
        else:
            col3.metric(
                "Total Quantity Sold",
                "N/A"
            )

        if "Sales" in df_filtered.columns:
            col4.metric(
                "Total Revenue",
                f"${df_filtered['Sales'].sum():,.2f}"
            )
        else:
            col4.metric(
                "Total Revenue",
                "N/A"
            )

        # ----------------------------------------------------
        # DATA TABLE
        # ----------------------------------------------------

        st.dataframe(
            df_filtered.head(100),
            use_container_width=True
        )

        # ----------------------------------------------------
        # STATISTICAL SUMMARY
        # ----------------------------------------------------

        with st.expander("📊 Show Dataset Statistical Summary"):

            st.write(
                df_filtered.describe(
                    include="all"
                )
            )

    # ========================================================
    # TAB 2 - EXPLORATORY DATA ANALYSIS
    # ========================================================

    with tab2:

        st.subheader("Pharmaceutical Visual Analytics")

        col_left, col_right = st.columns(2)

        # ----------------------------------------------------
        # HISTORICAL SALES TREND
        # ----------------------------------------------------

        with col_left:

            if (
                "Date" in df_filtered.columns
                and "Sales" in df_filtered.columns
                and not df_filtered.empty
            ):

                st.markdown(
                    "#### Historical Sales Trend Over Time"
                )

                daily_sales = (
                    df_filtered
                    .groupby("Date")["Sales"]
                    .sum()
                    .reset_index()
                )

                fig_line = px.line(
                    daily_sales,
                    x="Date",
                    y="Sales",
                    title="Daily Revenue Trend",
                    color_discrete_sequence=["#1f77b4"]
                )

                st.plotly_chart(
                    fig_line,
                    use_container_width=True
                )

            else:

                st.info(
                    "Date and Sales columns are required "
                    "for the sales trend."
                )

        # ----------------------------------------------------
        # SALES BY CATEGORY
        # ----------------------------------------------------

        with col_right:

            if (
                "Category" in df_filtered.columns
                and "Sales" in df_filtered.columns
                and not df_filtered.empty
            ):

                st.markdown(
                    "#### Sales Revenue by Drug Category"
                )

                cat_sales = (
                    df_filtered
                    .groupby("Category")["Sales"]
                    .sum()
                    .reset_index()
                )

                fig_bar = px.bar(
                    cat_sales,
                    x="Category",
                    y="Sales",
                    color="Category",
                    title="Category Revenue Distribution"
                )

                st.plotly_chart(
                    fig_bar,
                    use_container_width=True
                )

            else:

                st.info(
                    "Category and Sales columns are required "
                    "for category analysis."
                )

        # ----------------------------------------------------
        # CORRELATION MATRIX
        # ----------------------------------------------------

        st.markdown("#### Feature Correlation Matrix")

        numeric_cols = (
            df_filtered
            .select_dtypes(include=np.number)
            .columns
        )

        if len(numeric_cols) >= 2:

            corr = df_filtered[numeric_cols].corr()

            fig_corr, ax = plt.subplots(
                figsize=(8, 4)
            )

            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )

            st.pyplot(fig_corr)

        else:

            st.info(
                "At least two numeric columns are required "
                "for the correlation matrix."
            )

    # ========================================================
    # TAB 3 - MACHINE LEARNING PREDICTION
    # ========================================================

    with tab3:

        st.subheader("Sales Demand Prediction Engine")

        st.write(
            "Train Machine Learning models to predict future "
            "medicine sales based on temporal features and pricing."
        )

        # ----------------------------------------------------
        # MODEL CONFIGURATION
        # ----------------------------------------------------

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:

            model_choice = st.selectbox(
                "Select ML Algorithm",
                [
                    "Random Forest Regressor",
                    "Linear Regression"
                ]
            )

        with col_m2:

            test_size = st.slider(
                "Test Set Split Ratio (%)",
                min_value=10,
                max_value=40,
                value=20
            ) / 100.0

        with col_m3:

            target_options = []

            if "Sales" in df_filtered.columns:
                target_options.append("Sales")

            if "Quantity" in df_filtered.columns:
                target_options.append("Quantity")

            if len(target_options) == 0:

                st.error(
                    "The dataset must contain either "
                    "'Sales' or 'Quantity' column."
                )

                target_col = None

            else:

                target_col = st.selectbox(
                    "Select Prediction Target Column",
                    target_options
                )

        # ----------------------------------------------------
        # FEATURE SELECTION
        # ----------------------------------------------------

        st.markdown("### 🎯 Feature Selection")

        available_features = [
            c
            for c in [
                "Year",
                "Month",
                "Day",
                "DayOfWeek",
                "Price",
                "Stock"
            ]
            if c in df_filtered.columns
        ]

        if available_features:

            selected_features = st.multiselect(
                "Select Model Predictor Features",
                available_features,
                default=available_features
            )

        else:

            selected_features = []

            st.warning(
                "No supported numeric predictor features "
                "were found in the dataset."
            )

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        if st.button(
            "🚀 Train Model & Generate Predictions"
        ):

            if target_col is None:

                st.error(
                    "Please select a valid prediction target."
                )

            elif len(selected_features) == 0:

                st.error(
                    "Please select at least one feature column "
                    "for training."
                )

            elif df_filtered.empty:

                st.error(
                    "No data available after applying the filters."
                )

            else:

                # --------------------------------------------
                # PREPARE DATA
                # --------------------------------------------

                model_data = df_filtered[
                    selected_features + [target_col]
                ].copy()

                # Convert features and target to numeric
                for col in selected_features:
                    model_data[col] = pd.to_numeric(
                        model_data[col],
                        errors="coerce"
                    )

                model_data[target_col] = pd.to_numeric(
                    model_data[target_col],
                    errors="coerce"
                )

                # Remove missing values
                model_data = model_data.dropna()

                if len(model_data) < 5:

                    st.error(
                        "Not enough valid records available "
                        "for model training."
                    )

                else:

                    X = model_data[selected_features]
                    y = model_data[target_col]

                    # ----------------------------------------
                    # TRAIN / TEST SPLIT
                    # ----------------------------------------

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=test_size,
                            random_state=42
                        )
                    )

                    # ----------------------------------------
                    # SELECT MODEL
                    # ----------------------------------------

                    if model_choice == "Random Forest Regressor":

                        model = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                        )

                    else:

                        model = LinearRegression()

                    # ----------------------------------------
                    # TRAIN MODEL
                    # ----------------------------------------

                    model.fit(
                        X_train,
                        y_train
                    )

                    # ----------------------------------------
                    # TEST PREDICTIONS
                    # ----------------------------------------

                    y_pred = model.predict(X_test)

                    # ----------------------------------------
                    # STORE RESULTS
                    # ----------------------------------------

                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    st.session_state["model_trained"] = True
                    st.session_state["model"] = model
                    st.session_state["model_choice"] = model_choice
                    st.session_state["target_col"] = target_col
                    st.session_state["selected_features"] = (
                        selected_features
                    )

                    # ----------------------------------------
                    # FULL DATA PREDICTIONS
                    # ----------------------------------------

                    prediction_df = df_filtered.copy()

                    full_model_data = prediction_df[
                        selected_features
                    ].copy()

                    for col in selected_features:
                        full_model_data[col] = pd.to_numeric(
                            full_model_data[col],
                            errors="coerce"
                        )

                    # Handle missing feature values
                    full_model_data = full_model_data.fillna(
                        full_model_data.median()
                    )

                    full_predictions = model.predict(
                        full_model_data
                    )

                    prediction_df[
                        "Predicted_" + target_col
                    ] = full_predictions

                    st.session_state[
                        "predicted_df"
                    ] = prediction_df

                    st.success(
                        f"Model successfully trained using "
                        f"{model_choice}!"
                    )

                    st.info(
                        f"Target column: **{target_col}**"
                    )

                    # ----------------------------------------
                    # SAMPLE PREDICTIONS
                    # ----------------------------------------

                    st.markdown(
                        "### 🔮 Sample Predictions"
                    )

                    display_cols = []

                    if "Date" in prediction_df.columns:
                        display_cols.append("Date")

                    if "Category" in prediction_df.columns:
                        display_cols.append("Category")

                    display_cols.append(target_col)
                    display_cols.append(
                        "Predicted_" + target_col
                    )

                    st.dataframe(
                        prediction_df[display_cols].head(20),
                        use_container_width=True
                    )


    # ========================================================
    # TAB 4 - MODEL PERFORMANCE METRICS
    # ========================================================

    with tab4:

        st.subheader("Model Validation & Error Analysis")

        if st.session_state.get(
            "model_trained",
            False
        ):

            y_test = st.session_state["y_test"]
            y_pred = st.session_state["y_pred"]

            target_col = st.session_state.get(
                "target_col",
                "Sales"
            )

            # ------------------------------------------------
            # CALCULATE METRICS
            # ------------------------------------------------

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )

            r2 = r2_score(
                y_test,
                y_pred
            )

            # ------------------------------------------------
            # METRIC DISPLAY
            # ------------------------------------------------

            col_eval1, col_eval2, col_eval3 = st.columns(3)

            col_eval1.metric(
                "Mean Absolute Error (MAE)",
                f"{mae:.2f}"
            )

            col_eval2.metric(
                "Root Mean Squared Error (RMSE)",
                f"{rmse:.2f}"
            )

            col_eval3.metric(
                "R² Score",
                f"{r2 * 100:.2f}%"
            )

            # ------------------------------------------------
            # ACTUAL VS PREDICTED SCATTER
            # ------------------------------------------------

            st.markdown(
                f"### Actual vs Predicted {target_col}"
            )

            eval_df = pd.DataFrame(
                {
                    f"Actual {target_col}": y_test,
                    f"Predicted {target_col}": y_pred
                }
            )

            fig_eval = px.scatter(
                eval_df,
                x=f"Actual {target_col}",
                y=f"Predicted {target_col}",
                title=f"Actual vs. Predicted {target_col}",
                trendline="ols"
            )

            st.plotly_chart(
                fig_eval,
                use_container_width=True
            )

            # ------------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------------

            st.markdown("### 🤖 Model Information")

            st.write(
                f"**Algorithm:** "
                f"{st.session_state.get('model_choice', 'N/A')}"
            )

            st.write(
                f"**Prediction Target:** "
                f"{target_col}"
            )

            st.write(
                "**Features Used:** "
                + ", ".join(
                    st.session_state.get(
                        "selected_features",
                        []
                    )
                )
            )

        else:

            st.info(
                "Please train a Machine Learning model "
                "in the 'Machine Learning Prediction' tab first."
            )


    # ========================================================
    # TAB 5 - DOWNLOAD PREDICTIONS
    # ========================================================

    with tab5:

        st.subheader("Export Forecasted Sales Data")

        if (
            st.session_state.get(
                "predicted_df"
            ) is not None
        ):

            export_df = st.session_state[
                "predicted_df"
            ]

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            st.dataframe(
                export_df.head(50),
                use_container_width=True
            )

            # ------------------------------------------------
            # CONVERT TO CSV
            # ------------------------------------------------

            csv_buffer = export_df.to_csv(
                index=False
            ).encode("utf-8")

            # ------------------------------------------------
            # DOWNLOAD BUTTON
            # ------------------------------------------------

            st.download_button(
                label="📥 Download Full Predictions Report as CSV",
                data=csv_buffer,
                file_name="medicine_sales_predictions.csv",
                mime="text/csv"
            )

            st.success(
                "Prediction report is ready for download."
            )

        else:

            st.info(
                "No prediction data available. "
                "Please generate predictions in "
                "Tab 3 first."
            )

# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.warning(
        "👉 Please upload a valid pharmacy sales CSV dataset "
        "from the sidebar to activate the dashboard."
    )

    st.markdown(
        """
        ### 📌 Expected Dataset Columns

        Your CSV can contain columns such as:

        - `Date`
        - `Category`
        - `Sales`
        - `Quantity`
        - `Price`
        - `Stock`

        The dashboard automatically creates:

        - `Year`
        - `Month`
        - `Day`
        - `DayOfWeek`

        from the `Date` column.
        """
    )