import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")

# 2. Sidebar: Dataset Ingestion
st.sidebar.header("Dataset Ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # --- Secure file validation ---
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or df.shape[1] == 0:
            st.error("The uploaded file is empty or has no columns. Please upload a valid CSV.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read this file as a valid CSV. Error: {e}")
        st.stop()

    # 3. Dataset Overview
    st.subheader("Dataset Overview")

    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Shape (rows, columns):**")
        st.write(df.shape)

        st.write("**Column Data Types:**")
        st.write(df.dtypes.astype(str))

    with col2:
        st.write("**Missing Values per Column:**")
        missing = pd.DataFrame({
            "Missing Count": df.isnull().sum(),
            "Missing Percentage (%)": (df.isnull().mean() * 100).round(2)
        })
        st.dataframe(missing)

    # Basic statistics for numerical columns (mean, median, min, max, etc.)
    st.write("**Basic Numerical Statistics:**")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        stats = numeric_df.describe().T
        stats["median"] = numeric_df.median()
        stats = stats[["mean", "median", "min", "max", "std", "25%", "50%", "75%", "count"]]
        st.dataframe(stats)
    else:
        st.info("No numerical columns found in this dataset.")

    # 4. Attribute Selection
    st.sidebar.header("Attribute Selection")

    column = st.sidebar.selectbox(
        "Choose an attribute",
        df.columns
    )

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    st.write(f"**Selected Attribute:** {column}")
    st.write(f"**Column Type:** {column_type}")

    # 5. Visualization Module
    st.subheader("Visualization")

    if column_type == "Numerical":
        fig, ax = plt.subplots()
        sns.histplot(
            data=df,
            x=column,
            kde=True,
            ax=ax
        )
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:
        counts = df[column].value_counts()
        percentages = (counts / counts.sum() * 100).round(1)

        fig, ax = plt.subplots()
        bars = ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette("viridis", len(counts)))

        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        # Percentage labels above each bar
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")