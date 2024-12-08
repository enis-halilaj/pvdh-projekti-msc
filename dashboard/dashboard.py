import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import seaborn as sns
from io import BytesIO
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def download_plot(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return st.download_button(
        label="Download visualization",
        data=buf,
        file_name=filename,
        mime="image/png",
    )

@st.cache_data
def load_data(raw_path, preprocessed_path):
    raw = pd.read_csv(raw_path)
    preprocessed = pd.read_csv(preprocessed_path)
    return raw, preprocessed

raw_file_path = "dataset/StudentPerformanceFactors.csv"
preprocessed_file_path = "dataset/StudentPerformanceFactors_new.csv"

raw_data, preprocessed_data = load_data(raw_file_path, preprocessed_file_path)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:", 
    [
        "Intro",
        "Preprocessed Data Quality",
        "Outliers", 
        "Frequency Distribution", 
        "Data Skewness", 
        "Correlation Analysis", 
        "3D Surface Plot", 
        "Comparison",
        "Class Imbalance"
    ]
)

if page == "Intro":
    st.markdown(
        """
        <div align="center">

        ### UNIVERSITY OF PRISHTINA  
        ### FACULTY OF ELECTRICAL AND COMPUTER ENGINEERING  

        <img src="https://github.com/enis-halilaj/pvdh-projekti-msc/raw/main/readme_files/up_logo.png" alt="UP Logo" style="width: 100px;"/>

        ### Subject: Data Preparation and Visualization  
        ### Project: Student Performance Factors

        </div>

        <div align="left"><h3>Mentor: Prof. Dr. Mërgim Hoti</h3></div>
        <div align="left"><h3>Contributors: Lirim Islami, Arbnor Puka, Enis Halilaj</h3></div>

        # Student Performance Factors

        This project is part of the course **Data Preparation and Visualization** under the Master's program IKS 2024/25. 
        Our goal is to analyze and visualize factors influencing students' academic performance using data processing and visualization techniques.

        For this project, we used the following dataset: **[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)**

        ## Instructions to Run the Project  
        To execute the project on your device, follow these steps:  

        ### 1. Requirements  
        Ensure you have **Python 3.6** or a newer version installed. You can download it from [python.org](https://www.python.org/downloads/).  

        ### 2. Clone the Project  
        Clone the project using the following command:  
        ```bash
        git clone https://github.com/enis-halilaj/pvdh-projekti-msc
        ```

        ### 3. Install Packages  
        After navigating to the project directory, install the required packages using the following command:  
        ```bash
        pip3 install -r requirements.txt
        ```

        <h1>Student Performance Factors</h1>

        <p>The purpose of preprocessing the dataset <b>Student Performance Factors</b> is to structure and clean the data, enabling deep analysis of factors influencing academic performance. This process ensures that the data is clean, organized, and ready to reveal key relationships affecting students' success. Through this processed data analysis, effective educational strategies can be identified and implemented to improve students' achievements and elevate the quality of educational processes.</p>
        """,
        unsafe_allow_html=True,
    )

elif page == "Preprocessed Data Quality":
    st.title("Preprocessed Data Quality Check")
    st.header("Preprocessed Dataset Overview")
    st.write(preprocessed_data.head())

    st.subheader("1. Duplicates Check")
    duplicates_count = preprocessed_data.duplicated().sum()
    if duplicates_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning(f"Found {duplicates_count} duplicate rows.")

    st.subheader("2. Missing Values Check")
    missing_values = preprocessed_data.isnull().sum()
    if missing_values.sum() == 0:
        st.success("No missing values in the dataset.")
    else:
        st.warning("Missing values detected:")
        st.write(missing_values[missing_values > 0])

    st.subheader("3. Summary of Improvements")
    st.write("""
        - Duplicates removed.
        - Missing values handled.
        - Outliers reduced to improve data quality.
        - Data skewness improved.
    """)

if page == "Outliers":
    st.title("Outliers Analysis")
    data_selection = st.selectbox("Select Dataset:", ["Raw Data", "Preprocessed Data"])
    data = raw_data if data_selection == "Raw Data" else preprocessed_data
    st.header(f"Outliers in {data_selection}")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    selected_feature = st.selectbox("Select a Numeric Feature for Outliers:", numeric_columns)

    fig, ax = plt.subplots()
    sns.boxplot(x=data[selected_feature], ax=ax)
    ax.set_title(f"Outliers in {selected_feature} ({data_selection})")
    st.pyplot(fig)

    download_plot(fig, f"Outliers_{selected_feature}_{data_selection}.png")

elif page == "Frequency Distribution":
    st.title("Frequency Distribution Analysis")
    data_selection = st.selectbox("Select Dataset:", ["Raw Data", "Preprocessed Data"])
    data = raw_data if data_selection == "Raw Data" else preprocessed_data
    st.header(f"Frequency Distribution in {data_selection}")
    selected_feature = st.selectbox("Select a Feature for Distribution:", data.columns)

    fig, ax = plt.subplots()
    sns.histplot(data[selected_feature], kde=True, bins=20, ax=ax)
    ax.set_title(f"Frequency Distribution of {selected_feature} ({data_selection})")
    st.pyplot(fig)

    download_plot(fig, f"Frequency_Distribution_{selected_feature}_{data_selection}.png")

elif page == "Data Skewness":
    st.title("Data Skewness Analysis")
    data_selection = st.selectbox("Select Dataset:", ["Raw Data", "Preprocessed Data"])
    data = raw_data if data_selection == "Raw Data" else preprocessed_data
    st.header(f"Data Skewness in {data_selection}")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

    skewness = data[numeric_columns].skew()
    st.write("Skewness of Numeric Features:")
    st.write(skewness)

    highly_skewed = skewness[abs(skewness) > 1]
    if not highly_skewed.empty:
        st.subheader("Highly Skewed Features (|skewness| > 1):")
        st.write(highly_skewed)
    else:
        st.write("No highly skewed features.")

elif page == "Correlation Analysis":
    st.title("Correlation Analysis")
    data_selection = st.selectbox("Select Dataset:", ["Raw Data", "Preprocessed Data"])
    data = raw_data if data_selection == "Raw Data" else preprocessed_data
    st.header(f"Correlation Analysis for {data_selection}")

    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    selected_features = st.multiselect("Select Features for Correlation Analysis:", numeric_columns)

    if len(selected_features) >= 2:
        correlation = data[selected_features].corr()

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        download_plot(fig, f"Correlation_Heatmap_{data_selection}.png")

        st.subheader("Correlation Table")
        st.write(correlation)
    else:
        st.warning("Please select at least two features for correlation analysis.")

elif page == "3D Surface Plot":
    st.title("3D Surface Plot")
    st.header("Visualize Data Relationships in 3D")

    data = preprocessed_data
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

    x_attr = st.selectbox("Select X-axis Attribute:", numeric_columns, index=numeric_columns.get_loc("Hours_Studied"))
    y_attr = st.selectbox("Select Y-axis Attribute:", [col for col in numeric_columns if col != x_attr], index=numeric_columns.get_loc("Exam_Score") - 1)
    z_attr = st.selectbox("Select Z-axis Attribute:", [col for col in numeric_columns if col != x_attr and col != y_attr], index=numeric_columns.get_loc("Sleep_Hours") - 2)

    if x_attr and y_attr and z_attr:
        x = data[x_attr].values
        y = data[y_attr].values
        z = data[z_attr].values

        try:
            xi, yi = np.linspace(x.min(), x.max(), 50), np.linspace(y.min(), y.max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi, yi), method='linear')

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.8)

            ax.set_xlabel(x_attr)
            ax.set_ylabel(y_attr)
            ax.set_zlabel(z_attr)
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            plt.title("3D Surface Plot")
            st.pyplot(fig)

            download_plot(fig, f"3D_Surface_{x_attr}_{y_attr}_{z_attr}.png")

        except Exception as e:
            st.error(f"Surface plot failed due to: {e}. Displaying a scatter plot with jitter instead.")
            jitter_x = x + np.random.normal(0, 0.01, size=x.shape)
            jitter_y = y + np.random.normal(0, 0.01, size=y.shape)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(jitter_x, jitter_y, z, c=z, cmap='viridis', alpha=0.8)
            ax.set_xlabel(x_attr)
            ax.set_ylabel(y_attr)
            ax.set_zlabel(z_attr)
            plt.title("3D Scatter Plot with Jitter")
            st.pyplot(fig)

elif page == "Comparison":
    st.title("Raw vs Preprocessed Data Comparison")

    st.subheader("Column Differences")
    raw_columns = set(raw_data.columns)
    preprocessed_columns = set(preprocessed_data.columns)
    added_columns = preprocessed_columns - raw_columns
    removed_columns = raw_columns - preprocessed_columns

    st.write(f"Added Columns: {added_columns if added_columns else 'None'}")
    st.write(f"Removed Columns: {removed_columns if removed_columns else 'None'}")

    st.subheader("Record Count Comparison")
    st.write(f"Raw Dataset: {raw_data.shape[0]} rows")
    st.write(f"Preprocessed Dataset: {preprocessed_data.shape[0]} rows")

    st.subheader("Outliers Comparison")
    numeric_columns = raw_data.select_dtypes(include=["float64", "int64"]).columns
    selected_feature = st.selectbox("Select a Numeric Feature for Outliers Comparison:", numeric_columns)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(x=raw_data[selected_feature], ax=ax[0])
    ax[0].set_title(f"Raw: Outliers in {selected_feature}")
    sns.boxplot(x=preprocessed_data[selected_feature], ax=ax[1])
    ax[1].set_title(f"Preprocessed: Outliers in {selected_feature}")
    st.pyplot(fig)

    download_plot(fig, f"Comparison_Outliers_{selected_feature}.png")

    st.subheader("Skewness Comparison")
    raw_skewness = raw_data[numeric_columns].skew()
    preprocessed_skewness = preprocessed_data[numeric_columns].skew()
    skewness_comparison = pd.DataFrame({
        "Raw Skewness": raw_skewness,
        "Preprocessed Skewness": preprocessed_skewness
    })
    st.write(skewness_comparison)

elif page == "Class Imbalance":
    st.title("Class Imbalance: Raw vs Preprocessed Data")

    raw_data = pd.read_csv(raw_file_path)
    main_df = raw_data.copy()

    main_df['Score_Category'] = pd.cut(main_df['Exam_Score'], bins=[0, 59, 80, 100], labels=['Low', 'Medium', 'High'])

    categorical_cols = main_df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = main_df.select_dtypes(include=['int64', 'float64']).columns

    for col in categorical_cols:
        main_df[col] = main_df[col].fillna(main_df[col].mode()[0])

    main_df[numerical_cols] = main_df[numerical_cols].fillna(main_df[numerical_cols].median())

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        main_df[col] = le.fit_transform(main_df[col].astype(str))
        label_encoders[col] = le

    X = main_df.drop(['Exam_Score', 'Score_Category'], axis=1)
    y = main_df['Score_Category']

    class_counts_before = y.value_counts()

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    class_counts_after = pd.Series(y_resampled).value_counts()

    st.subheader("Select Visualization Type")
    visualization_type = st.selectbox("Choose how to visualize the class balance:", ["Bar Chart", "Pie Chart"])

    st.subheader("Class Balance Visualization")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    if visualization_type == "Bar Chart":
        class_counts_before.plot(kind='bar', color='lightcoral', ax=ax[0])
        ax[0].set_title('Before SMOTE')
        ax[0].set_xlabel('Score Category')
        ax[0].set_ylabel('Count')
        for i, v in enumerate(class_counts_before):
            ax[0].text(i, v + 1, str(v), ha='center')

        class_counts_after.plot(kind='bar', color='skyblue', ax=ax[1])
        ax[1].set_title('After SMOTE')
        ax[1].set_xlabel('Score Category')
        ax[1].set_ylabel('Count')
        for i, v in enumerate(class_counts_after):
            ax[1].text(i, v + 1, str(v), ha='center')

    elif visualization_type == "Pie Chart":
        ax[0].pie(class_counts_before, labels=class_counts_before.index, autopct='%1.1f%%', colors=['lightcoral', 'salmon', 'red'])
        ax[0].set_title('Before SMOTE')

        ax[1].pie(class_counts_after, labels=class_counts_after.index, autopct='%1.1f%%', colors=['skyblue', 'deepskyblue', 'dodgerblue'])
        ax[1].set_title('After SMOTE')

    plt.tight_layout()
    st.pyplot(fig)

# Run Streamlit app by typing: streamlit run dashboard.py
