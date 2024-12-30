# # schisto_eda_app.py
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
#
# # -------------------------- Streamlit App -------------------------- #
#
# def main():
#     # Set the title of the app
#     st.title("Schistosomiasis Survey Data EDA and Regression Analysis")
#
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     options = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Regression Analysis"])
#
#     # Load the data
#     data_file = "Schisto Survey Coding - Data For Analysis.csv"  # Ensure this file is in the same directory
#     df = load_data(data_file)
#
#     if df is None:
#         st.error("Failed to load data. Please check the file path and format.")
#         return
#
#     if options == "Home":
#         home_page()
#
#     elif options == "Data Overview":
#         data_overview(df)
#
#     elif options == "EDA":
#         eda_section(df)
#
#     elif options == "Regression Analysis":
#         regression_section(df)
#
#     else:
#         st.error("Invalid option selected.")
#
# def load_data(filepath):
#     """
#     Loads and preprocesses the survey data.
#
#     Parameters:
#     - filepath (str): Path to the survey CSV file.
#
#     Returns:
#     - pd.DataFrame: Loaded and preprocessed DataFrame.
#     """
#     try:
#         df = pd.read_csv(filepath)
#         st.sidebar.success(f"Data loaded successfully from '{filepath}'")
#     except FileNotFoundError:
#         st.sidebar.error(f"The file '{filepath}' does not exist in the specified directory.")
#         return None
#     except pd.errors.ParserError:
#         st.sidebar.error(f"Error parsing the file '{filepath}'. Please ensure it's a valid CSV.")
#         return None
#     except Exception as e:
#         st.sidebar.error(f"An unexpected error occurred: {e}")
#         return None
#
#     # Handle missing values: Replace 'N/A' with np.nan for proper handling
#     df.replace('N/A', np.nan, inplace=True)
#
#     # Fill remaining NaN values with 'No' for categorical variables or 0 for numerical if appropriate
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#     categorical_cols.remove('ID') if 'ID' in categorical_cols else None
#     df[categorical_cols] = df[categorical_cols].fillna('No')
#
#     # For numerical columns, fill NaN with median or a specific value if necessary
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#
#     # Ensure 'ID' is treated as a string to avoid numeric issues
#     if 'ID' in df.columns:
#         df['ID'] = df['ID'].astype(str)
#
#     return df
#
# def home_page():
#     """
#     Displays the Home page content.
#     """
#     st.header("Welcome to the Schistosomiasis Data Analysis App")
#     st.write("""
#         This application allows you to perform **Exploratory Data Analysis (EDA)** and
#         **Logistic Regression Analysis** on the Schistosomiasis survey dataset.
#
#         ### **Features:**
#         - **Data Overview:** Inspect the dataset, view summary statistics, and understand data types.
#         - **EDA:** Create interactive plots such as bar charts, histograms, box plots, and correlation matrices.
#         - **Regression Analysis:** Build and evaluate logistic regression models to predict the Schistosomiasis Test Result.
#
#         Navigate through the sidebar to explore different functionalities.
#     """)
#     st.image("https://i.imgur.com/4UQaxS1.png", use_column_width=True)  # Optional: Add an illustrative image
#
# def data_overview(df):
#     """
#     Displays the Data Overview section.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.header("Dataset Overview")
#
#     st.subheader("First 5 Rows of the Dataset")
#     st.dataframe(df.head())
#
#     st.subheader("Summary Statistics")
#     st.write(df.describe(include='all'))
#
#     st.subheader("Data Types")
#     data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
#     st.dataframe(data_types)
#
#     st.subheader("Missing Values")
#     missing_values = df.isnull().sum()
#     st.write(missing_values)
#
# def eda_section(df):
#     """
#     Displays the Exploratory Data Analysis (EDA) section.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.header("Exploratory Data Analysis (EDA)")
#
#     # Select EDA type
#     eda_options = ["Bar Chart", "Histogram", "Box Plot", "Correlation Matrix"]
#     eda_choice = st.selectbox("Select EDA Type", eda_options)
#
#     if eda_choice == "Bar Chart":
#         bar_chart(df)
#     elif eda_choice == "Histogram":
#         histogram(df)
#     elif eda_choice == "Box Plot":
#         box_plot(df)
#     elif eda_choice == "Correlation Matrix":
#         correlation_matrix(df)
#     else:
#         st.error("Invalid EDA type selected.")
#
# def bar_chart(df):
#     """
#     Creates a bar chart based on user-selected X and Y variables.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.subheader("Bar Chart")
#
#     # Identify categorical columns
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#     categorical_cols.remove('ID') if 'ID' in categorical_cols else None
#
#     # User selections
#     x_var = st.selectbox("Select X-axis Variable", categorical_cols)
#     y_var = st.selectbox("Select Y-axis Variable", categorical_cols)
#
#     if x_var and y_var:
#         fig, ax = plt.subplots(figsize=(10,6))
#         sns.countplot(data=df, x=x_var, hue=y_var, ax=ax)
#         plt.xticks(rotation=45)
#         plt.title(f"Count of {y_var} by {x_var}")
#         st.pyplot(fig)
#
# def histogram(df):
#     """
#     Creates a histogram for a selected numerical variable.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.subheader("Histogram")
#
#     # Identify numerical columns
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#
#     # User selection
#     var = st.selectbox("Select Variable for Histogram", numerical_cols)
#
#     if var:
#         fig, ax = plt.subplots(figsize=(10,6))
#         sns.histplot(df[var], kde=True, ax=ax, bins=30, color='skyblue')
#         plt.title(f"Distribution of {var}")
#         st.pyplot(fig)
#
# def box_plot(df):
#     """
#     Creates a box plot for a selected numerical variable grouped by a categorical variable.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.subheader("Box Plot")
#
#     # Identify numerical and categorical columns
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#     categorical_cols.remove('ID') if 'ID' in categorical_cols else None
#
#     # User selections
#     var = st.selectbox("Select Numerical Variable", numerical_cols)
#     group = st.selectbox("Select Grouping Variable", categorical_cols)
#
#     if var and group:
#         fig, ax = plt.subplots(figsize=(10,6))
#         sns.boxplot(data=df, x=group, y=var, ax=ax)
#         plt.xticks(rotation=45)
#         plt.title(f"{var} by {group}")
#         st.pyplot(fig)
#
# def correlation_matrix(df):
#     """
#     Displays a correlation matrix heatmap for numerical variables.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.subheader("Correlation Matrix")
#
#     # Identify numerical columns
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#
#     if numerical_cols:
#         corr = df[numerical_cols].corr()
#         fig, ax = plt.subplots(figsize=(12,10))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
#         plt.title("Correlation Matrix of Numerical Variables")
#         st.pyplot(fig)
#     else:
#         st.write("No numerical variables available for correlation matrix.")
#
# def regression_section(df):
#     """
#     Displays the Regression Analysis section.
#
#     Parameters:
#     - df (pd.DataFrame): The survey DataFrame.
#     """
#     st.header("Regression Analysis: Predicting Schistosomiasis Test Result")
#
#     # Define the target variable
#     target = 'Schistosomiasis Test Result'
#     if target not in df.columns:
#         st.error(f"Target variable '{target}' not found in the dataset.")
#         return
#
#     # Encode the target variable
#     df[target] = df[target].map({'Positive':1, 'Negative':0})
#
#     # Check for any remaining NaN in target
#     if df[target].isnull().sum() > 0:
#         st.warning("Some entries in the target variable are missing and will be excluded from the analysis.")
#         df = df.dropna(subset=[target])
#
#     # Features and target
#     X = df.drop(['ID', target], axis=1)
#     y = df[target]
#
#     # Identify categorical and numerical features
#     categorical_features = X.select_dtypes(include=['object']).columns.tolist()
#     numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#
#     # Encode categorical variables using One-Hot Encoding
#     X_encoded = pd.get_dummies(X, drop_first=True)
#
#     # Replace any remaining 'No' or other strings with appropriate numeric values if necessary
#     # Assuming all categorical variables are now encoded
#
#     # Check for any remaining non-numeric columns
#     non_numeric_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
#     if non_numeric_cols:
#         st.error(f"The following columns are still non-numeric and cannot be used for regression: {non_numeric_cols}")
#         return
#
#     # Check for any remaining NaN or infinite values
#     if X_encoded.isnull().sum().sum() > 0 or not np.isfinite(X_encoded.values).all():
#         st.error("The feature set contains NaN or infinite values. Please clean the data before proceeding.")
#         return
#
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#
#     # Feature Scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     # Initialize and train the Logistic Regression model
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train_scaled, y_train)
#
#     # Predictions
#     y_pred = model.predict(X_test_scaled)
#     y_prob = model.predict_proba(X_test_scaled)[:,1]
#
#     # Model Evaluation
#     st.subheader("Model Performance")
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write(f"**Accuracy:** {accuracy:.2f}")
#
#     st.write("**Confusion Matrix:**")
#     cm = confusion_matrix(y_test, y_pred)
#     fig_cm, ax_cm = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax_cm)
#     ax_cm.set_xlabel('Predicted')
#     ax_cm.set_ylabel('Actual')
#     ax_cm.set_title('Confusion Matrix')
#     st.pyplot(fig_cm)
#
#     st.write("**Classification Report:**")
#     report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
#     report_df = pd.DataFrame(report).transpose()
#     st.dataframe(report_df)
#
#     st.write("**ROC Curve:**")
#     fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#     roc_auc = auc(fpr, tpr)
#
#     fig_roc, ax_roc = plt.subplots()
#     ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
#     ax_roc.plot([0,1], [0,1], 'k--')
#     ax_roc.set_xlabel('False Positive Rate')
#     ax_roc.set_ylabel('True Positive Rate')
#     ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
#     ax_roc.legend(loc='lower right')
#     st.pyplot(fig_roc)
#
#     # Display model coefficients
#     st.subheader("Model Coefficients")
#     coef_df = pd.DataFrame({
#         'Feature': X_encoded.columns,
#         'Coefficient': model.coef_[0]
#     })
#     coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
#     st.dataframe(coef_df)
#
#     # Allow user to select features for a customized model
#     st.subheader("Customize Regression Model")
#     all_features = X_encoded.columns.tolist()
#     selected_features = st.multiselect("Select Features for the Model", all_features, default=all_features[:10])  # Default to first 10 for brevity
#
#     if st.button("Run Custom Regression Model"):
#         if not selected_features:
#             st.error("Please select at least one feature.")
#         else:
#             X_custom = X_encoded[selected_features]
#             X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_custom, y, test_size=0.2, random_state=42)
#
#             # Feature Scaling
#             scaler_c = StandardScaler()
#             X_train_scaled_c = scaler_c.fit_transform(X_train_c)
#             X_test_scaled_c = scaler_c.transform(X_test_c)
#
#             # Initialize and train the model
#             model_c = LogisticRegression(max_iter=1000)
#             model_c.fit(X_train_scaled_c, y_train_c)
#
#             # Predictions
#             y_pred_c = model_c.predict(X_test_scaled_c)
#             y_prob_c = model_c.predict_proba(X_test_scaled_c)[:,1]
#
#             # Model Evaluation
#             st.write(f"**Accuracy:** {accuracy_score(y_test_c, y_pred_c):.2f}")
#
#             st.write("**Confusion Matrix:**")
#             cm_c = confusion_matrix(y_test_c, y_pred_c)
#             fig_cm_c, ax_cm_c = plt.subplots()
#             sns.heatmap(cm_c, annot=True, fmt="d", cmap='Greens', ax=ax_cm_c)
#             ax_cm_c.set_xlabel('Predicted')
#             ax_cm_c.set_ylabel('Actual')
#             ax_cm_c.set_title('Confusion Matrix (Custom Model)')
#             st.pyplot(fig_cm_c)
#
#             st.write("**Classification Report:**")
#             report_c = classification_report(y_test_c, y_pred_c, target_names=['Negative', 'Positive'], output_dict=True)
#             report_df_c = pd.DataFrame(report_c).transpose()
#             st.dataframe(report_df_c)
#
#             st.write("**ROC Curve:**")
#             fpr_c, tpr_c, thresholds_c = roc_curve(y_test_c, y_prob_c)
#             roc_auc_c = auc(fpr_c, tpr_c)
#
#             fig_roc_c, ax_roc_c = plt.subplots()
#             ax_roc_c.plot(fpr_c, tpr_c, label=f'ROC Curve (AUC = {roc_auc_c:.2f})')
#             ax_roc_c.plot([0,1], [0,1], 'k--')
#             ax_roc_c.set_xlabel('False Positive Rate')
#             ax_roc_c.set_ylabel('True Positive Rate')
#             ax_roc_c.set_title('Receiver Operating Characteristic (ROC) Curve (Custom Model)')
#             ax_roc_c.legend(loc='lower right')
#             st.pyplot(fig_roc_c)
#
#             # Model coefficients
#             st.write("**Model Coefficients:**")
#             coef_df_c = pd.DataFrame({
#                 'Feature': selected_features,
#                 'Coefficient': model_c.coef_[0]
#             })
#             coef_df_c = coef_df_c.sort_values(by='Coefficient', ascending=False)
#             st.dataframe(coef_df_c)
#
# # -------------------------- Streamlit Run -------------------------- #
#
# if __name__ == "__main__":
#     main()
# schisto_eda_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# -------------------------- Streamlit App -------------------------- #

def main():
    # Set the title of the app
    st.title("Schistosomiasis Survey Data EDA and Regression Analysis")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Regression Analysis"])

    # Load the data
    data_file = "Schisto Survey Coding - Data For Analysis.csv"  # Ensure this file is in the same directory
    df = load_data(data_file)

    if df is None:
        st.error("Failed to load data. Please check the file path and format.")
        return

    if options == "Home":
        home_page()

    elif options == "Data Overview":
        data_overview(df)

    elif options == "EDA":
        eda_section(df)

    elif options == "Regression Analysis":
        regression_section(df)

    else:
        st.error("Invalid option selected.")

def load_data(filepath):
    """
    Loads and preprocesses the survey data.

    Parameters:
    - filepath (str): Path to the survey CSV file.

    Returns:
    - pd.DataFrame: Loaded and preprocessed DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        st.sidebar.success(f"Data loaded successfully from '{filepath}'")
    except FileNotFoundError:
        st.sidebar.error(f"The file '{filepath}' does not exist in the specified directory.")
        return None
    except pd.errors.ParserError:
        st.sidebar.error(f"Error parsing the file '{filepath}'. Please ensure it's a valid CSV.")
        return None
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred: {e}")
        return None

    # Display initial information
    st.sidebar.write("## Data Preprocessing")

    # Handle missing values: Replace 'N/A' with np.nan for proper handling
    df.replace('N/A', np.nan, inplace=True)

    # Fill remaining NaN values with 'No' for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'ID']  # Exclude 'ID' from categorical variables

    df[categorical_cols] = df[categorical_cols].fillna('No')

    # For numerical columns, fill NaN with median or a specific value if necessary
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    # Ensure 'ID' is treated as a string to avoid numeric issues
    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str)

    # Drop 'Data Type' column if present to avoid serialization issues
    if 'Data Type' in df.columns:
        df = df.drop('Data Type', axis=1)
        st.sidebar.warning("Column 'Data Type' was found and has been excluded from the analysis.")

    return df

def home_page():
    """
    Displays the Home page content.
    """
    st.header("Welcome to the Schistosomiasis Data Analysis App")
    st.write("""
        This application allows you to perform **Exploratory Data Analysis (EDA)** and
        **Logistic Regression Analysis** on the Schistosomiasis survey dataset.

        ### **Features:**
        - **Data Overview:** Inspect the dataset, view summary statistics, and understand data types.
        - **EDA:** Create interactive plots such as bar charts, histograms, box plots, and correlation matrices.
        - **Regression Analysis:** Build and evaluate logistic regression models to predict the Schistosomiasis Test Result.

        Navigate through the sidebar to explore different functionalities.
    """)
    st.image("https://i.imgur.com/4UQaxS1.png", use_column_width=True)  # Optional: Add an illustrative image

def data_overview(df):
    """
    Displays the Data Overview section.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.header("Dataset Overview")

    st.subheader("First 5 Rows of the Dataset")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    st.subheader("Data Types")
    data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.dataframe(data_types)

    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

def eda_section(df):
    """
    Displays the Exploratory Data Analysis (EDA) section.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.header("Exploratory Data Analysis (EDA)")

    # Select EDA type
    eda_options = ["Bar Chart", "Histogram", "Box Plot", "Correlation Matrix"]
    eda_choice = st.selectbox("Select EDA Type", eda_options)

    if eda_choice == "Bar Chart":
        bar_chart(df)
    elif eda_choice == "Histogram":
        histogram(df)
    elif eda_choice == "Box Plot":
        box_plot(df)
    elif eda_choice == "Correlation Matrix":
        correlation_matrix(df)
    else:
        st.error("Invalid EDA type selected.")

def bar_chart(df):
    """
    Creates a bar chart based on user-selected X and Y variables.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.subheader("Bar Chart")

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'ID']  # Exclude 'ID' from categorical variables

    # User selections
    x_var = st.selectbox("Select X-axis Variable", categorical_cols)
    y_var = st.selectbox("Select Y-axis Variable", categorical_cols)

    if x_var and y_var:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.countplot(data=df, x=x_var, hue=y_var, ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Count of {y_var} by {x_var}")
        st.pyplot(fig)

def histogram(df):
    """
    Creates a histogram for a selected numerical variable.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.subheader("Histogram")

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # User selection
    var = st.selectbox("Select Variable for Histogram", numerical_cols)

    if var:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df[var], kde=True, ax=ax, bins=30, color='skyblue')
        plt.title(f"Distribution of {var}")
        st.pyplot(fig)

def box_plot(df):
    """
    Creates a box plot for a selected numerical variable grouped by a categorical variable.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.subheader("Box Plot")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'ID']  # Exclude 'ID' from categorical variables

    # User selections
    var = st.selectbox("Select Numerical Variable", numerical_cols)
    group = st.selectbox("Select Grouping Variable", categorical_cols)

    if var and group:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=df, x=group, y=var, ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"{var} by {group}")
        st.pyplot(fig)

def correlation_matrix(df):
    """
    Displays a correlation matrix heatmap for numerical variables.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.subheader("Correlation Matrix")

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numerical_cols:
        corr = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Correlation Matrix of Numerical Variables")
        st.pyplot(fig)
    else:
        st.write("No numerical variables available for correlation matrix.")

def regression_section(df):
    """
    Displays the Regression Analysis section.

    Parameters:
    - df (pd.DataFrame): The survey DataFrame.
    """
    st.header("Regression Analysis: Predicting Schistosomiasis Test Result")

    # Define the target variable
    target = 'Schistosomiasis Test Result'
    if target not in df.columns:
        st.error(f"Target variable '{target}' not found in the dataset.")
        return

    # Encode the target variable
    df[target] = df[target].map({'Positive':1, 'Negative':0})

    # Check for any remaining NaN in target
    if df[target].isnull().sum() > 0:
        st.warning("Some entries in the target variable are missing and will be excluded from the analysis.")
        df = df.dropna(subset=[target])

    # Features and target
    features_to_exclude = ['ID', target]
    # Additionally exclude 'Data Type' if present
    if 'Data Type' in df.columns:
        features_to_exclude.append('Data Type')

    X = df.drop(features_to_exclude, axis=1)
    y = df[target]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encode categorical variables using One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Check for any remaining non-numeric columns
    non_numeric_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        st.error(f"The following columns are still non-numeric and cannot be used for regression: {non_numeric_cols}")
        return

    # Force conversion to numeric (just in case)
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')

    # Replace any remaining 'inf' values with NaN and then fill NaNs
    X_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_encoded = X_encoded.fillna(X_encoded.median())

    # Check for any remaining NaN or infinite values
    if X_encoded.isnull().sum().sum() > 0 or not np.isfinite(X_encoded.values).all():
        st.error("The feature set contains NaN or infinite values even after cleaning. Please clean the data further.")
        return

    # Debugging Information: Display data types after encoding and cleaning
    st.write("**Data Types After Encoding and Cleaning:**")
    st.write(X_encoded.dtypes)

    # Display the number of features
    st.write(f"**Total Features after Encoding:** {X_encoded.shape[1]}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        st.error(f"An error occurred while training the model: {e}")
        return

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    # Model Evaluation
    st.subheader("Model Performance")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {accuracy:.2f}")

    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write("**ROC Curve:**")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0,1], [0,1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    # Display model coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Coefficient': model.coef_[0]
    })
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    st.dataframe(coef_df)

    # Allow user to select features for a customized model
    st.subheader("Customize Regression Model")
    all_features = X_encoded.columns.tolist()
    selected_features = st.multiselect("Select Features for the Model", all_features, default=all_features[:10])  # Default to first 10 for brevity

    if st.button("Run Custom Regression Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            X_custom = X_encoded[selected_features]
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_custom, y, test_size=0.2, random_state=42)

            # Feature Scaling
            scaler_c = StandardScaler()
            X_train_scaled_c = scaler_c.fit_transform(X_train_c)
            X_test_scaled_c = scaler_c.transform(X_test_c)

            # Initialize and train the model
            model_c = LogisticRegression(max_iter=1000)
            try:
                model_c.fit(X_train_scaled_c, y_train_c)
            except Exception as e:
                st.error(f"An error occurred while training the custom model: {e}")
                return

            # Predictions
            y_pred_c = model_c.predict(X_test_scaled_c)
            y_prob_c = model_c.predict_proba(X_test_scaled_c)[:,1]

            # Model Evaluation
            st.write(f"**Accuracy:** {accuracy_score(y_test_c, y_pred_c):.2f}")

            st.write("**Confusion Matrix:**")
            cm_c = confusion_matrix(y_test_c, y_pred_c)
            fig_cm_c, ax_cm_c = plt.subplots()
            sns.heatmap(cm_c, annot=True, fmt="d", cmap='Greens', ax=ax_cm_c)
            ax_cm_c.set_xlabel('Predicted')
            ax_cm_c.set_ylabel('Actual')
            ax_cm_c.set_title('Confusion Matrix (Custom Model)')
            st.pyplot(fig_cm_c)

            st.write("**Classification Report:**")
            report_c = classification_report(y_test_c, y_pred_c, target_names=['Negative', 'Positive'], output_dict=True)
            report_df_c = pd.DataFrame(report_c).transpose()
            st.dataframe(report_df_c)

            st.write("**ROC Curve:**")
            fpr_c, tpr_c, thresholds_c = roc_curve(y_test_c, y_prob_c)
            roc_auc_c = auc(fpr_c, tpr_c)

            fig_roc_c, ax_roc_c = plt.subplots()
            ax_roc_c.plot(fpr_c, tpr_c, label=f'ROC Curve (AUC = {roc_auc_c:.2f})')
            ax_roc_c.plot([0,1], [0,1], 'k--')
            ax_roc_c.set_xlabel('False Positive Rate')
            ax_roc_c.set_ylabel('True Positive Rate')
            ax_roc_c.set_title('Receiver Operating Characteristic (ROC) Curve (Custom Model)')
            ax_roc_c.legend(loc='lower right')
            st.pyplot(fig_roc_c)

            # Model coefficients
            st.write("**Model Coefficients:**")
            coef_df_c = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': model_c.coef_[0]
            })
            coef_df_c = coef_df_c.sort_values(by='Coefficient', ascending=False)
            st.dataframe(coef_df_c)

            # Additional Debugging Information
            st.write("**Custom Model Data Types:**")
            st.write(X_custom.dtypes)

# -------------------------- Streamlit Run -------------------------- #

if __name__ == "__main__":
    main()
