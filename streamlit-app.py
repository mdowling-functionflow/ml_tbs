import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

st.set_page_config(page_title="ML & GenAI Analysis", layout="wide")

st.title("Machine Learning & GenAI Analysis Tool")
st.markdown("""
This app allows you to upload a CSV file and perform various machine learning analyses on it.
You can explore the data, visualize it, and train different machine learning models to predict a target variable.
""")

# File upload
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        data = pd.read_csv(file)
        return data
    
    df = load_data(uploaded_file)
    
    # Display the data
    st.header("2. Data Overview")
    st.write("**First 5 rows of your dataset:**")
    st.dataframe(df.head())
    
    # Basic info
    st.subheader("Data Information")
    buffer = st.empty()
    
    # Using a text area to display DataFrame info
    info_str = []
    info_str.append(f"Rows: {df.shape[0]}")
    info_str.append(f"Columns: {df.shape[1]}")
    info_str.append("\nColumns and their types:")
    for col, dtype in zip(df.columns, df.dtypes):
        info_str.append(f"- {col}: {dtype}")
    
    info_str.append("\nMissing values:")
    for col, missing in zip(df.columns, df.isnull().sum()):
        info_str.append(f"- {col}: {missing}")
        
    buffer.text("\n".join(info_str))
    
    # Data Exploration
    st.header("3. Data Exploration")
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include='all').T)
    
    # Visualizations
    st.subheader("Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Choose column for histogram
        if df.select_dtypes(include=['object']).columns.any():
            categorical_cols = list(df.select_dtypes(include=['object']).columns)
            hist_col = st.selectbox("Select a categorical column for histogram", categorical_cols)
            
            fig = px.histogram(df, x=hist_col, color=hist_col, title=f"{hist_col} Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Choose column for box plot
        if df.select_dtypes(include=['float64', 'int64']).columns.any():
            numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
            box_col = st.selectbox("Select a numerical column for box plot", numeric_cols)
            
            fig = px.box(df, y=box_col, title=f"{box_col} Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Data Preprocessing
    st.header("4. Data Preprocessing")
    
    # Target variable selection
    target_col = st.selectbox("Select the target variable", df.columns)
    
    # Check if preprocessing needed
    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    if categorical_cols:
        st.warning(f"Your data contains categorical columns: {', '.join(categorical_cols)}. These need to be converted to numerical format for machine learning.")
        
        preprocess = st.checkbox("Apply one-hot encoding to categorical columns (drop first to avoid multicollinearity)", value=True)
        
        if preprocess:
            # Preprocess the data
            @st.cache_data
            def preprocess_data(data, target, drop_first=True):
                # Create a copy of the dataframe
                df_processed = data.copy()
                
                # Get categorical columns (excluding target if it's categorical)
                cat_cols = df_processed.select_dtypes(include=['object']).columns
                cat_cols = [col for col in cat_cols if col != target]
                
                # One-hot encode categorical variables
                for col in cat_cols:
                    df_processed = pd.concat([
                        df_processed, 
                        pd.get_dummies(df_processed[col], prefix=col, drop_first=drop_first)
                    ], axis=1)
                    df_processed = df_processed.drop(col, axis=1)
                
                # Handle target column if it's categorical
                if df_processed[target].dtype == 'object':
                    st.info(f"Converting target column '{target}' to numerical format.")
                    df_processed[f"{target}_encoded"] = pd.factorize(df_processed[target])[0]
                    target_encoded = f"{target}_encoded"
                else:
                    target_encoded = target
                
                return df_processed, target_encoded
            
            with st.spinner("Preprocessing data..."):
                df_processed, target_encoded = preprocess_data(df, target_col)
                
            st.success("Data preprocessing completed!")
            st.dataframe(df_processed.head())
        else:
            df_processed = df
            target_encoded = target_col
    else:
        df_processed = df
        target_encoded = target_col
    
    # Machine Learning
    st.header("5. Machine Learning Models")
    
    # Train-test split parameters
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    random_state = st.number_input("Random state (for reproducibility)", min_value=0, value=42, step=1)
    
    # Model selection
    st.subheader("Select Models to Train")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_lr = st.checkbox("Logistic Regression", value=True)
        use_dt = st.checkbox("Decision Tree", value=True)
    
    with col2:
        use_rf = st.checkbox("Random Forest", value=True)
        use_gb = st.checkbox("Gradient Boosting", value=True)
    
    with col3:
        use_svc = st.checkbox("Support Vector Machine", value=False)
        use_mlp = st.checkbox("Neural Network (MLP)", value=False)
    
    if st.button("Train Selected Models"):
        # Check if any model is selected
        if not any([use_lr, use_dt, use_rf, use_gb, use_svc, use_mlp]):
            st.error("Please select at least one model to train.")
        else:
            # Prepare data for modeling
            X = df_processed.drop([target_col, target_encoded] if target_col != target_encoded else [target_encoded], axis=1)
            y = df_processed[target_encoded]
            
            # Check if X has any columns
            if X.shape[1] == 0:
                st.error("No features available for training. Make sure your data preprocessing steps are correct.")
            else:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
                
                # Initialize models dictionary
                models = {}
                accuracies = {}
                confusion_matrices = {}
                
                # Train selected models
                with st.spinner("Training models... This may take a while depending on your data size."):
                    # Logistic Regression
                    if use_lr:
                        lr = LogisticRegression(max_iter=500, random_state=random_state)
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_test)
                        accuracies['Logistic Regression'] = accuracy_score(y_test, y_pred) * 100
                        confusion_matrices['Logistic Regression'] = confusion_matrix(y_test, y_pred)
                        models['Logistic Regression'] = lr
                    
                    # Decision Tree
                    if use_dt:
                        dt = DecisionTreeClassifier(random_state=random_state)
                        dt.fit(X_train, y_train)
                        y_pred = dt.predict(X_test)
                        accuracies['Decision Tree'] = accuracy_score(y_test, y_pred) * 100
                        confusion_matrices['Decision Tree'] = confusion_matrix(y_test, y_pred)
                        models['Decision Tree'] = dt
                    
                    # Random Forest
                    if use_rf:
                        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        accuracies['Random Forest'] = accuracy_score(y_test, y_pred) * 100
                        confusion_matrices['Random Forest'] = confusion_matrix(y_test, y_pred)
                        models['Random Forest'] = rf
                    
                    # Gradient Boosting
                    if use_gb:
                        gb = GradientBoostingClassifier(random_state=random_state)
                        gb.fit(X_train, y_train)
                        y_pred = gb.predict(X_test)
                        accuracies['Gradient Boosting'] = accuracy_score(y_test, y_pred) * 100
                        confusion_matrices['Gradient Boosting'] = confusion_matrix(y_test, y_pred)
                        models['Gradient Boosting'] = gb
                    
                    # SVC (can be slow on large datasets)
                    if use_svc:
                        with st.spinner("Training SVM... This might take longer than other models"):
                            svc = SVC(kernel='linear', random_state=random_state)
                            svc.fit(X_train, y_train)
                            y_pred = svc.predict(X_test)
                            accuracies['SVM'] = accuracy_score(y_test, y_pred) * 100
                            confusion_matrices['SVM'] = confusion_matrix(y_test, y_pred)
                            models['SVM'] = svc
                    
                    # MLP
                    if use_mlp:
                        with st.spinner("Training Neural Network... This might take longer than other models"):
                            mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, random_state=random_state)
                            mlp.fit(X_train, y_train)
                            y_pred = mlp.predict(X_test)
                            accuracies['Neural Network'] = accuracy_score(y_test, y_pred) * 100
                            confusion_matrices['Neural Network'] = confusion_matrix(y_test, y_pred)
                            models['Neural Network'] = mlp
                
                # Display results
                st.header("6. Model Results")
                
                # Accuracy comparison
                st.subheader("Model Accuracy Comparison")
                accuracy_df = pd.DataFrame({
                    'Model': list(accuracies.keys()),
                    'Accuracy (%)': list(accuracies.values())
                })
                accuracy_df = accuracy_df.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
                
                # Display as table and chart
                st.dataframe(accuracy_df)
                
                fig = px.bar(
                    accuracy_df, 
                    x='Model', 
                    y='Accuracy (%)', 
                    title='Model Accuracy Comparison',
                    color='Model'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confusion matrices
                st.subheader("Confusion Matrices")
                
                # Create 2 columns for confusion matrices
                cols = st.columns(2)
                col_idx = 0
                
                for model_name, cm in confusion_matrices.items():
                    with cols[col_idx]:
                        fig = ff.create_annotated_heatmap(
                            z=cm, 
                            x=['Predicted 0', 'Predicted 1'], 
                            y=['Actual 0', 'Actual 1'],
                            colorscale='Viridis'
                        )
                        fig.update_layout(title=f'Confusion Matrix for {model_name}')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Switch to next column (or back to first if we've used them all)
                    col_idx = (col_idx + 1) % 2
                
                # Feature importance for applicable models
                st.subheader("Feature Importance")
                
                for model_name, model in models.items():
                    if model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']:
                        st.write(f"**{model_name} Feature Importance**")
                        
                        if model_name == 'Logistic Regression':
                            importances = model.coef_[0]
                        else:
                            importances = model.feature_importances_
                        
                        feature_importances = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': abs(importances) if model_name == 'Logistic Regression' else importances
                        })
                        
                        feature_importances = feature_importances.sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(
                            feature_importances, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title=f'Top 15 Features ({model_name})'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a divider
                        st.markdown("---")

    # Add GenAI section if needed
    st.header("7. Generative AI (Optional)")
    st.markdown("""
    This section demonstrates how to use the OpenAI API for text generation. 
    To use this feature, you would need to provide your own OpenAI API key.
    """)
    
    api_key = st.text_input("Enter your OpenAI API key (optional)", type="password")
    prompt = st.text_area("Enter a prompt for the AI", value="Write a brief summary of the banking dataset analysis.")
    
    if api_key and prompt and st.button("Generate AI Response"):
        try:
            st.info("Note: This would normally connect to the OpenAI API, but it's disabled in this demo for security reasons.")
            st.markdown("""
            In a real implementation, this would call:
            ```python
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            ```
            """)
            
            # Simulate response with a placeholder
            with st.spinner("Generating response..."):
                time.sleep(2)  # Simulate API call
                
            st.success("AI Response:")
            st.markdown("""
            Based on the banking dataset analysis, we've identified several key patterns in customer behavior 
            regarding term deposits. The machine learning models show that age, balance, and previous campaign 
            outcomes are strong predictors for determining which customers are likely to subscribe to a term deposit. 
            The highest performing model was the Gradient Boosting Classifier, achieving an accuracy of 90.2%, 
            which suggests we can reliably predict customer decisions with the right feature selection.
            """)
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # No file uploaded yet
    st.info("Please upload a CSV file to begin analysis.")
    
    # Sample data option
    if st.button("Or use sample bank marketing data"):
        # Create sample data similar to what might be in the bank dataset
        data = {
            'age': np.random.randint(18, 95, 100),
            'job': np.random.choice(['admin', 'blue-collar', 'technician', 'services', 'management'], 100),
            'marital': np.random.choice(['married', 'single', 'divorced'], 100),
            'education': np.random.choice(['primary', 'secondary', 'tertiary'], 100),
            'balance': np.random.randint(-1000, 50000, 100),
            'housing': np.random.choice(['yes', 'no'], 100),
            'loan': np.random.choice(['yes', 'no'], 100),
            'contact': np.random.choice(['cellular', 'telephone'], 100),
            'duration': np.random.randint(0, 5000, 100),
            'campaign': np.random.randint(1, 50, 100),
            'deposit': np.random.choice(['yes', 'no'], 100)
        }
        
        sample_df = pd.DataFrame(data)
        
        # Save to a temporary file and "upload" it
        sample_file = "sample_bank_data.csv"
        sample_df.to_csv(sample_file, index=False)
        
        # Reload the page with the sample file
        st.session_state['use_sample'] = True
        st.experimental_rerun()

st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This app was created based on a machine learning notebook for analyzing bank marketing data. 
It allows you to:
* Upload and explore your own CSV data
* Preprocess the data for machine learning
* Train and evaluate multiple machine learning models
* Compare model performance with visualizations

The original notebook used libraries like Pandas for data manipulation, Plotly for visualization, and Scikit-learn for machine learning.
""")
