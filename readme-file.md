# Bank Marketing ML Analysis Streamlit App

This Streamlit application provides an interactive interface for analyzing banking data using various machine learning models. It's based on the BU7331 class notebook on ML and GenAI.

## Features

- **Data Upload**: Upload your own CSV file for analysis
- **Data Exploration**: View basic statistics and visualize your data
- **Data Preprocessing**: One-hot encode categorical variables automatically
- **Machine Learning**: Train multiple ML models with a single click
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (optional)
  - Neural Network (MLP) (optional)
- **Model Evaluation**: Compare model performance with accuracy metrics and confusion matrices
- **Feature Importance**: Visualize which features have the most impact on predictions
- **Generative AI Integration**: Example of how to connect to OpenAI API (requires your own API key)

## Getting Started

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the App

Run the following command in your terminal:
```
streamlit run app.py
```

The app will open in your default web browser.

### Using Sample Data

If you don't have your own dataset, you can use the sample bank marketing data provided in the app.

### Expected Data Format

For optimal results, your data should:
- Be in CSV format
- Include both categorical and numerical features
- Have a target variable (preferably binary for classification)

## Workflow

1. Upload your CSV file
2. Explore the data with automatic visualizations
3. Select your target variable
4. Apply preprocessing to handle categorical variables
5. Choose which ML models to train
6. Compare model performance
7. Examine feature importance

## Technical Details

This app uses:
- **Streamlit**: For the web interface
- **Pandas**: For data manipulation
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning models
- **NumPy**: For numerical operations

## About the Original Notebook

This Streamlit app is based on the BU7331 class notebook focusing on Machine Learning and Generative AI concepts. The original notebook demonstrated:

- Basic data exploration
- Data preprocessing
- Training multiple ML models
- Model evaluation with confusion matrices
- Feature importance analysis
- Optional integration with OpenAI API
