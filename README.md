# ANN_Classification

## Project Overview
This project focuses on predicting customer churn using an Artificial Neural Network (ANN). Customer churn refers to the phenomenon where customers stop doing business with a company. The goal is to identify customers who are likely to churn so that proactive measures can be taken to retain them.

## Key Concepts
- **Customer Churn Prediction**: Using machine learning to predict whether a customer will leave a service.
- **Artificial Neural Networks (ANN)**: A deep learning model inspired by the human brain, used for classification tasks.
- **Data Preprocessing**: Encoding categorical variables, scaling numerical features, and preparing data for model input.

## Process
1. **Data Loading**: The dataset `Churn_Modelling.csv` is loaded and preprocessed.
2. **Feature Encoding**: Categorical features like `Gender` and `Geography` are encoded using `LabelEncoder` and one-hot encoding.
3. **Feature Scaling**: Numerical features are scaled using `StandardScaler`.
4. **Model Training**: An ANN model is trained to classify whether a customer will churn.
5. **Prediction**: The trained model is used to predict churn probabilities for new input data.
6. **Deployment**: A Streamlit app is created to provide an interactive interface for predictions.

## Tools and Libraries
- **TensorFlow**: For building and training the ANN model.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For preprocessing tasks like encoding and scaling.
- **Streamlit**: For building the web application.
- **Matplotlib**: For data visualization.

## How to Use
1. Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Use the app interface to input customer details and get churn predictions.

## Files in the Repository
- `app.py`: The Streamlit app for customer churn prediction.
- `Churn_Modelling.csv`: The dataset used for training and testing.
- `model.keras`: The trained ANN model.
- `label_encoder_gender.pkl`, `label_encoder_geography.pkl`: Pickle files for encoding categorical variables.
- `scaler.pkl`: Pickle file for scaling numerical features.
- `prediction.ipynb`: A Jupyter notebook for testing the prediction pipeline.
- `requirements.txt`: List of dependencies required for the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.