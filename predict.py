import joblib
import pandas as pd
from preprocessing import DataPreprocessor
import matplotlib.pyplot as plt

def load_data():
    preprocessor = DataPreprocessor(movies_data_path='tmdb_5000_movies.csv', 
                                    credits_data_path='tmdb_5000_credits.csv', 
                                    reference_year=2024)
    preprocessor.load_data()
    dataset = preprocessor.preprocess()
    return dataset

def predict_random_movies(model, dataset, n=20):
    sample = dataset.sample(n)  # Randomly select n samples
    X_sample = sample.iloc[:, :-1].values
    y_sample_actual = sample.iloc[:, -1].values
    
    y_sample_pred = model.predict(X_sample)
    
    return y_sample_actual, y_sample_pred, sample

if __name__ == "__main__":
    # Load the saved model
    model = joblib.load('randomforest_model.pkl')
    print("Model loaded from randomforest_model.pkl")

    # Load dataset
    dataset = load_data()

    # Select random sample for prediction
    y_sample_actual, y_sample_pred, sample = predict_random_movies(model, dataset)
    
    # Display the comparison
    comparison_df = pd.DataFrame({'Actual Revenue': y_sample_actual, 'Predicted Revenue': y_sample_pred}, index=sample.index)
    
    # Format the numbers with commas
    comparison_df['Actual Revenue'] = comparison_df['Actual Revenue'].apply(lambda x: f"{x:,.0f}")
    comparison_df['Predicted Revenue'] = comparison_df['Predicted Revenue'].apply(lambda x: f"{x:,.0f}")

    print("\nComparison of Actual vs Predicted Revenue for Random 10 Movies:\n")
    print(comparison_df)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_sample_actual, y_sample_pred, color='green', alpha=0.5, label='Predicted vs Actual (Random Sample)')
    plt.plot([min(y_sample_actual), max(y_sample_actual)], [min(y_sample_actual), max(y_sample_actual)], color='red', label='Ideal Fit')  # Diagonal line
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title('Actual vs Predicted Revenue (Random Sample)')
    plt.legend()
    plt.grid(True)
    plt.show()
