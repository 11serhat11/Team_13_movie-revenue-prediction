import joblib
import pandas as pd
from preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_data():
    preprocessor = DataPreprocessor(movies_data_path='tmdb_5000_movies.csv', 
                                    credits_data_path='tmdb_5000_credits.csv', 
                                    reference_year=2024)
    preprocessor.load_data()
    dataset = preprocessor.preprocess()
    return dataset

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        mse, mae, rmse = evaluate_model(model, X_test, y_test)
        results[name] = {"MSE": mse, "MAE": mae, "RMSE": rmse}
    
    return results

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

    # Split dataset
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Compare models
    results = compare_models(X_train, y_train, X_test, y_test)
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")


    # Select random sample for prediction
    y_sample_actual, y_sample_pred, sample = predict_random_movies(model, dataset)
    
    # Display the comparison
    comparison_df = pd.DataFrame({'Actual Revenue': y_sample_actual, 'Predicted Revenue': y_sample_pred}, index=sample.index)
    
    # Format the numbers with commas
    comparison_df['Actual Revenue'] = comparison_df['Actual Revenue'].apply(lambda x: f"{x:,.0f}")
    comparison_df['Predicted Revenue'] = comparison_df['Predicted Revenue'].apply(lambda x: f"{x:,.0f}")

    print("\nComparison of Actual vs Predicted Revenue for Random 10 Movies:\n")
    print(comparison_df)

    # Plot predicted vs actual revenue
    plt.figure(figsize=(8, 6))
    plt.scatter(y_sample_actual, y_sample_pred, color='green', alpha=0.5, label='Predicted vs Actual (Random Sample)')
    plt.plot([min(y_sample_actual), max(y_sample_actual)], [min(y_sample_actual), max(y_sample_actual)], color='red', label='Ideal Fit')  # Diagonal line
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title('Actual vs Predicted Revenue (Random Sample)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate residuals
    residuals = y_sample_actual - y_sample_pred
    
    # Plot histogram of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Residuals (Actual - Predicted Revenue)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(True)
    plt.show()

    # Plot feature importances
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_names = dataset.columns[:-1]

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances, color='purple')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances (Random Forest)')
        plt.grid(True)
        plt.show()
