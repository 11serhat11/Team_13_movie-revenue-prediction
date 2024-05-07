from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#import tuning_interpretation as ti #This will be the hyperparameter tuning part of the project (tuning_interpretation.py)
#NOT NECESSARY FOR NOW
import preprocessing as pp #This will be the pre-processing part of the project (pre_processing.py)

def load_data():
    
    preprocessor = pp.DataPreprocessor(movies_data_path='tmdb_5000_movies.csv', 
                                       credits_data_path='tmdb_5000_credits.csv', 
                                       reference_year=2024)
    preprocessor.load_data()
    dataset = preprocessor.preprocess()
    return dataset


def split_data(dataset):
   
    X = dataset.iloc[:, :-1].values #all values except last column (budget - popularity - runtime - vote_average - vote_count - age)
    y = dataset.iloc[:, -1].values  #values of last column which is dependent attribute(revenue)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
   
    y_pred = model.predict(X_test) #Testing part for evaulation (20% of the dataset is seperated for testing)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    # Load dataset from pre_processing.py
    dataset = load_data()

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Train the model (fitting)
    model = train_model(X_train, y_train)

    # Evaluating the model with mse
    mse = evaluate_model(model, X_test, y_test)
    print("Mean Squared Error:", mse)

    y_pred = model.predict(X_test)
    
    # Plot predicted vs actual revenue
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title('Actual vs Predicted Revenue')
    plt.show()