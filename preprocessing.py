import pandas as pd

class DataPreprocessor:
    def __init__(self, movies_data_path, credits_data_path, reference_year):
        self.movies_data_path = movies_data_path
        self.credits_data_path = credits_data_path
        self.reference_year = reference_year
        self.movies_data = None
        
    def load_data(self):
        # Read data from files
        self.movies_data = pd.read_csv(self.movies_data_path)
        credits_data = pd.read_csv(self.credits_data_path)
        
        # Rename column for consistency
        credits_data = credits_data.rename(columns={'movie_id':'id', 'tittle': 'title'})
        
        # Merge datasets
        self.movies_data = self.movies_data.merge(credits_data, on='id')
        
        # Convert release_date to year
        self.movies_data['release_date'] = pd.to_datetime(self.movies_data['release_date']).dt.year
        
    def prepare_X(self):
        if self.movies_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # List of basic features
        base = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

        # Calculate age of movies
        self.movies_data['age'] = self.reference_year - self.movies_data['release_date']
        
        # Extract numerical features
        features = base + ['age']
        df_num = self.movies_data[features].fillna(0)
        X = df_num.values
        return X
