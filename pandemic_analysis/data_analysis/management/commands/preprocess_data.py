from django.core.management.base import BaseCommand
import pandas as pd
from data_analysis.models import PandemicData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data():
    print("Loading data from database...")
    
    # Load data from the database
    data = pd.DataFrame(list(PandemicData.objects.all().values()))
    
    if data.empty:
        print("No data found in the database.")
        return
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values(by='date')
    
    # Handle missing values (fill NaNs with median values)
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    # Select relevant features
    features = ['newCases', 'intenciveCareUnit', 'deaths']
    target = ['newCases', 'intenciveCareUnit', 'deaths']  # Predict the same columns
    
    # Split dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    test_data[features] = scaler.transform(test_data[features])
    
    print("Preprocessing complete. Train size:", len(train_data), "Test size:", len(test_data))
    return train_data, test_data, scaler

class Command(BaseCommand):
    help = "Preprocess pandemic data for machine learning models."

    def handle(self, *args, **kwargs):
        preprocess_data()
        self.stdout.write(self.style.SUCCESS('Data preprocessing complete!'))
