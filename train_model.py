import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time

def load_data(filepath='data/processed_train.csv'):
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data(data):
    try:
        X = data.drop('class', axis=1)
        y = data['class']
        print("Data prepared for training.")
        return X, y
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        start_train_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_train_time
        print(f"Training completed in {training_time:.2f} seconds.")
        
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        print(f"Model accuracy on test set: {accuracy:.2f}")
        print(f"Model precision on test set: {precision:.2f}")
        print(f"Model recall on test set: {recall:.2f}")
        print(f"Model F1-score on test set: {f1:.2f}")
        
        return model, training_time, accuracy, precision, recall, f1
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None, None, None, None

def save_model(model, filename='model/trained_model.pkl'):
    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        X, y = prepare_data(data)
        if X is not None and y is not None:
            model, training_time, accuracy, precision, recall, f1 = train_model(X, y)
            if model is not None:
                save_model(model)
            else:
                print("Model training did not complete successfully.")
        else:
            print("Data preparation did not complete successfully.")
    else:
        print("Data loading did not complete successfully.")
