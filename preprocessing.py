import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_data():
    # Load the datasets
    train_data = pd.read_csv('data/KDDTrain.csv')
    test_data = pd.read_csv('data/KDDTest.csv')
    
    # Clean up column names by stripping extra quotes
    train_data.columns = train_data.columns.str.strip("'")
    test_data.columns = test_data.columns.str.strip("'")
    
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Handle categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        combined = pd.concat([train_data[col], test_data[col]])
        encoder.fit(combined)
        train_data[col] = encoder.transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])
        encoders[col] = encoder
    
    # Save encoders
    joblib.dump(encoders['protocol_type'], 'model/protocol_type_encoder.pkl')
    joblib.dump(encoders['service'], 'model/service_encoder.pkl')
    joblib.dump(encoders['flag'], 'model/flag_encoder.pkl')
    
    # Scale numerical features
    numerical_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    scaler = StandardScaler()
    train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
    test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
    
    # Save scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_data()
    train_data, test_data = preprocess_data(train_data, test_data)
    train_data.to_csv('data/processed_train.csv', index=False)
    test_data.to_csv('data/processed_test.csv', index=False)
