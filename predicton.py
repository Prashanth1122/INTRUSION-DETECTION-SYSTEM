import pandas as pd
import joblib

def load_model(filepath='model/trained_model.pkl'):
    try:
        model = joblib.load(filepath)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_encoders_scaler():
    try:
        protocol_type_encoder = joblib.load('model/protocol_type_encoder.pkl')
        service_encoder = joblib.load('model/service_encoder.pkl')
        flag_encoder = joblib.load('model/flag_encoder.pkl')
        scaler = joblib.load('model/scaler.pkl')
        print("Encoders and scaler loaded successfully.")
        return protocol_type_encoder, service_encoder, flag_encoder, scaler
    except Exception as e:
        print(f"Failed to load encoders or scaler: {e}")
        return None, None, None, None

def prepare_input_data(input_list, encoders_scaler):
    protocol_type_encoder, service_encoder, flag_encoder, scaler = encoders_scaler
    columns = ['id', 'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
               'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
               'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    input_df = pd.DataFrame([input_list], columns=columns)
    input_df['protocol_type'] = protocol_type_encoder.transform(input_df['protocol_type'].values.ravel())
    input_df['service'] = service_encoder.transform(input_df['service'].values.ravel())
    input_df['flag'] = flag_encoder.transform(input_df['flag'].values.ravel())
    numerical_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df

def predict_input(model, input_df):
    try:
        prediction = model.predict(input_df)
        print("Prediction output:", prediction)  # Debugging statement
        classification = "Anomaly" if prediction[0] == 'anomaly' else "Normal"
        print(f"The network event is classified as: {classification}")
        return classification
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Error in prediction"

if __name__ == "__main__":
    model = load_model()
    encoders_scaler = load_encoders_scaler()
    
    if model and all(encoders_scaler):
        while True:
            user_input = input("Enter the network event details: ")
            if user_input.lower() == 'exit':
                break
            input_list = [float(i) if i.replace('.', '', 1).isdigit() else i.strip("'") for i in user_input.split(',')]
            input_df = prepare_input_data(input_list, encoders_scaler)
            result = predict_input(model, input_df)
    else:
        print("Model or encoders could not be loaded. Exiting the program.")

