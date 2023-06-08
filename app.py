from flask import Flask, request, jsonify
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

app = Flask(__name__)

# Load the XGBoost model
xgb_model = joblib.load('xgb_model.h5')

# Load the scaler model
scaler = joblib.load('scaler.h5')

# Load the label encoder model
label_encoder = LabelEncoder()
label_encoder = joblib.load('label_encoder.pkl')
#y = label_encoder.fit_transform(["Diabetes", "No diabetes"])
#label_encoder.classes_ = joblib.load('label_encoder.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the request
        user_inputs = request.json['user_inputs']

        # Prepare user inputs for prediction
        user_inputs = [float(value) for value in user_inputs]

        # Perform any required preprocessing steps on user inputs
        user_inputs_scaled = scaler.transform([user_inputs])

        # Predict diabetes for user inputs
        prediction = xgb_model.predict(user_inputs_scaled)
        predicted_class = label_encoder.inverse_transform(prediction)

        return jsonify({'predicted_class': predicted_class.tolist()})

    except FileNotFoundError:
        return jsonify({'error': 'File not found.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
 
        
