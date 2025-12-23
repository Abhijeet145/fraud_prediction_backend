from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow localhost:3000 React to call this API

# ---------- Load model and encoders ----------
model = load_model('lstm_prefix_model.h5')
event_encoder = joblib.load('event_encoder.pkl')
status_encoder = joblib.load('status_encoder.pkl')
max_len = joblib.load('max_len.pkl')

# Positive class index (assuming 'positive'/'negative' mapping used earlier)
# If you encoded 'positive'/'negative' labels, find index of 'positive'
classes = list(status_encoder.classes_)
if 'positive' in classes:
    positive_index = classes.index('positive')
else:
    # fallback: highest-index class
    positive_index = len(classes) - 1


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prefix_events = data.get('prefix', [])  # list of event names, e.g. ["SelfieLiveness", "Vendor1Request"]

    if not prefix_events:
        return jsonify({'error': 'prefix is required'}), 400

    # ---------- Encode events ----------
    # Unknown events are ignored or mapped to 0; here we ignore unknowns and rely on padding
    encoded = []
    for ev in prefix_events:
        if ev in event_encoder.classes_:
            idx = int(np.where(event_encoder.classes_ == ev)[0][0])
            encoded.append(idx)
        # else: skip unknown events
            

    if not encoded:
        # if nothing could be encoded, return low risk
        return jsonify({'risk_score': 0.0})

    # pad to max_len (same as during training)
    padded = pad_sequences([encoded], maxlen=max_len, padding='post', value=0)

    # ---------- Predict ----------
    proba = model.predict(padded, verbose=0)[0]  # array of class probabilities
    fraud_prob = float(proba[positive_index])  # probability of 'positive' class
    risk_score = round(fraud_prob * 100, 2)

    # Optional recommendation
    if risk_score <= 50:
        recommendation = 'Allow to Continue'
    elif risk_score <= 85:
        recommendation = 'Trigger Challenge'
    else:
        recommendation = 'Block & Alert'

    return jsonify({
        'risk_score': risk_score,
        'fraud_probability': fraud_prob,
        'recommendation': recommendation,
        'classes': classes
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
