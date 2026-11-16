import pickle
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('student-grade-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()
    X = dv.transform([student])
    y_pred = float(model.predict(X))
    # Clip prediction cleanly within valid 0-20 grade range
    y_pred_clipped = max(0.0, min(20.0, y_pred))
    return jsonify({
        'predicted_g3_grade': round(y_pred_clipped, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
# predict.py (Flask Web Service)