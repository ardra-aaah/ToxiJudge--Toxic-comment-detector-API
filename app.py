# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('toxic_comment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/')
# def index():
#     return "🚀 Toxic Comment Detector API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        comment = data['comment']
        prediction = model.predict([comment])[0]
        result = 'Toxic' if prediction == 1 else 'Safe'
        return jsonify({
            'input_comment': comment,
            'prediction': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
