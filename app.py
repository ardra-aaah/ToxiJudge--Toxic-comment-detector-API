# app.py
from flask import Flask, request, jsonify, render_template
import joblib
from pymongo import MongoClient



app = Flask(__name__)

# Load trained model
model = joblib.load('toxic_comment_model.pkl')
# Connect to MongoDB
client = MongoClient("mongodb+srv://user1:geethaajith52@cluster0.yswwkmf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['toxic_comment_db']
collection = db['predictions']

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

        # Save to MongoDB
        collection.insert_one({
          'comment': comment,
          'prediction': result
    })
        return jsonify({
          'input_comment': comment,
          'prediction': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
