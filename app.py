# app.py
from flask import Flask, request, jsonify, render_template
import joblib
from pymongo import MongoClient



app = Flask(__name__)

# Load trained model
model = joblib.load('toxic_comment_model.pkl')
# Connect to MongoDB
client = MongoClient("mongodb+srv://user1:geethaajith52@cluster0.yswwkmf.mongodb.net/?retryWrites=true&w=majority")
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
        print("Received data:", data)  # 👈 Debug print

        comment = data['comment']
        print("Comment:", comment)     # 👈 Debug print

        prediction = model.predict([comment])[0]
        print("Raw prediction:", prediction)  # 👈 Debug print

        result = 'Toxic' if prediction == 1 else 'Safe'
        print("Result:", result)  # 👈 Debug print

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
        print("Error during prediction:", str(e))  # 👈 Debug print
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
