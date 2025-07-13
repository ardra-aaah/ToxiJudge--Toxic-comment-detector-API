# ToxiJudge
ToxiJudge is a simple web app that detects whether a user’s comment is **toxic** or **safe**. I made use of a machine learning model trained on the Jigsaw Toxic Comment dataset and deployed the application with Flask on Render. Also, a MongoDB Atlas database was used to store predictions.  

 ## Live Demo: [https://toxijudge-toxic-comment-detector-api.onrender.com](https://toxijudge-toxic-comment-detector-api.onrender.com)  
####  Note: The app was deployed and live, but the MongoDB connection was disabled later for security.
---

##  Features
-  **ML-powered predictions** – Logistic Regression with TF-IDF vectorisation.  
-  **Web interface** – clean UI to enter and test comments.  
-  **MongoDB integration** – saves predictions for future analytics.  
-  **Deployed on Render** – accessible from anywhere.  

---

##  Tech Stack
- Python (flask)  
- scikit-learn  
- MongoDB Atlas  
- Render (deployment)  
- HTML/CSS for frontend  

---

##  Project Structure
 ─ app.py # Flask API
 ─ ModelTrain.py # Script to train and save ML model
 ─ toxic_comment_model.pkl # Saved ML model
 ─ templates/
    ── index.html # Frontend page
    ── requirements.txt # Python dependencies
    ── Procfile # For deployment on Render


---

##  Running Locally
1. **To clone the repo**  
   ```bash
   git clone https://github.com/ardra-aaah/ToxiJudge--Toxic-comment-detector-API.git
   cd ToxiJudge--Toxic-comment-detector-API
## To install dependencies, run: 
 ```bash
 pip install -r requirements.txt
```
## What I Learned
  - Major takeaway for me was fiddling around with deploying Flask apps with Gunicorn on Render. 
  - Setting up MongoDB Atlas and connecting via Python.
  - Training and saving scikit-learn models for production.

## Future enhancements
  - Add user authentication to the web app.
  - Visualise stored predictions in a dynamic dashboard.
  - Extend ML model with more advanced architectures (like BERT).
## Author

  Ardra T J
  
