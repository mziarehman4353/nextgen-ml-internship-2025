
#  Career Guidance Chatbot

A smart and interactive chatbot built using **Machine Learning** and **Streamlit** to help users explore career options based on their natural language questions.

>  Project Submission for **NextGen ML Internship – Week 4**

---

##  Features

-  Predicts career roles from natural language input
-  Returns career advice based on predicted role
-  Clean and responsive UI with toggle for light/dark mode
-  Chat history with confidence scores
-  Favicon + branding via logo
-  Deployed via [Streamlit Community Cloud](https://streamlit.io/cloud)

---

##  Folder Structure

```
career-chatbot/
├── app.py                    # Streamlit app UI
├── train_model.py           # Model training script
├── career.csv               # Career guidance dataset
├── intent_model.pkl         # Trained classification model
├── vectorizer.pkl           # Saved TF-IDF vectorizer
├── logo.png                 # Favicon/logo
└── .streamlit/
    └── config.toml          # Theme config (optional)
```

---

##  Requirements


```
streamlit
pandas
scikit-learn
joblib
```

Install via:

```bash
pip install -r requirements.txt
```

---

##  How to Run Locally

```bash
git clone https://github.com/yourusername/career-chatbot.git
cd career-chatbot
streamlit run app.py
```

---

##  Deployment (Streamlit Cloud)

1. Push this project to GitHub
2. Visit: [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and deploy `app.py`
4. Done! 🎉

---


## Author

**Zia Ul Rehman Zafar**  
ML Intern @ NextGen  
[codewithzia.vercel.app](https://codewithzia.vercel.app)  
 nextgenlearners.official@gmail.com

---

## License

This project is open source and free to use under the [MIT License](LICENSE).
