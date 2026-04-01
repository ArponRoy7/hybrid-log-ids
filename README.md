# 🔐 Hybrid Log-based Intrusion Detection System (IDS)

A real-time **Log-based Intrusion Detection System** using a hybrid machine learning approach combining:

* 🔹 Logistic Regression (Supervised)
* 🔹 Isolation Forest (Unsupervised)
* 🔹 TF-IDF Feature Engineering

The system detects anomalous log patterns and simulates **real-time brute-force attack detection** using a Streamlit web interface.

---

## 🚀 Features

* Real-time intrusion detection using logs
* Hybrid ML model (LR + Isolation Forest)
* TF-IDF based log representation
* Zero-day attack evaluation support
* Interactive Streamlit dashboard
* Lightweight and deployable system

---

## 📂 Project Structure

```
hybrid-log-ids/
│
├── app.py                  # Streamlit app (entry point)
├── requirements.txt
├── models/                # Trained models (must exist)
│   ├── lr.pkl
│   ├── iforest.pkl
│   ├── tfidf.pkl
│   ├── if_min.npy
│   ├── if_max.npy
│
├── src/
│   ├── predict.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   ├── zero_day.py
```

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/hybrid-log-ids.git
cd hybrid-log-ids
```

---

### 2️⃣ Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🧠 How It Works

1. User input (login attempts) generates log sequences
2. Logs are converted into TF-IDF feature vectors
3. Hybrid model computes anomaly score:

   * Logistic Regression → probability
   * Isolation Forest → anomaly score
4. Scores are combined to detect attacks
5. Streamlit UI displays real-time results

---

## 🎯 Demo Usage

* ✅ Correct login → Normal behavior
* ⚠️ Few wrong attempts → Suspicious
* 🚨 Multiple failures → Attack detected

---

## 🔧 Training the Model (Optional)

If you want to retrain:

```
python -m src.train
```

---

## 🧪 Zero-Day Evaluation (Optional)

```
python -m src.zero_day
```

---

## 🌐 Deployment

This project can be deployed on:

* Streamlit Cloud
* Local server
* Raspberry Pi

---

## 📌 Requirements

* Python 3.8+
* Streamlit
* Scikit-learn
* NumPy / SciPy

---

## 👨‍💻 Author

**Arpon Roy**
M.Tech CSE, IIT Bhubaneswar

---

## ⭐ Notes

* Ensure `models/` folder is present before running
* No raw dataset required for inference
* Designed for research + demonstration purposes

---

## 📜 License

This project is for academic and research purposes.
