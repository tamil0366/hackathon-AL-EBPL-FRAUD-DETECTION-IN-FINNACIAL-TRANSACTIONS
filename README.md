# hackathon-AL-EBPL-FRAUD-DETECTION-IN-FINNACIAL-TRANSACTIONS

💻 Fraud Detection GUI Application (Python + Tkinter)

✅ Main Features:

1. 🔐 Secure Login with 2FA

Password-based login

Random 4-digit 2FA verification



2. 📊 Fraud Prediction (Single Entry)

Input: amount, transaction_type, location_code, time_hour

Model: RandomForestClassifier (trained on dummy data)

Output: Legitimate or Fraudulent

Voice feedback using pyttsx3

Color feedback (green/red background)

SHAP explanations of prediction (feature importance)



3. 📁 Upload & Analyze CSV File

Bulk fraud prediction

Input CSV must have amount, transaction_type, location_code, time_hour

Saves results to fraud_report.csv

Stores to SQLite database (transactions.db)

💻 Fraud Detection GUI Application (Python + Tkinter)

✅ Main Features:

    🔐 Secure Login with 2FA


Password-based login

Random 4-digit 2FA verification



    📊 Fraud Prediction (Single Entry)


Input: amount, transaction_type, location_code, time_hour

Model: RandomForestClassifier (trained on dummy data)

Output: Legitimate or Fraudulent

Voice feedback using pyttsx3

Color feedback (green/red background)

SHAP explanations of prediction (feature importance)



    📁 Upload & Analyze CSV File


Bulk fraud prediction

Input CSV must have amount, transaction_type, location_code, time_hour

Saves results to fraud_report.csv

Stores to SQLite database (transactions.db)



    📈 Transaction Dashboard


Boxplot comparing fraudulent and legitimate transaction amounts

Auto-refreshes after every prediction



    🧠 Model Metrics

💻 Fraud Detection GUI Application (Python + Tkinter)

✅ Main Features:

    🔐 Secure Login with 2FA


Password-based login

Random 4-digit 2FA verification



    📊 Fraud Prediction (Single Entry)


Input: amount, transaction_type, location_code, time_hour

Model: RandomForestClassifier (trained on dummy data)

Output: Legitimate or Fraudulent

Voice feedback using pyttsx3

Color feedback (green/red background)

SHAP explanations of prediction (feature importance)



    📁 Upload & Analyze CSV File


Bulk fraud prediction

Input CSV must have amount, transaction_type, location_code, time_hour

Saves results to fraud_report.csv

Stores to SQLite database (transactions.db)



    📈 Transaction Dashboard


Boxplot comparing fraudulent and legitimate transaction amounts

Auto-refreshes after every prediction



    🧠 Model Metrics


Shows model accuracy, precision, recall in a message box





---

📂 File Outputs:

fraud_report.csv: Stores each prediction

transactions.db: SQLite database for historical transactions



---

🖼 GUI Libraries Used:

tkinter — GUI building

ttk — Modern widgets

matplotlib — Plotting dashboard

pyttsx3 — Voice alerts

shap — Explainability



---

🧠 ML Model:

RandomForestClassifier(n_estimators=100, random_state=42)

Trained on:

X_train = [[amount, transaction_type, location_code, time_hour], ...]
y_train = [0, 1, 0, 1, ...]


---

🚀 How to Run:

    Install dependencies:


pip install numpy pandas scikit-learn matplotlib pyttsx3 shap


    Save code to fraud_gui.py and run:


python fraud_gui.py


    Login with:


Password: secure123

2FA: A random 4-digit code shown in a popup

Shows model accuracy, precision, recall in a message box





---

📂 File Outputs:

fraud_report.csv: Stores each prediction

transactions.db: SQLite database for historical transactions



---

🖼 GUI Libraries Used:

tkinter — GUI building

ttk — Modern widgets

matplotlib — Plotting dashboard

pyttsx3 — Voice alerts

shap — Explainability



---

🧠 ML Model:

RandomForestClassifier(n_estimators=100, random_state=42)

Trained on:

X_train = [[amount, transaction_type, location_code, time_hour], ...]
y_train = [0, 1, 0, 1, ...]


---

🚀 How to Run:

    Install dependencies:


pip install numpy pandas scikit-learn matplotlib pyttsx3 shap


    Save code to fraud_gui.py and run:


python fraud_gui.py


    Login with:


Password: secure123

2FA: A random 4-digit code shown in a popup


4. 📈 Transaction Dashboard

Boxplot comparing fraudulent and legitimate transaction amounts

Auto-refreshes after every prediction



5. 🧠 Model Metrics

Shows model accuracy, precision, recall in a message box





---

📂 File Outputs:

fraud_report.csv: Stores each prediction

transactions.db: SQLite database for historical transactions



---

🖼 GUI Libraries Used:

tkinter — GUI building

ttk — Modern widgets

matplotlib — Plotting dashboard

pyttsx3 — Voice alerts

shap — Explainability



---

🧠 ML Model:

RandomForestClassifier(n_estimators=100, random_state=42)

Trained on:

X_train = [[amount, transaction_type, location_code, time_hour], ...]
y_train = [0, 1, 0, 1, ...]


---

🚀 How to Run:

1. Install dependencies:

pip install numpy pandas scikit-learn matplotlib pyttsx3 shap


2. Save code to fraud_gui.py and run:

python fraud_gui.py


3. Login with:

Password: secure123

2FA: A random 4-digit code shown in a popup
