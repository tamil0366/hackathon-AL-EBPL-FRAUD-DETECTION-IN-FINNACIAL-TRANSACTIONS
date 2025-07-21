import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyttsx3
import random
import sqlite3
import shap

# --- Constants and Globals ---
CORRECT_PASSWORD = "secure123"
attempts = 0
MAX_ATTEMPTS = 3

# --- Model Training ---
X_train = np.array([
    [100, 0, 1, 14], [2000, 1, 2, 23], [50, 0, 1, 10],
    [5000, 1, 3, 2], [20, 0, 1, 15], [3000, 1, 2, 22],
    [10, 0, 1, 9], [7000, 1, 3, 1], [150, 0, 1, 12], [4000, 1, 2, 0]
])
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
report = classification_report(y_train, y_pred, output_dict=True)
transaction_history = {'legitimate': 0, 'fraudulent': 0}

# --- Utility Functions ---
def predict_fraud(amount, transaction_type, location_code, time_hour):
    return model.predict([[amount, transaction_type, location_code, time_hour]])[0]

def show_model_metrics():
    messagebox.showinfo("Model Performance", f"Accuracy: {accuracy:.2f}\n"
                                            f"Precision (Fraud): {report['1']['precision']:.2f}\n"
                                            f"Recall (Fraud): {report['1']['recall']:.2f}")

def explain_prediction(amount, transaction_type, location_code, time_hour):
    try:
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        input_array = np.array([amount, transaction_type, location_code, time_hour]).reshape(1, -1)
        shap_values = explainer.shap_values(input_array)

        messagebox.showinfo("Explanation", f"Feature contributions:\n"
                                            f"Amount: {shap_values[0][0]:.2f}\n"
                                            f"Type: {shap_values[0][1]:.2f}\n"
                                            f"Location: {shap_values[0][2]:.2f}\n"
                                            f"Time: {shap_values[0][3]:.2f}")
    except Exception as e:
        messagebox.showerror("SHAP Error", str(e))

def create_dashboard(window):
    for widget in window.winfo_children():
        if isinstance(widget, tk.Canvas):
            widget.destroy()

    fig, ax = plt.subplots(figsize=(4, 4))

    legit = transaction_history.get('legitimate', 0)
    fraud = transaction_history.get('fraudulent', 0)

    if legit + fraud > 0:
        # Simulated data for boxplot purposes
        legit_data = np.random.normal(loc=100, scale=30, size=int(legit)) if legit > 0 else []
        fraud_data = np.random.normal(loc=300, scale=100, size=int(fraud)) if fraud > 0 else []


        data_to_plot = [legit_data, fraud_data]
        ax.boxplot(data_to_plot, labels=["Legitimate", "Fraudulent"], patch_artist=True,
                   boxprops=dict(facecolor='#ADD8E6'),
                   medianprops=dict(color='red'))

        ax.set_title("Transaction Amount Distribution")
        ax.set_ylabel("Amount")
    else:
        ax.text(0.5, 0.5, "No transactions yet", fontsize=12, ha='center', va='center')
        ax.axis('off')

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)
    plt.close(fig)

def update_history(result):
    key = 'legitimate' if result == 0 else 'fraudulent'
    transaction_history[key] += 1

def announce_result(result):
    try:
        engine = pyttsx3.init()
        text = "Fraudulent transaction detected!" if result else "Transaction is legitimate."
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        messagebox.showwarning("Voice Error", str(e))

def save_to_db(amount, transaction_type, location_code, time_hour, result):
    try:
        with sqlite3.connect('transactions.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS transactions
                         (amount REAL, type INTEGER, location INTEGER, time REAL, fraud INTEGER)''')
            c.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
                      (amount, transaction_type, location_code, time_hour, result))
            conn.commit()
    except Exception as e:
        messagebox.showerror("Database Error", str(e))

def upload_transactions(window):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['amount', 'transaction_type', 'location_code', 'time_hour']):
                messagebox.showerror("CSV Error", "CSV must contain: amount, transaction_type, location_code, time_hour")
                return
            df['fraud_prediction'] = model.predict(df[['amount', 'transaction_type', 'location_code', 'time_hour']])
            df.to_csv('fraud_report.csv', index=False)
            for _, row in df.iterrows():
                save_to_db(row['amount'], row['transaction_type'], row['location_code'], row['time_hour'], row['fraud_prediction'])

            fraud_count = sum(df['fraud_prediction'])
            legit_count = len(df) - fraud_count
            transaction_history['fraudulent'] += fraud_count
            transaction_history['legitimate'] += legit_count

            messagebox.showinfo("Batch Prediction", f"Processed {len(df)} transactions.\nFraudulent: {fraud_count}")
            create_dashboard(window)
        except Exception as e:
            messagebox.showerror("Error", str(e))

def save_report(amount, transaction_type, location_code, time_hour, result):
    df = pd.DataFrame([{
        'amount': amount,
        'transaction_type': transaction_type,
        'location_code': location_code,
        'time_hour': time_hour,
        'fraud': result
    }])
    df.to_csv('fraud_report.csv', mode='a', index=False, header=not pd.io.common.file_exists('fraud_report.csv'))
    messagebox.showinfo("Report Saved", "Results saved to fraud_report.csv")

def check_2fa():
    code = random.randint(1000, 9999)
    messagebox.showinfo("2FA", f"Your code is: {code}")
    return simpledialog.askstring("2FA", "Enter the 4-digit code:", parent=root) == str(code)

def check_password():
    global attempts
    if password_entry.get() == CORRECT_PASSWORD:
        if check_2fa():
            messagebox.showinfo("Access Granted", "Welcome!")
            attempts = 0
            root.withdraw()
            open_fraud_detection_window()
        else:
            messagebox.showerror("2FA Failed", "Incorrect 2FA code!")
    else:
        attempts += 1
        messagebox.showwarning("Access Denied", f"Wrong password! Attempt {attempts} of {MAX_ATTEMPTS}.")
        if attempts >= MAX_ATTEMPTS:
            messagebox.showerror("ALERT 🚨", "Multiple failed attempts detected!")

def open_fraud_detection_window():
    global amount_entry, type_entry, location_entry, time_entry

    fraud_window = tk.Toplevel()
    fraud_window.title("Fraud Detection")
    fraud_window.geometry("400x500")
    fraud_window.config(bg="#FFFFFF")

    def on_predict():
        progress = ttk.Progressbar(fraud_window, mode='indeterminate')
        progress.pack(pady=5)
        progress.start()
        fraud_window.update()

        try:
            amount = float(amount_entry.get())
            transaction_type = int(type_entry.get())
            location_code = int(location_entry.get())
            time_hour = float(time_entry.get())

            if transaction_type not in [0, 1] or location_code not in [1, 2, 3] or not (0 <= time_hour <= 23):
                messagebox.showerror("Input Error", "Invalid input range.")
                return

            result = predict_fraud(amount, transaction_type, location_code, time_hour)
            update_history(result)
            announce_result(result)
            save_report(amount, transaction_type, location_code, time_hour, result)
            save_to_db(amount, transaction_type, location_code, time_hour, result)
            explain_prediction(amount, transaction_type, location_code, time_hour)
            create_dashboard(fraud_window)

            if result:
                messagebox.showwarning("Fraud Result", "This transaction is likely FRAUDULENT!")
                fraud_window.config(bg="#FF6347")
            else:
                messagebox.showinfo("Fraud Result", "This transaction appears LEGITIMATE.")
                fraud_window.config(bg="#98FB98")

            fraud_window.after(3000, lambda: fraud_window.config(bg="#FFFFFF"))

        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numeric values.")
        finally:
            progress.stop()
            progress.destroy()

    tk.Label(fraud_window, text=f"Model Accuracy: {accuracy:.2f}", font=("Arial", 12)).pack(pady=10)

    tk.Label(fraud_window, text="Transaction Amount:", font=("Arial", 12)).pack(pady=5)
    amount_entry = tk.Entry(fraud_window, font=("Arial", 12), relief="solid", bd=2)
    amount_entry.pack(pady=5)

    tk.Label(fraud_window, text="Transaction Type (0=debit, 1=credit):", font=("Arial", 12)).pack(pady=5)
    type_entry = tk.Entry(fraud_window, font=("Arial", 12), relief="solid", bd=2)
    type_entry.pack(pady=5)

    tk.Label(fraud_window, text="Location Code (1-3):", font=("Arial", 12)).pack(pady=5)
    location_entry = tk.Entry(fraud_window, font=("Arial", 12), relief="solid", bd=2)
    location_entry.pack(pady=5)

    tk.Label(fraud_window, text="Time (Hour 0-23):", font=("Arial", 12)).pack(pady=5)
    time_entry = tk.Entry(fraud_window, font=("Arial", 12), relief="solid", bd=2)
    time_entry.pack(pady=5)

    predict_btn = ttk.Button(fraud_window, text="Check Fraud", command=on_predict)
    predict_btn.pack(pady=10)

    ttk.Button(fraud_window, text="Upload CSV", command=lambda: upload_transactions(fraud_window)).pack(pady=5)
    ttk.Button(fraud_window, text="Show Model Metrics", command=show_model_metrics).pack(pady=5)

    create_dashboard(fraud_window)
    fraud_window.after(60000, lambda: fraud_window.destroy())

# --- Main Login GUI ---
root = tk.Tk()
root.title("Fraud Protection - Secure Login")
root.geometry("300x180")
root.config(bg="#FFF0F5")

tk.Label(root, text="Enter your password:", font=("Arial", 12), bg="#FFF0F5").pack(pady=10)
password_entry = tk.Entry(root, show="*", font=("Arial", 12), relief="solid", bd=2)
password_entry.pack(pady=5)

submit_btn = ttk.Button(root, text="Submit", command=check_password)
submit_btn.pack(pady=15)

root.mainloop()
