import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import Label, Entry, Button, messagebox, filedialog
from PIL import Image, ImageTk  # Import PIL

# Load the dataset (same as in your code)
dataset = pd.read_csv('Final_data.csv.csv')

# Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)
X = dataset[['Age', 'Salary', 'previous_purchase']].values  # Include 'previous_purchase' as an input feature
Y = dataset['previous_purchase'].values  # Assuming 'Purchased' is the target variable

# Splitting Dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
joblib.dump(model, 'trained_model.joblib')
print("Model Loaded")
# Load the trained model
model = joblib.load('trained_model.joblib')
# Create a function to predict based on user input
def predict():
    try:
        age = int(age_entry.get())
        salary = int(salary_entry.get())
        previous_purchase = int(previous_purchase_entry.get())  # New input field for previous_purchase
        new_cust = [[age, salary, previous_purchase]]  # Include previous_purchase in the input data
        result = model.predict(sc.transform(new_cust))

        if result == 1:
            messagebox.showinfo("Prediction", "Customer will Buy")
        else:
            messagebox.showinfo("Prediction", "Customer won't Buy")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid age, salary, and previous_purchase.")

# Create a function to handle file upload and prediction
def upload_and_predict():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            # Load the user's CSV file
            user_data = pd.read_csv(file_path)

            # Predict using the loaded data
            user_data['Prediction'] = model.predict(sc.transform(user_data[['Age', 'Salary', 'previous_purchase']].values))
            
            # Save the predictions to a new CSV file
            save_path = "predictions_output.csv"
            user_data.to_csv(save_path, index=False)

            messagebox.showinfo("File Saved", f"Predictions saved to {save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main application window
app = tk.Tk()
app.title("Customer Purchase Prediction")

# Load the background image
background_image = Image.open("sales_forcasting.jpg")
background_image = background_image.resize((400, 250), Image.ANTIALIAS)  # Resize the image to match GUI size
background_photo = ImageTk.PhotoImage(background_image)

# Set the background image
background_label = tk.Label(app, image=background_photo)
background_label.image = background_photo
background_label.place(relwidth=1, relheight=1)

# Create labels, entry fields, and a predict button
age_label = Label(app, text="Enter Customer Age:")
age_label.pack()
age_entry = Entry(app)
age_entry.pack()

salary_label = Label(app, text="Enter Customer Budget:")
salary_label.pack()
salary_entry = Entry(app)
salary_entry.pack()

previous_purchase_label = Label(app, text="Enter Previous Purchase (0 or 1):")
previous_purchase_label.pack()
previous_purchase_entry = Entry(app)
previous_purchase_entry.pack()

predict_button = Button(app, text="Predict", command=predict)
predict_button.pack()

# Create an upload button
upload_button = Button(app, text="Upload CSV and Predict", command=upload_and_predict)
upload_button.pack()

# Set the GUI size according to the image size
app.geometry(f"{background_image.width}x{background_image.height}")

# Start the GUI main loop
app.mainloop()
