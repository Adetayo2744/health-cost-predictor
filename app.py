from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        age = int(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        prediction = round(model.predict(data)[0], 2)

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions
        (age, sex, bmi, smoker, children, region, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            age,
            sex,
            bmi,
            smoker,
            children,
            region,
            float(prediction)
        ))

        conn.commit()
        conn.close()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
