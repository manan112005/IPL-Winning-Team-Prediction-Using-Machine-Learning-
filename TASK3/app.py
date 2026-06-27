from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

model = load("forest_model.pkl")
model_cols = load("model_columns.pkl")

teams = [
    'Chennai Super Kings','Delhi Daredevils','Kings XI Punjab',
    'Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals',
    'Royal Challengers Bangalore','Sunrisers Hyderabad'
]

venues = [
    'Wankhede Stadium',
    'Eden Gardens',
    'M. Chinnaswamy Stadium',
    'Arun Jaitley Stadium',
    'MA Chidambaram Stadium',
    'Rajiv Gandhi International Stadium',
    'Narendra Modi Stadium'
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x / 10))

@app.route("/", methods=["GET", "POST"])
def index():
    batting_prob = bowling_prob = None
    batting_team = bowling_team = venue = None

    if request.method == "POST":
        batting_team = request.form.get("batting_team")
        bowling_team = request.form.get("bowling_team")
        venue = request.form.get("venue")

        input_data = {
            'runs': int(request.form.get("runs")),
            'wickets': int(request.form.get("wickets")),
            'overs': float(request.form.get("overs")),
            'runs_last_5': int(request.form.get("runs_last_5")),
            'wickets_last_5': int(request.form.get("wickets_last_5"))
        }

        # One-hot encode teams
        for col in model_cols:
            if col.startswith("batting_team_"):
                input_data[col] = 1 if col == f"batting_team_{batting_team}" else 0
            elif col.startswith("bowling_team_"):
                input_data[col] = 1 if col == f"bowling_team_{bowling_team}" else 0

        df = pd.DataFrame([input_data])[model_cols]
        predicted_score = model.predict(df)[0]

        # Probability calculation
        diff = predicted_score - 160  # average IPL chase baseline
        batting_prob = round(sigmoid(diff) * 100, 2)
        bowling_prob = round(100 - batting_prob, 2)

    return render_template(
        "index.html",
        teams=teams,
        venues=venues,
        batting_prob=batting_prob,
        bowling_prob=bowling_prob,
        batting_team=batting_team,
        bowling_team=bowling_team,
        venue=venue,
        form=request.form
    )

if __name__ == "__main__":
    app.run(debug=True)
