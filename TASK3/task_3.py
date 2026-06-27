import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from joblib import dump

# Load data
df = pd.read_csv("TASK3/data.csv")

# Drop irrelevant columns
drop_cols = ['mid','date','venue','batsman','bowler','striker','non-striker']
df.drop(columns=drop_cols, inplace=True)

# Keep consistent teams
teams = [
    'Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals',
    'Mumbai Indians','Kings XI Punjab','Royal Challengers Bangalore',
    'Delhi Daredevils','Sunrisers Hyderabad'
]

df = df[
    (df['batting_team'].isin(teams)) &
    (df['bowling_team'].isin(teams))
]

# Remove first 5 overs
df = df[df['overs'] >= 5.0]

# One-hot encoding
df = pd.get_dummies(df, columns=['batting_team','bowling_team'])

# Split features and target
X = df.drop('total', axis=1)
y = df['total']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Save model and columns
dump(model, "forest_model.pkl")
dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Model trained and saved successfully")
