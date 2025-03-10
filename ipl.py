import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Load datasets
matches = pd.read_csv("matches.csv")
balls = pd.read_csv("deliveries.csv")

# Preprocess match dataset
df = matches[['id', 'season', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].dropna()

# Encode categorical variables
le = LabelEncoder()
for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'winner']:
    df[col] = le.fit_transform(df[col])

# Split data
X = df[['team1', 'team2', 'toss_winner', 'toss_decision']]
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "SVM": SVC(kernel="linear", probability=True)
}

best_model, best_accuracy = None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    if acc > best_accuracy:
        best_accuracy, best_model = acc, model

# Save the best model and label encoder
pickle.dump(best_model, open("ipl_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

# Load the model & encoder
model = pickle.load(open("ipl_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Streamlit UI
st.title("🏏 IPL Match Winner Prediction")
st.sidebar.header("📊 Select Match Details")

# Team selection
teams = sorted(matches["team1"].dropna().unique())
team1 = st.sidebar.selectbox("Select Team 1", teams)
team2 = st.sidebar.selectbox("Select Team 2", [t for t in teams if t != team1])
toss_winner = st.sidebar.selectbox("Select Toss Winner", [team1, team2])
toss_decision = st.sidebar.radio("Toss Decision", ["bat", "field"])

# Prediction button
if st.sidebar.button("🔮 Predict Winner"):
    try:
        if team1 not in label_encoder.classes_ or team2 not in label_encoder.classes_ or toss_winner not in label_encoder.classes_:
            st.error("❌ Selected teams are not in the training data. Please select valid teams.")
        else:
            team1_encoded = label_encoder.transform([team1])[0]
            team2_encoded = label_encoder.transform([team2])[0]
            toss_winner_encoded = label_encoder.transform([toss_winner])[0]
            toss_decision_encoded = 0 if toss_decision == "bat" else 1

            prediction_input = np.array([[team1_encoded, team2_encoded, toss_winner_encoded, toss_decision_encoded]])
            predicted_winner_encoded = model.predict(prediction_input)[0]
            predicted_winner = label_encoder.inverse_transform([predicted_winner_encoded])[0]

            st.subheader("🏆 Predicted Match Winner")
            st.success(f"🎉 {predicted_winner} is likely to win!")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

# Team performance stats
st.subheader("📊 Team Performance Stats")
teams_wins = matches['winner'].value_counts()
st.write(f"✅ {team1} Wins: {teams_wins.get(team1, 0)}")
st.write(f"✅ {team2} Wins: {teams_wins.get(team2, 0)}")

# Visualization
st.subheader("📈 Team Win Comparison")
fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(x=[team1, team2], y=[teams_wins.get(team1, 0), teams_wins.get(team2, 0)], palette="coolwarm", ax=ax)
ax.set_ylabel("Total Wins")
st.pyplot(fig)