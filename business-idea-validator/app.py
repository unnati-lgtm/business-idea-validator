import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------- Dataset ----------------
data = {
    "funding": [100000, 500000, 200000, 800000, 300000],
    "team_size": [2, 5, 3, 8, 4],
    "market_size": [1, 3, 2, 3, 2],
    "competition": [3, 2, 3, 1, 2],
    "experience": [0, 1, 0, 1, 1],
    "success": [0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# ---------------- Model ----------------
X = df[['funding', 'team_size', 'market_size', 'competition', 'experience']]
y = df['success']

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- UI ----------------
st.title("🚀 Business Idea Validator")

funding = st.number_input("Funding Amount", min_value=10000)
team_size = st.slider("Team Size", 1, 10)

market = st.selectbox("Market Size", ["Small", "Medium", "Large"])
competition = st.selectbox("Competition Level", ["Low", "Medium", "High"])
experience = st.selectbox("Founder Experience", ["No", "Yes"])

market_map = {"Small": 1, "Medium": 2, "Large": 3}
competition_map = {"Low": 1, "Medium": 2, "High": 3}
experience_map = {"No": 0, "Yes": 1}

input_data = np.array([[
    funding,
    team_size,
    market_map[market],
    competition_map[competition],
    experience_map[experience]
]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    st.write("Success Probability:", round(probability * 100, 2), "%")

    if prediction == 1:
        st.success("High chance of success 🚀")
    else:
        st.error("Risky idea ⚠️")