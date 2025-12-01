import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Teams and cities
teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
         'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamshala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load model pipeline
with open('pipe.pkl','rb') as f:
    model = pickle.load(f)

st.title('IPL Win Predictor')

st.header("Single Match Prediction")

# Single match input
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=0, max_value=300, value=0, format="%d")

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0, max_value=300, value=0, format="%d")
with col4:
    overs = st.number_input('Overs completed', min_value=0, max_value=20, value=0, format="%d")
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, value=0, format="%d")

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets_out
    crr = score / overs if overs != 0 else 0
    rrr = (runs_left*6)/balls_left if balls_left != 0 else 0

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[selected_city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets],
        'total_runs_x':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = model.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.subheader("Win Probability")
    st.write(f"{batting_team}: {round(win*100)}%")
    st.write(f"{bowling_team}: {round(loss*100)}%")

    # Plot graph
    fig, ax = plt.subplots(figsize=(6,4))
    teams_plot = [batting_team, bowling_team]
    probabilities = [win*100, loss*100]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax.bar(teams_plot, probabilities, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height+1, f"{height:.1f}%",
                ha='center', va='bottom', fontsize=12)
    ax.set_ylabel("Win Probability (%)")
    ax.set_ylim(0,100)
    ax.set_title("Win Probability Comparison")
    st.pyplot(fig)

st.header("Batch Prediction via CSV Upload")

# CSV upload
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data)

    predictions = model.predict(data)
    prediction_proba = model.predict_proba(data)

    data['Prediction'] = predictions
    data['Win Probability (%)'] = prediction_proba.max(axis=1) * 100

    st.subheader("Prediction Results")
    st.dataframe(data)

    # Optional: Graph for batch (top 5 matches or first match)
    st.subheader("Graph for First Uploaded Match")
    first_match = data.iloc[0]
    fig, ax = plt.subplots(figsize=(6,4))
    teams_plot = [first_match['batting_team'], first_match['bowling_team']]
    probabilities = [prediction_proba[0][1]*100, prediction_proba[0][0]*100]
    bars = ax.bar(teams_plot, probabilities, color=["#1f77b4", "#ff7f0e"])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height+1, f"{height:.1f}%",
                ha='center', va='bottom', fontsize=12)
    ax.set_ylabel("Win Probability (%)")
    ax.set_ylim(0,100)
    ax.set_title("Win Probability Comparison - First Match")
    st.pyplot(fig)
