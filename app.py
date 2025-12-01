import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamshala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target' , min_value=0, max_value=300, value=0, format="%d")

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, max_value=300, value=0, format="%d")
with col4:
    overs = st.number_input('Overs completed', min_value=0, max_value=20, value=0, format="%d")
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, value=0, format="%d")

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")


import matplotlib.pyplot as plt

    # Data
    teams_plot = [batting_team, bowling_team]
    probabilities = [win * 100, loss * 100]

    # Colors like your sample image
    colors = ["#1f77b4", "#ff7f0e"]  # Blue and Orange

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(teams_plot, probabilities, color=colors)

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha='center', va='bottom', fontsize=12
        )

    # Labels and title
    ax.set_ylabel("Win Probability (%)")
    ax.set_title("Win Probability Comparison")
    ax.set_ylim(0, 100)

    st.pyplot(fig)
