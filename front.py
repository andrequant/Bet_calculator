import streamlit as st
from codes import *
import utils
import matplotlib.pyplot as plt
import plotly.express as px


st.title('Bets Calculator')



# Include explaining text

st.number_input('Bankroll', min_value = 0.0, step=1.0, key='bank', placeholder=0)


x = st.slider('Choose how many games to bet on', 1, 15)  # 👈 this is a widget


left_column, mid_column, right_column, last_column = st.columns(4)

team_sides = ['Home','Away']
results = ['Home', 'Draw','Away']



# Initialize an empty dictionary to hold the input data
#if 'table_names' not in st.session_state:
st.session_state.table_names = {row: {col: "" for col in range(2)} for row in range(x)}
st.session_state.table_probs = {row: {col: "" for col in range(3)} for row in range(x)}
st.session_state.table_odds = {row: {col: "" for col in range(3)} for row in range(x)}

# Function to display the table with text_input for each cell
def display_table_names():
    st.write('Team Names')
    for row in range(x):
        cols = st.columns(2)
        for col, st_col in zip(range(2), cols):
            with st_col:
                # Create a text input for each cell, you can replace it with other widgets if needed
                st.session_state.table_names[row][col] = st.text_input("", key=f"N:{row}_{col}", placeholder=f"{team_sides[col]} {row}")

def display_table_probs():
    st.write('Probabilities')
    for row in range(x):
        cols = st.columns(3)
        for col, st_col in zip(range(3), cols):
            with st_col:
                # Create a text input for each cell, you can replace it with other widgets if needed
                st.session_state.table_probs[row][col] = utils.convert_to_float(
                    st.text_input("", value=None, key=f"P:{row}_{col}", placeholder=f"{results[col]} {row}"))

def display_table_odds():
    st.write('Odds')
    for row in range(x):
        cols = st.columns(3)
        for col, st_col in zip(range(3), cols):
            with st_col:
                # Create a text input for each cell, you can replace it with other widgets if needed
                st.session_state.table_odds[row][col] = utils.convert_to_float(
                    st.text_input("", value=None, key=f"O:{row}_{col}", placeholder=f"{results[col]} {row}"))


with left_column:
    display_table_names()

with mid_column:
    display_table_probs()

with right_column:
    display_table_odds()


st.session_state.test = 0

odds = utils.dict_to_array(st.session_state.table_odds)
probs = utils.dict_to_array(st.session_state.table_probs)

# To check if any field is left blank
try:
    np.sum(odds)
    np.sum(probs)
except:
    st.session_state.empty = True
else:
    st.session_state.empty = False

if not st.session_state.empty:
    idx = select_bets(probs, odds)
    probs_filt, odds_filt = filtered_bets(idx, probs, odds)
    kelly = simple_kelly(probs_filt, odds_filt)
    w0 = kelly/(len(probs_filt)*2)


    bounds = [(0.0,1)] * len(w0)
    result = minimize(neg_compute_expected_value, w0, args=(probs_filt, odds_filt), 
                        constraints={'type': 'ineq', 'fun': constraint}, bounds=bounds)

    returns, probs = compute_expected_value(result.x, probs_filt, odds_filt)
    st.session_state.optimal_weights = result.x * st.session_state.bank

    bets_sides = [f'Game {i+1}: {results[idx[i]]}' for i in range(len(idx))]


    # Plot
    #plt.figure(figsize=(10, 6))
    #plt.bar(bets_sides, st.session_state.optimal_weights)
    #plt.xlabel('Categories')
    #plt.ylabel('Values')
    #plt.title('Categorical Bar Plot')

    #st.pyplot(plt)
    with last_column:
        st.write('Result')
        #st.write("")
        for row in range(x):
            st.write("")
            st.write(results[idx[row]], st.session_state.optimal_weights[row])

    df = {'Bets': bets_sides, 'Value': st.session_state.optimal_weights}
    fig = px.bar(df, x='Bets', y='Value', barmode='group')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write(f'The total amount to be bet is {np.sum(result.x):.2%} or $ {np.sum(result.x)*st.session_state.bank:.2f}.')
    st.write(f'The expected return is $ {returns*st.session_state.bank:.2f}.')
    st.write(f'This is {returns:.2%} of the bankroll or {returns*st.session_state.bank*(1/(np.sum(result.x)*st.session_state.bank)):.2%} over the amount invested.')



#st.session_state