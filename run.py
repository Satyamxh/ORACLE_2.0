"""
To run this in terminal use

streamlit run "run.py"
"""

import streamlit as st
from model import OracleModel
import pandas as pd
import altair as alt

# to add paragraph in help (formatting) - streamlit does not allow markdown and this is a way to bypass that restriction

help_attack = """
Tick if you want to enable a $p+\\varepsilon$ attack. 

$p+\\varepsilon$ attacks are done via smart contracts, they are publicly visible and target all jurors.

This simply means the juror's payoff matrix adapts to fit the attack.
"""

help_payoff_mech = """
Choose how jurors are rewarded depending on how their vote aligns with the outcome.

**Basic**: Jurors who vote with the majority get a fixed reward; others lose their deposit.

**Redistributive**: Losers' deposits are redistributed among winners. Payoff depends on how many others voted the same way.

**Symbiotic**: Rewards increase with coordination. The more jurors vote the same way, the greater the reward — fostering consensus.
"""

help_noise = """
This models uncertainty in the juror's perception of expected payoffs for each option.

A higher value increases the likelihood that jurors misjudge which option maximises their payoff.

This helps to simulate cognitive bias or limited understanding. This affects strategic (rational) voting behaviour.
"""

help_x_guess = """
This models uncertainty in the juror's belief about how many other jurors will vote for X.

A conservative estimate of 50$\%$ of the total number of jurors is selected for $x$.

This parameter sets the level of variation. A higher value means more variation in the juror's internal estimate of $x$.

This helps to human decision-making.
"""

st.title("Schelling Oracle Simulation")

# Sidebar controls for model parameters
st.sidebar.header("Simulation Parameters")
num_jurors = st.sidebar.slider("Number of Jurors", min_value=1, max_value=100, value=10, step=1,
                               help="Specifies the number of jurors voting.")
honesty = st.sidebar.slider("Honesty", 0.0, 1.0, value=0.35, step=0.05,
                            help="Specifies the probability of the juror sticking to their initial belief.")
rationality = st.sidebar.slider("Rationality", 0.0, 1.0, value=0.7, step=0.05,
                                help="Specifies the probability of the juror attempting to maximise their payoff.")
noise = st.sidebar.slider("Perception Noise (Payoff Uncertainty)", 0.0, 1.0, value=0.1, step=0.01,
                          help=help_noise)
deposit = st.sidebar.slider("Deposit ($d$)", 0.0, 5.0, value=0.0, step=0.1,
                            help="Specifies the initial deposit paid by the juror ($d$ in payoff matrix).")
base_reward_frac = st.sidebar.slider("Base Reward ($p$)", 0.0, 5.0, value=1.0, step=0.1,
                                     help="Specifies the reward for voting with the majority ($p$ in payoff matrix).")
payoff_mode = st.sidebar.selectbox("Payoff Mechanism", ["Basic", "Redistributive", "Symbiotic"],
                                   help=help_payoff_mech)
x_guess_noise = st.sidebar.slider("Belief Noise in Peer Votes ($x$)", 0.0, 1.0, value=0.0, step=0.01, disabled=(payoff_mode == "Basic"),
                                  help=help_x_guess)
attack_mode = st.sidebar.checkbox(r"Enable p+$\varepsilon$ Attack", value=False,
                                  help=help_attack)
epsilon_bonus = st.sidebar.slider(r"Epsilon (Bribe amount $\varepsilon$)", 0.0, 5.0, value=0.0, step=0.1, disabled=(not attack_mode),
                                  help=r"Specifies Bribe amount ($\varepsilon$ in payoff matrix).")
num_rounds = st.sidebar.number_input("Number of Simulation Rounds", min_value=1, max_value=10000, value=100, step=1,
                                     help="Specifies number of simulations to run.")

# Initialize the Oracle model with selected parameters
model = OracleModel(num_jurors=num_jurors,
                    honesty=honesty,
                    rationality=rationality,
                    noise=noise,
                    p=base_reward_frac,
                    d=deposit,
                    epsilon=epsilon_bonus,
                    payoff_type=payoff_mode,
                    attack=attack_mode,
                    x_guess_noise=x_guess_noise)

progress_bar = st.progress(0)
status_text = st.empty()

# Run the simulation for the specified number of rounds
results = model.run_simulations(int(num_rounds), progress_bar=progress_bar, status_text=status_text)

progress_bar.empty()
status_text.empty()

# Payoff matrix visualisation
st.subheader("Payoff Mechanism Matrix")

# Prepare table content based on selected payoff type and attack mode
if payoff_mode == "Basic":
    st.markdown("#### Basic Mechanism" + (" with Attack" if attack_mode else ""))
    data = {
        "X wins": [
            r"$p$",
            r"$-d$" if not attack_mode else r"$p+\varepsilon$"],
        "Y wins": [
            r"$-d$",
            r"$p$"
        ]
    }
    variables = [
        "- **$p$**: Base reward",
        "- **$d$**: Deposit amount",
    ]
    if attack_mode:
        variables.append(r"- **$\varepsilon$**: Bribe amount")

elif payoff_mode == "Redistributive":
    st.markdown("#### Redistributive Mechanism" + (" with Attack" if attack_mode else ""))
    data = {
        "X wins": [
            r"$\frac{(M - x - 1)d + Mp}{x + 1}$",
            r"$-d$" if not attack_mode else r"$\frac{(M - x - 1)d + Mp}{x + 1} + \varepsilon$"],
        "Y wins": [
            r"$-d$",
            r"$\frac{xd + Mp}{M - x}$"
        ]
    }
    variables = [
        "- **$p$**: Base reward multiplier",
        "- **$d$**: Deposit amount",
        "- **$x$**: Number of jurors who voted for X (other than user)",
        "- **$M$**: Total number of jurors",
    ]
    if attack_mode:
        variables.append(r"- **$\varepsilon$**: Bribe amount")

elif payoff_mode == "Symbiotic":
    st.markdown("#### Symbiotic Mechanism" + (" with Attack" if attack_mode else ""))
    data = {
        "X wins": [
            r"$\frac{p(x + 1)}{M}$", 
            r"$-d$" if not attack_mode else r"$\frac{p(x + 1)}{M} + \varepsilon$"],
        "Y wins": [
            r"$-d$",
            r"$\frac{p(M - x)}{M}$"
        ]
    }
    variables = [
        "- **$p$**: Base reward multiplier",
        "- **$d$**: Deposit amount",
        "- **$x$**: Number of jurors who voted for X (other than user)",
        "- **$M$**: Total number of jurors",
    ]
    if attack_mode:
        variables.append(r"- **$\varepsilon$**: Bribe amount")

# Display the table and variable explanations

df_details = pd.DataFrame(data, index=["User votes X", "User votes Y"])
st.table(df_details)

st.markdown("### Variables")
for var in variables:
    st.markdown(var)

# Display simulation results
st.subheader("Simulation Results")
if num_rounds == 1:
    # Single round: show outcome and votes
    outcome_counts = results["outcome_counts"]
    outcome = "X" if outcome_counts["X"] == 1 else "Y"
    votes_for_X = int(results["average_votes_X"])
    votes_for_Y = int(results["average_votes_Y"])
    st.write(f"Outcome of this round: **{outcome}**")
    st.write(f"Votes — X: {votes_for_X}, Y: {votes_for_Y}")
    if attack_mode:
        if outcome == "Y":
            st.write("Attack Outcome: **Succeeded** (Target outcome Y achieved)")
        else:
            st.write("Attack Outcome: **Failed** (Target outcome Y not achieved)")
else:
    # Multiple rounds: show aggregated statistics
    total_runs = results["total_runs"]
    outcome_counts = results["outcome_counts"]
    wins_X = outcome_counts["X"]
    wins_Y = outcome_counts["Y"]
    pct_X = (wins_X / total_runs) * 100
    pct_Y = (wins_Y / total_runs) * 100
    st.write(f"Out of **{total_runs}** simulation rounds:")
    st.write(f"- Outcome **X** won **{wins_X}** times ({pct_X:.1f}%)")
    st.write(f"- Outcome **Y** won **{wins_Y}** times ({pct_Y:.1f}%)")
    if attack_mode:
        success_rate = results["attack_success_rate"] * 100
        st.write(f"Attack Success Rate (Outcome Y wins): **{success_rate:.1f}%**")
    avg_X = results["average_votes_X"]
    avg_Y = results["average_votes_Y"]
    st.write(f"Average votes per round — X: **{avg_X:.2f}**, Y: **{avg_Y:.2f}**")

# download results data into a CVS file

# Extract vote history from model
history_X = model.history_X
history_Y = model.history_Y

# Prepare DataFrame for plotting and CSV download
rounds_index = list(range(1, len(history_X) + 1))
df = pd.DataFrame({
    "Round": rounds_index,
    "X_votes": history_X,
    "Y_votes": history_Y,
})

df["avg_payoff_X"] = model.avg_payoff_X
df["avg_payoff_Y"] = model.avg_payoff_Y

# Determine majority and whether attack succeeded
df["Majority"] = df.apply(lambda row: "Y" if row["Y_votes"] > row["X_votes"] else "X", axis=1)
if attack_mode:
    df["AttackSucceeded"] = df["Majority"].apply(lambda m: 1 if m == "Y" else 0)
else:
    df["AttackSucceeded"] = 0

# Line chart of vote counts across rounds (only shown if multiple rounds)
if len(df) > 1:
    st.subheader("Voting Dynamics Across Rounds")

    chart = alt.Chart(df).transform_fold(
        ["X_votes", "Y_votes"],
        as_=["Vote Type", "Count"]
    ).mark_line().encode(
        x=alt.X("Round:Q", title="Simulation Round"),
        y=alt.Y("Count:Q", title="Number of Votes"),
        color=alt.Color("Vote Type:N", title="Vote Option",
            scale=alt.Scale(domain=["X_votes", "Y_votes"], range=["steelblue", "red"]))
    ).properties(
        width=800,  # Change this to your desired width
        height=400,  # Change this to your desired height
    )

    st.altair_chart(chart, use_container_width=False)

# Average payoff

if len(df) == 1:
    st.subheader("Payoff per Vote Type of This Round")
    st.write(f"Average payoff — X: **{model.avg_payoff_X[0]:.2f}**, Y: **{model.avg_payoff_Y[0]:.2f}**")
elif len(df) > 1 and "avg_payoff_X" in df.columns and "avg_payoff_Y" in df.columns: # Line chart of Average payoffs across rounds (only shown if multiple rounds)
    st.subheader("Average Payoff per Vote Type Across Rounds")

    payoff_chart = alt.Chart(df).transform_fold(
        ["avg_payoff_X", "avg_payoff_Y"],
        as_=["Vote Type", "Average Payoff"]
    ).mark_line().encode(
        x=alt.X("Round:Q", title="Simulation Round"),
        y=alt.Y("Average Payoff:Q", title="Payoff"),
        color=alt.Color("Vote Type:N", title="Vote Option",
            scale=alt.Scale(domain=["avg_payoff_X", "avg_payoff_Y"], range=["steelblue", "red"]))
    ).properties(
        width=800,
        height=400,
    )

    st.altair_chart(payoff_chart, use_container_width=False)

# CSV download for all results (voting dynamics and average payoff)
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Simulation Results as a CSV file",
    data=csv_data,
    file_name="Voting_Dynamics_Results.csv",
    mime="text/csv"
)
