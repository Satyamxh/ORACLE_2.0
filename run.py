"""
To run this in terminal use

streamlit run "run.py"
"""

import streamlit as st
from model import OracleModel
import pandas as pd

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
                          help="This models uncertainty in the juror's perception of expected payoffs for each option. "
                               "A higher value increases the likelihood that jurors misjudge which option maximises their payoff. "
                               "This helps to simulate cognitive bias or limited understanding. This affects strategic (rational) voting behaviour.")
deposit = st.sidebar.slider("Deposit ($d$)", 0.0, 5.0, value=0.0, step=0.1,
                            help="Specifies the initial deposit paid by the juror ($d$ in payoff matrix).")
base_reward_frac = st.sidebar.slider("Base Reward ($p$)", 0.0, 5.0, value=1.0, step=0.1,
                                     help="Specifies the reward for voting with the majority ($p$ in payoff matrix).")
payoff_mode = st.sidebar.selectbox("Payoff Mechanism", ["Basic", "Redistributive", "Symbiotic"],
                                   help="Used to choose which payoff mechanism to use for simulation.")
x_guess_noise = st.sidebar.slider("Belief Noise in Peer Votes ($x$)", 0.0, 1.0, value=0.0, step=0.01, disabled=(payoff_mode == "Basic"),
                                  help="This models uncertainty in the juror's belief about how many other jurors will vote the same way. "
                                       r"A conservative estimate of 50$\%$ is chosen and you can set the level of variation. "
                                       "A higher value means more variation in their internal estimate of $x$. "
                                       "This is to help simulate human decision-making.")
attack_mode = st.sidebar.checkbox(r"Enable p+$\varepsilon$ Attack", value=False,
                                  help=r"Tick if you want to enable a $p+\varepsilon$ attack.")
bribe_fraction = st.sidebar.slider("Bribed Fraction of Jurors", 0.0, 1.0, value=0.0, step=0.05, disabled=(not attack_mode),
                                   help="Specifies the percentage of jurors upon which a bribe is sent.")
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
                    bribed_fraction=bribe_fraction,
                    payoff_type=payoff_mode,
                    attack=attack_mode,
                    x_guess_noise=x_guess_noise)

# Run the simulation for the specified number of rounds
results = model.run_simulations(int(num_rounds))

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
        "- **$p$**: Base reward multiplier",
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
    outcome = "A" if outcome_counts["A"] == 1 else "B"
    votes_for_A = int(results["average_votes_A"])
    votes_for_B = int(results["average_votes_B"])
    st.write(f"Outcome of this round: **{outcome}**")
    st.write(f"Votes — A: {votes_for_A}, B: {votes_for_B}")
    if attack_mode:
        if outcome == "B":
            st.write("Attack Outcome: **Succeeded** (Target outcome B achieved)")
        else:
            st.write("Attack Outcome: **Failed** (Target outcome B not achieved)")
else:
    # Multiple rounds: show aggregated statistics
    total_runs = results["total_runs"]
    outcome_counts = results["outcome_counts"]
    wins_A = outcome_counts["A"]
    wins_B = outcome_counts["B"]
    pct_A = (wins_A / total_runs) * 100
    pct_B = (wins_B / total_runs) * 100
    st.write(f"Out of **{total_runs}** simulation rounds:")
    st.write(f"- Outcome **A** won **{wins_A}** times ({pct_A:.1f}%)")
    st.write(f"- Outcome **B** won **{wins_B}** times ({pct_B:.1f}%)")
    if attack_mode:
        success_rate = results["attack_success_rate"] * 100
        st.write(f"Attack Success Rate (Outcome B wins): **{success_rate:.1f}%**")
    avg_A = results["average_votes_A"]
    avg_B = results["average_votes_B"]
    st.write(f"Average votes per round — A: **{avg_A:.2f}**, B: **{avg_B:.2f}**")

# download results data into a CVS file

# Extract vote history from model
history_X = model.history_X
history_Y = model.history_Y

# Prepare DataFrame for plotting and CSV download
rounds_index = list(range(1, len(history_X) + 1))
df = pd.DataFrame({
    "Round": rounds_index,
    "X_votes": history_X,
    "Y_votes": history_Y
})

# Determine majority and whether attack succeeded
df["Majority"] = df.apply(lambda row: "Y" if row["Y_votes"] > row["X_votes"] else "X", axis=1)
if attack_mode:
    df["AttackSucceeded"] = df["Majority"].apply(lambda m: 1 if m == "Y" else 0)
else:
    df["AttackSucceeded"] = 0

# Line chart of vote counts over time (only shown if multiple rounds)
if len(df) > 1:
    st.subheader("Votes Over Time")
    chart_data = df.set_index("Round")[["X_votes", "Y_votes"]]
    st.line_chart(chart_data)

# CSV download
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Results as CSV",
    data=csv_data,
    file_name="Oracle_Simulation_Results.csv",
    mime="text/csv"
)
