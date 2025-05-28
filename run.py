import streamlit as st
from model import OracleModel

st.title("Schelling Oracle Simulation")

# Sidebar controls for model parameters
st.sidebar.header("Simulation Parameters")
num_jurors = st.sidebar.slider("Number of Jurors", min_value=1, max_value=100, value=11, step=1)
honesty = st.sidebar.slider("Honesty (Probability of sincere vote)", 0.0, 1.0, value=0.8, step=0.05)
rationality = st.sidebar.slider("Rationality (Probability of choosing best payoff)", 0.0, 1.0, value=0.7, step=0.05)
noise = st.sidebar.slider("Noise (Std dev of payoff perception)", 0.0, 1.0, value=0.1, step=0.01)
payoff_mode = st.sidebar.selectbox("Payoff Mechanism", ["Basic", "Redistributive", "Symbiotic"])
attack_mode = st.sidebar.checkbox("Enable p+ε Attack", value=False)
bribe_fraction = st.sidebar.slider("Bribed Fraction of Jurors", 0.0, 1.0, value=0.0, step=0.05, disabled=(not attack_mode))
epsilon_bonus = st.sidebar.slider("Epsilon Bonus (Bribe amount ε)", 0.0, 5.0, value=0.0, step=0.1, disabled=(not attack_mode))
num_rounds = st.sidebar.number_input("Number of Simulation Rounds", min_value=1, max_value=10000, value=100, step=1)

# Fixed deposit and base reward fraction (can be adjusted if needed)
deposit = 1.0
base_reward_frac = 1.0

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
                    attack=attack_mode)

# Run the simulation for the specified number of rounds
results = model.run_simulations(int(num_rounds))

# Display simulation outcomes
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
