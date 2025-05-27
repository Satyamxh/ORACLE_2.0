# === Updated run.py with Honesty, Rationality, and Noise ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import OracleModel
from payoff_mechanisms import (
    compute_payoff_basic_attack,
    compute_payoff_redistributive_attack,
    compute_payoff_symbiotic_attack,
    compute_payoff_basic_no_attack,
    compute_payoff_redistributive_no_attack,
    compute_payoff_symbiotic_no_attack
)

# Sidebar controls
st.sidebar.title("Simulation Controls")
num_jurors = st.sidebar.slider("Number of Jurors", 3, 101, 11, step=2)
honesty = st.sidebar.slider("Honesty Level (0 = Never vote belief, 1 = Always vote belief)", 0.0, 1.0, 0.8)
rationality = st.sidebar.slider("Rationality (0 = Random vote, 1 = Maximize payoff)", 0.0, 1.0, 0.7)
noise_level = st.sidebar.slider("Noise Level (Variance in perceived payoff)", 0.0, 2.0, 0.1)
p = st.sidebar.slider("Reward (p)", 0.0, 5.0, 1.0)
d = st.sidebar.slider("Penalty (d)", 0.0, 5.0, 1.0)
epsilon = st.sidebar.slider("Bribe Bonus (epsilon)", 0.0, 2.0, 0.5)
bribed_fraction = st.sidebar.slider("Bribed Fraction", 0.0, 1.0, 0.3)
num_simulations = st.sidebar.slider("Simulations", 50, 5000, 500)
attack_enabled = st.sidebar.checkbox("Enable Attack (p + epsilon)", value=True)

payoff_type = st.sidebar.selectbox("Payoff Mechanism", ["Basic", "Redistributive", "Symbiotic"])

# Map to correct payoff mechanism based on attack toggle
payoff_map_attack = {
    "Basic": compute_payoff_basic_attack,
    "Redistributive": compute_payoff_redistributive_attack,
    "Symbiotic": compute_payoff_symbiotic_attack,
}

payoff_map_no_attack = {
    "Basic": compute_payoff_basic_no_attack,
    "Redistributive": compute_payoff_redistributive_no_attack,
    "Symbiotic": compute_payoff_symbiotic_no_attack,
}

payoff_function = payoff_map_attack[payoff_type] if attack_enabled else payoff_map_no_attack[payoff_type]

# Run simulations
results = {"X": 0, "Y": 0, "Attack Successes": 0}
for _ in range(num_simulations):
    model = OracleModel(
        num_jurors=num_jurors,
        honesty=honesty,
        rationality=rationality,
        noise=noise_level,
        p=p,
        d=d,
        epsilon=epsilon,
        bribed_fraction=bribed_fraction,
        payoff_mechanism=payoff_function
    )
    model.step()
    outcome = "X" if model.votes["X"] >= model.votes["Y"] else "Y"
    results[outcome] += 1
    if attack_enabled and outcome == "Y":
        results["Attack Successes"] += 1

# Display results
st.title("Schelling Oracle ABM Simulation")
st.markdown(f"**Total Simulations:** {num_simulations}")
st.markdown(f"**Attack Success Rate:** {results['Attack Successes'] / num_simulations:.2%}" if attack_enabled else "**Attack Success Rate:** N/A")

fig, ax = plt.subplots()
ax.bar(["X", "Y"], [results["X"], results["Y"]], color=["blue", "red"])
ax.set_ylabel("Number of Wins")
ax.set_title("Outcome Distribution")
st.pyplot(fig)
