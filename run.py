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

help_appeals = """
This models the appeals process found in Kleros, where rulings can be challenged and retried by a larger jury pool.

Each round has a probability of being appealed. If an appeal occurs, a new panel of jurors is selected — double the size of the previous one plus one — and the case is retried under the same parameters.

Appeals can stack up to a limit of 3 appeal rounds.

This helps to simulate robustness and escalation in dispute resolution, where controversial outcomes may be re-evaluated, potentially leading to reversals or stronger majorities at higher levels.
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
appeal_mode = st.sidebar.checkbox("Enable Appeals", value=False,
                                  help=help_appeals)
appeal_prob = st.sidebar.slider("Appeal Probability", 0.0, 0.3, value=0.0, step=0.01, disabled=(not appeal_mode),
                                help=r"Probability of an appeal occurring after each round (capped at 30$\%$).")

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
if appeal_mode:
    from appeals import run_simulations_with_appeals
    results = run_simulations_with_appeals(
        num_simulations=int(num_rounds),
        appeal_prob=appeal_prob,
        max_appeals=3,             # capped at 3 appeals
        num_jurors=num_jurors,
        honesty=honesty,
        rationality=rationality,
        noise=noise,
        p=base_reward_frac,
        d=deposit,
        epsilon=epsilon_bonus,
        payoff_type=payoff_mode,
        attack=attack_mode,
        x_guess_noise=x_guess_noise
    )
else:
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

if appeal_mode:
    total_runs = num_rounds
    wins_X = results["final_outcome_counts"]["X"]
    wins_Y = results["final_outcome_counts"]["Y"]
    st.write(f"Out of **{total_runs}** simulation rounds (with appeals):")
    st.write(f"- Outcome **X** won **{wins_X}** times ({wins_X/total_runs*100:.1f}%)")
    st.write(f"- Outcome **Y** won **{wins_Y}** times ({wins_Y/total_runs*100:.1f}%)")
    if attack_mode:
        st.write(f"Attack Success Rate (final Y wins): **{(wins_Y/total_runs*100):.1f}%**")
    # Averages of votes per level and payoffs will be shown below as charts

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
    votes_for_X = int(results.get("average_votes_X", 0))
    votes_for_Y = int(results.get("average_votes_Y", 0))
    st.write(f"Outcome of this round: **{outcome}**")
    st.write(f"Votes — X: {votes_for_X}, Y: {votes_for_Y}")
    if attack_mode:
        if outcome == "Y":
            st.write("Attack Outcome: **Succeeded** (Target outcome Y achieved)")
        else:
            st.write("Attack Outcome: **Failed** (Target outcome Y not achieved)")
else:
    # Multiple rounds: show aggregated statistics
    total_runs = num_rounds 
    outcome_counts = results.get("outcome_counts", results.get("final_outcome_counts"))
    wins_X = outcome_counts["X"]
    wins_Y = outcome_counts["Y"]
    pct_X = (wins_X / total_runs) * 100
    pct_Y = (wins_Y / total_runs) * 100
    st.write(f"Out of **{total_runs}** simulation rounds:")
    st.write(f"- Outcome **X** won **{wins_X}** times ({pct_X:.1f}%)")
    st.write(f"- Outcome **Y** won **{wins_Y}** times ({pct_Y:.1f}%)")
    if attack_mode:
        success_rate = results.get("attack_success_rate", wins_Y / total_runs) * 100
        st.write(f"Attack Success Rate (Outcome Y wins): **{success_rate:.1f}%**")
    avg_X = results.get("average_votes_X", None)
    avg_Y = results.get("average_votes_Y", None)
    
    if avg_X is not None and avg_Y is not None:
        st.write(f"Average votes per round — X: **{avg_X:.2f}**, Y: **{avg_Y:.2f}**")

# Use model history for normal rounds even in appeal mode
history_X = results.get("history_X", [])
history_Y = results.get("history_Y", [])
avg_payoff_X = results.get("avg_payoff_X", [])
avg_payoff_Y = results.get("avg_payoff_Y", [])

# Prepare DataFrame for plotting and CSV download
index_label = "Round"
rounds_index = list(range(1, len(history_X) + 1))

df_overlay_plot = None

if appeal_mode and "appeals_df" in results:
    df_overlay = results["appeals_df"].copy()
    df_overlay["Round"] = df_overlay["Simulation"] + 1
    has_appeals = (df_overlay["Level"] > 0).any()

    # only use appeal level 0 for base chart
    df = df_overlay[df_overlay["Level"] == 0].copy()

    df_overlay_plot = pd.DataFrame({
        "Round": rounds_index,
        "X_votes": history_X,
        "Y_votes": history_Y,
        "avg_payoff_X": avg_payoff_X,
        "avg_payoff_Y": avg_payoff_Y,
    }) if has_appeals else None

    if has_appeals:
        if has_appeals:
            df_overlay_plot_CVS = df_overlay[df_overlay["Level"] > 0].copy()
            
            # Determine majority based on vote counts
            df_overlay_plot_CVS["Majority"] = df_overlay_plot_CVS.apply(lambda row: "Y" if row["Y_votes"] > row["X_votes"] else "X", axis=1)
            
            # Compute attack success only if attack mode is enabled
            
            if attack_mode:
                df_overlay_plot_CVS["AttackSucceeded"] = df_overlay_plot_CVS["Majority"].apply(lambda m: 1 if m == "Y" else 0)
            else:
                df_overlay_plot_CVS["AttackSucceeded"] = 0

else: # if no appeal mode is selected do regular
    df_overlay = None
    # fall back to regular history
    rounds_index = list(range(1, len(history_X) + 1))
    df = pd.DataFrame({
        "Round": rounds_index,
        "X_votes": history_X,
        "Y_votes": history_Y,
        "avg_payoff_X": avg_payoff_X,
        "avg_payoff_Y": avg_payoff_Y,
    })

# Determine majority and whether attack succeeded
df["Majority"] = df.apply(lambda row: "Y" if row["Y_votes"] > row["X_votes"] else "X", axis=1)
if attack_mode:
    df["AttackSucceeded"] = df["Majority"].apply(lambda m: 1 if m == "Y" else 0)
else:
    df["AttackSucceeded"] = 0

if df_overlay_plot is not None:
    df_overlay_plot["Majority"] = df_overlay_plot.apply(lambda row: "Y" if row["Y_votes"] > row["X_votes"] else "X", axis=1)
    if attack_mode:
        df_overlay_plot["AttackSucceeded"] = df_overlay_plot["Majority"].apply(lambda m: 1 if m == "Y" else 0)
    else:
        df_overlay_plot["AttackSucceeded"] = 0

# Line chart of vote counts across rounds (only shown if multiple rounds)
if len(df) > 1:
    st.subheader("Voting Dynamics Across Rounds")

    base_chart = alt.Chart(df).transform_fold(
        ["X_votes", "Y_votes"],
        as_=["Vote Type", "Count"]
    ).mark_line().encode(
        x=alt.X(f"{index_label}:Q", title=index_label),
        y=alt.Y("Count:Q", title="Number of Votes"),
        color=alt.Color("Vote Type:N", title="Vote Option",
            scale=alt.Scale(domain=["X_votes", "Y_votes"], range=["steelblue", "red"]),
            legend=alt.Legend(labelExpr="""{'X_votes': 'Votes for X', 'Y_votes': 'Votes for Y'}[datum.label]"""))
    ).properties(
        width=800,  # Change this to your desired width
        height=400,  # Change this to your desired height
    )

    if df_overlay is not None:
        overlay_chart = alt.Chart(df_overlay_plot).transform_fold(
            ["X_votes", "Y_votes"],
            as_=["Vote Type", "Count"]
        ).mark_line(strokeDash=[4, 2]).encode(
            x="Round:Q",
            y="Count:Q",
            color=alt.Color("Vote Type:N", title="Appeal Vote Option"),
            detail="Level:N"
        )
        
        st.altair_chart(base_chart + overlay_chart, use_container_width=False)
    else:
        st.altair_chart(base_chart, use_container_width=False)

# Average payoff

if len(df) == 1:
    st.subheader("Payoff per Vote Type of This Round")
    st.write(f"Average payoff — X: **{avg_payoff_X[0]:.2f}**, Y: **{avg_payoff_Y[0]:.2f}**")
elif len(df) > 1 and "avg_payoff_X" in df.columns and "avg_payoff_Y" in df.columns: # Line chart of Average payoffs across rounds (only shown if multiple rounds)
    st.subheader("Average Payoff per Vote Type Across Rounds")

    base_payoff = alt.Chart(df).transform_fold(
        ["avg_payoff_X", "avg_payoff_Y"],
        as_=["Vote Type", "Average Payoff"]
    ).mark_line().encode(
        x=alt.X(f"{index_label}:Q", title=index_label),
        y=alt.Y("Average Payoff:Q", title="Payoff"),
        color=alt.Color("Vote Type:N", title="Vote Option",
            scale=alt.Scale(domain=["avg_payoff_X", "avg_payoff_Y"], range=["steelblue", "red"]),
            legend=alt.Legend(labelExpr="""{'avg_payoff_X': 'Payoff for voting X', 'avg_payoff_Y': 'Payoff for voting Y'}[datum.label]"""))
    ).properties(
        width=800,
        height=400,
    )

    if df_overlay is not None:
        overlay_payoff = alt.Chart(df_overlay_plot).transform_fold(
            ["avg_payoff_X", "avg_payoff_Y"],
            as_=["Vote Type", "Average Payoff"]
        ).mark_line(strokeDash=[4,2]).encode(
            x="Round:Q",
            y="Average Payoff:Q",
            color=alt.Color("Vote Type:N", title="Appeal Payoff Option"),
            detail="Level:N"
        )
        
        st.altair_chart(base_payoff + overlay_payoff, use_container_width=False)
    else:
        st.altair_chart(base_payoff, use_container_width=False)

# CSV download for all results (voting dynamics and average payoff)

if df_overlay is not None and df_overlay_plot is not None and (df_overlay["Level"] > 0).any():
    col1, col2 = st.columns(2) # set two CVS button downloads side by side
    
    with col1:
        csv_data_1 = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Simulation Results",
            data=csv_data_1,
            file_name="Simulation_Results.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_data_2 = df_overlay_plot_CVS.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Simulation Results with Appeals",
            data=csv_data_2,
            file_name="Simulation_Results_with_Appeals.csv",
            mime="text/csv"
        )
else:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Simulation Results as a CSV file",
        data=csv_data,
        file_name="Simulation_Results.csv",
        mime="text/csv"
    )

if appeal_mode:
    if all(k in results for k in ["avg_votes_X_by_level", "avg_votes_Y_by_level", "avg_payoff_X_by_level", "avg_payoff_Y_by_level"]):
        levels = list(range(len(results["avg_votes_X_by_level"])))
        total_votes_by_level = [x + y for x, y in zip(results["avg_votes_X_by_level"], results["avg_votes_Y_by_level"])]

        df_levels = pd.DataFrame({
            "Level": levels,
            "Average X Votes": results["avg_votes_X_by_level"],
            "Average Y Votes": results["avg_votes_Y_by_level"],
            "Total Votes": total_votes_by_level,
            "X %": [100 * x / t if t > 0 else 0 for x, t in zip(results["avg_votes_X_by_level"], total_votes_by_level)],
            "Y %": [100 * y / t if t > 0 else 0 for y, t in zip(results["avg_votes_Y_by_level"], total_votes_by_level)],
            "Average X Payoff": results["avg_payoff_X_by_level"],
            "Average Y Payoff": results["avg_payoff_Y_by_level"]
        })
        
        # votes stacked bar chart
        st.subheader("Appeal Vote Dynamics")

        vote_chart = alt.Chart(df_levels).transform_fold(
            ["Average X Votes", "Average Y Votes"],
            as_=["Vote Type", "Average Votes"]
        ).mark_bar().encode(
            x=alt.X("Level:O", title="Amount of Appeals"),
            y=alt.Y("Average Votes:Q", title="Average Number of Votes", stack="zero"),
            color=alt.Color("Vote Type:N", title="Vote Option",
                scale=alt.Scale(domain=["Average X Votes", "Average Y Votes"], range=["steelblue", "red"]),
                legend=alt.Legend(labelExpr="""{'Average X Votes': 'Votes for X', 'Average Y Votes': 'Votes for Y'}[datum.label]"""))
        ).properties(width=600, height=300)

        st.altair_chart(vote_chart, use_container_width=False)

        st.markdown("### Appeal-Level Vote Share Percentages")
        
        # Determine which levels are actually present in the simulation
        used_levels = df_overlay["Level"].unique() if "Level" in df_overlay else df_levels["Level"].unique()
        
        # Filter df_levels to only include levels that were actually used
        df_levels_filtered = df_levels[df_levels["Level"].isin(used_levels)]

        for i, row in df_levels_filtered.iterrows():
            pct_x = row["X %"]
            pct_y = row["Y %"]
            st.write(f"- Appeal Amount = **{int(row['Level'])}**: **{pct_x:.1f}%** of users voted for X, **{pct_y:.1f}%** of users voted for Y")

        # payoff stacked bar chart
        st.subheader("Appeal Payoff Dynamics")
        
        payoff_chart = alt.Chart(df_levels).transform_fold(
            ["Average X Payoff", "Average Y Payoff"],
            as_=["Vote Type", "Average Payoff"]
        ).mark_bar().encode(
            x=alt.X("Level:O", title="Amount of Appeals"),
            y=alt.Y("Average Payoff:Q", title="Average Payoff", stack="zero"),
            color=alt.Color("Vote Type:N", title="Vote Option",
                scale=alt.Scale(domain=["Average X Payoff", "Average Y Payoff"], range=["steelblue", "red"]),
                legend=alt.Legend(labelExpr="""{'Average X Payoff': 'Payoff for X', 'Average Y Payoff': 'Payoff for Y'}[datum.label]"""))
        ).properties(width=600, height=300)
        
        st.altair_chart(payoff_chart, use_container_width=False)


