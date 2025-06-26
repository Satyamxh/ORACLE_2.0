import pandas as pd
import random
from model import OracleModel

def simulate_appeal_chain(model: OracleModel, appeal_prob: float, max_appeals: int):
    results = []
    current_jurors = model.num_jurors
    params = {
        'honesty': model.honesty, 'rationality': model.rationality,
        'noise': model.noise, 'p': model.p, 'd': model.d,
        'epsilon': model.epsilon, 'payoff_type': model.payoff_type,
        'attack': model.attack, 'x_guess_noise': model.x_guess_noise
    }
    def simulate_round(n_jurors):
        sim_model = OracleModel(num_jurors=n_jurors, **params)
        outcome, vx, vy, px, py = sim_model.simulate_once()
        return outcome, vx, vy, px, py

    # Initial round (level 0)
    level = 0
    outcome, vx, vy, px, py = simulate_round(current_jurors)
    results.append((level, current_jurors, vx, vy, px, py, outcome))
    
    while random.random() < appeal_prob and level < max_appeals:
        level += 1
        current_jurors = current_jurors * 2 + 1
        outcome, vx, vy, px, py = simulate_round(current_jurors)
        results.append((level, current_jurors, vx, vy, px, py, outcome))
    return results

def run_simulations_with_appeals(num_simulations: int, appeal_prob: float, max_appeals: int,
                                 num_jurors: int, honesty: float, rationality: float,
                                 noise: float, p: float, d: float, epsilon: float,
                                 payoff_type: str, attack: bool, x_guess_noise: float):
    """
    Run multiple rounds with appeals, returning aggregated stats by level.
    """
    appeal_records = []

    final_outcomes = {'X': 0, 'Y': 0}
    # Accumulators for each level (0..max_appeals)
    votes_X_sum = [0]*(max_appeals+1)
    votes_Y_sum = [0]*(max_appeals+1)
    payoff_X_sum = [0.0]*(max_appeals+1)
    payoff_Y_sum = [0.0]*(max_appeals+1)
    counts_per_level = [0]*(max_appeals+1)

    for sim_id in range(num_simulations):
        # Create the initial model (same as in run.py) for level 0
        init_model = OracleModel(num_jurors=num_jurors, honesty=honesty, 
                                 rationality=rationality, noise=noise,
                                 p=p, d=d, epsilon=epsilon,
                                 payoff_type=payoff_type, attack=attack,
                                 x_guess_noise=x_guess_noise)
        chain = simulate_appeal_chain(init_model, appeal_prob, max_appeals)
        # Final outcome from last level
        _, _, final_vx, final_vy, _, _, _ = chain[-1]
        final_outcomes["X" if final_vx >= final_vy else "Y"] += 1

        # Accumulate sums by level
        for step in chain:
            level, nj, vx, vy, px, py, outcome = step
            votes_X_sum[level]   += vx
            votes_Y_sum[level]   += vy
            payoff_X_sum[level]  += px
            payoff_Y_sum[level]  += py
            counts_per_level[level] += 1
        
        # Add to appeal-level records
        appeal_records.append({
            "Simulation": sim_id,
            "Level": level,
            "NumJurors": nj,
            "X_votes": vx,
            "Y_votes": vy,
            "avg_payoff_X": px,
            "avg_payoff_Y": py,
            "Outcome": outcome
        })

    # Compute averages at each level (skipping levels never reached)
    avg_votes_X = []
    avg_votes_Y = []
    avg_payoff_X = []
    avg_payoff_Y = []
    for lvl in range(max_appeals+1):
        if counts_per_level[lvl] > 0:
            avg_votes_X.append(votes_X_sum[lvl]/counts_per_level[lvl])
            avg_votes_Y.append(votes_Y_sum[lvl]/counts_per_level[lvl])
            avg_payoff_X.append(payoff_X_sum[lvl]/counts_per_level[lvl])
            avg_payoff_Y.append(payoff_Y_sum[lvl]/counts_per_level[lvl])
        else:
            avg_votes_X.append(0)
            avg_votes_Y.append(0)
            avg_payoff_X.append(0.0)
            avg_payoff_Y.append(0.0)
    
    # Sort by Simulation then Level to ensure chronological flattening
    appeal_records.sort(key=lambda r: (r["Simulation"], r["Level"]))

    # Flattened history for fallback plotting
    history_X = []
    history_Y = []
    avg_payoff_X_flat = []
    avg_payoff_Y_flat = []
    
    for record in appeal_records:
        history_X.append(record["X_votes"])
        history_Y.append(record["Y_votes"])
        avg_payoff_X_flat.append(record["avg_payoff_X"])
        avg_payoff_Y_flat.append(record["avg_payoff_Y"])

    df_appeals = pd.DataFrame(appeal_records)

    level_counts = df_appeals["Level"].value_counts().sort_index()
    payoff_by_level = df_appeals.groupby("Level")[["avg_payoff_X", "avg_payoff_Y"]].mean()
    outcome_dist = df_appeals.groupby(["Level", "Outcome"]).size().unstack().fillna(0)
    outcome_dist = outcome_dist.div(outcome_dist.sum(axis=1), axis=0)  # normalize to proportions
    
    return {
        "final_outcome_counts": final_outcomes,
        "avg_votes_X_by_level": avg_votes_X,
        "avg_votes_Y_by_level": avg_votes_Y,
        "avg_payoff_X_by_level": avg_payoff_X,
        "avg_payoff_Y_by_level": avg_payoff_Y,
        "appeals_df": df_appeals,
        "history_X": history_X,
        "history_Y": history_Y,
        "avg_payoff_X": avg_payoff_X_flat,
        "avg_payoff_Y": avg_payoff_Y_flat,
        "level_counts": level_counts,
        "payoff_by_level": payoff_by_level,
        "outcome_distribution": outcome_dist
    }
