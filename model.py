import random
import numpy as np
from scipy.stats import binom
from typing import List, Tuple
from agents import Juror
from payoff_mechanisms import (compute_payoff_basic_attack, compute_payoff_basic_no_attack, 
                               compute_payoff_redistributive_attack, compute_payoff_redistributive_no_attack, 
                               compute_payoff_symbiotic_attack, compute_payoff_symbiotic_no_attack)

class OracleModel:
    """
    Agent-based model of the Schelling point oracle. Simulates a single dispute resolution round 
    with a panel of jurors voting on outcome "X" vs "Y", including payoff mechanism and optional attack.
    """
    def __init__(self, num_jurors: int, honesty: float, rationality: float, noise: float,
                 p: float, d: float, epsilon: float, payoff_type: str, attack: bool, 
                 x_guess_noise: float):
        # Store model parameters
        self.num_jurors = num_jurors              # Number of jurors in the panel
        self.honesty = honesty                    # Honesty probability for all jurors (stick to their belief)
        self.rationality = rationality            # Rationality probability for all jurors (maximise their payoff)
        self.noise = noise                        # Noise standard deviation for payoff estimation (juror's imperfect perception of the payoff by adding a noise - simulate human decision - making)
        self.p = p                                # Base reward (p)
        self.d = d                                # Deposit (stake)
        self.epsilon = epsilon                    # Bonus payoff (epsilon) given by attacker for bribery
        self.payoff_type = payoff_type            # Payoff mechanism: "Basic", "Redistributive", or "Symbiotic"
        self.attack = attack                      # Whether an attack (p+epsilon attack) is enabled
        self.x_guess_noise = x_guess_noise        # Noise for estimating 'x' for symbiotic and redistributive mechanisms

        # Initialise the jurors
        self.jurors: List[Juror] = [Juror(honesty, rationality, noise) for _ in range(num_jurors)]
        # Treat all jurors as the selected panel for voting
        self.selected_jurors: List[Juror] = self.jurors

        # Appending results for CVS file
        self.history_X = []
        self.history_Y = []

    def _expected_payoffs(self, juror_index: int) -> Tuple[float, float]:
        other_count = self.num_jurors - 1
        
        # estimation to determine x (50% of jurors - conservative estimate)
        # noise is added to simulate human decision making
        mean = 0.5
        std = self.x_guess_noise
        q = random.gauss(mean, std)
        q = max(0.0, min(1.0, q))

        exp_payoff_X = 0.0
        exp_payoff_Y = 0.0
        majority_needed = self.num_jurors // 2 + 1

        k_values = np.arange(0, other_count + 1)
        prob_k = binom.pmf(k_values, other_count, q)

        for k, prob in zip(k_values, prob_k):
            votes_X_ifX = k + 1
            votes_Y_ifX = other_count - k
            votes_X_ifY = k
            votes_Y_ifY = other_count - k + 1


            # Compute payoffs for A and B votes under different mechanisms
            if self.attack:
                if self.payoff_type.lower() == "basic":
                    payoff_X = compute_payoff_basic_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", self.p, self.d, self.epsilon)
                    payoff_Y = compute_payoff_basic_attack("Y", "X" if votes_Y_ifY < majority_needed else "Y", self.p, self.d, self.epsilon)
                    if self.jurors[juror_index].bribed:
                        payoff_Y = compute_payoff_basic_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", self.p, self.d, self.epsilon) + self.epsilon
                elif self.payoff_type.lower() == "redistributive":
                    payoff_X = compute_payoff_redistributive_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    payoff_Y = compute_payoff_redistributive_attack("Y", "X" if votes_Y_ifY < majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    if self.jurors[juror_index].bribed:
                        payoff_Y = compute_payoff_redistributive_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon) + self.epsilon
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_X = compute_payoff_symbiotic_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    payoff_Y = compute_payoff_symbiotic_attack("Y", "X" if votes_Y_ifY < majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    if self.jurors[juror_index].bribed:
                        payoff_Y = compute_payoff_symbiotic_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon) + self.epsilon
                else:
                    payoff_X = payoff_Y = 0.0
            else:
                if self.payoff_type.lower() == "basic":
                    payoff_X = compute_payoff_basic_no_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", self.p, self.d)
                    payoff_Y = compute_payoff_basic_no_attack("Y", "X" if votes_Y_ifY < majority_needed else "Y", self.p, self.d)
                elif self.payoff_type.lower() == "redistributive":
                    payoff_X = compute_payoff_redistributive_no_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                    payoff_Y = compute_payoff_redistributive_no_attack("Y", "X" if votes_Y_ifY < majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_X = compute_payoff_symbiotic_no_attack("X", "X" if votes_X_ifX >= majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                    payoff_Y = compute_payoff_symbiotic_no_attack("Y", "X" if votes_Y_ifY < majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                else:
                    payoff_X = payoff_Y = 0.0

            exp_payoff_X += prob * payoff_X
            exp_payoff_Y += prob * payoff_Y

        return exp_payoff_X, exp_payoff_Y

    def simulate_once(self) -> Tuple[str, int, int, float, float]:
        """
        Run a single simulation round: assign juror beliefs, apply bribery if attack is enabled, 
        collect votes from all jurors, and determine the outcome.
        
        Returns:
            outcome (str): "X" or "Y" (winning outcome of this round)
            votes_for_X (int): Number of jurors who voted "X"
            votes_for_Y (int): Number of jurors who voted "Y"
        """

        x_payoffs = []
        y_payoffs = []

        # 1. Assign random beliefs to jurors ("X" belief with probability p, else "Y")
        for juror in self.jurors:
            juror.bribed = False  # reset bribery status each round
            juror.belief = "X" if random.random() < self.p else "Y"

        # 2. Mark a fraction of jurors as bribed (if attack is enabled)
        if self.attack:
            for juror in self.jurors:
                juror.bribed = True

        # 3. Each juror makes a voting decision
        votes = []
        for i, juror in enumerate(self.jurors):
            expX, expY = self._expected_payoffs(i)       # compute expected payoff of voting X vs Y
            vote = juror.decide_vote(expX, expY)         # juror decides vote based on expectations
            votes.append(vote)
            if vote == "X":
                x_payoffs.append(expX)
            else:
                y_payoffs.append(expY)

        # 4. Count votes
        votes_for_X = votes.count("X")
        votes_for_Y = votes.count("Y")

        # 5. Determine winning outcome (tie-break goes to "A")
        outcome = "X" if votes_for_X >= votes_for_Y else "Y"

        # Store votes in a dictionary for potential payoff analysis
        self.votes = {"X": votes_for_X, "Y": votes_for_Y}

        x_payoffs = []
        y_payoffs = []
        for i, juror in enumerate(self.jurors):
            vote = votes[i]
            k = votes.count("X") - (1 if vote == "X" else 0)
            
            if self.attack:
                if self.payoff_type.lower() == "basic":
                    payoff = compute_payoff_basic_attack(vote, outcome, self.p, self.d, self.epsilon)
                elif self.payoff_type.lower() == "redistributive":
                    payoff = compute_payoff_redistributive_attack(vote, outcome, k, self.num_jurors, self.p, self.d, self.epsilon)
                elif self.payoff_type.lower() == "symbiotic":
                    payoff = compute_payoff_symbiotic_attack(vote, outcome, k, self.num_jurors, self.p, self.d, self.epsilon)
                else:
                    payoff = 0.0
            else:
                if self.payoff_type.lower() == "basic":
                    payoff = compute_payoff_basic_no_attack(vote, outcome, self.p, self.d)
                elif self.payoff_type.lower() == "redistributive":
                    payoff = compute_payoff_redistributive_no_attack(vote, outcome, k, self.num_jurors, self.p, self.d)
                elif self.payoff_type.lower() == "symbiotic":
                    payoff = compute_payoff_symbiotic_no_attack(vote, outcome, k, self.num_jurors, self.p, self.d)
                else:
                    payoff = 0.0
            
            if vote == "X":
                x_payoffs.append(payoff)
            else:
                y_payoffs.append(payoff)
        
        # if no one voted for X then x_payoffs is set to 0 and same for Y
        return outcome, votes_for_X, votes_for_Y, np.mean(x_payoffs) if x_payoffs else 0.0, np.mean(y_payoffs) if y_payoffs else 0.0

    def run_simulations(self, num_simulations: int, progress_bar=None, status_text=None) -> dict:
        """
        Run the simulation for a given number of rounds and aggregate the results.
        
        Returns:
            results (dict) with keys:
              - "total_runs": number of rounds simulated
              - "outcome_counts": {'X': count_X_wins, 'Y': count_Y_wins}
              - "attack_success_rate": fraction of rounds outcome was "Y" (None if attack is False)
              - "average_votes_X": average number of "X" votes per round
              - "average_votes_Y": average number of "Y" votes per round
        """

        # in case function gets reused, clear the data for proper CVS file results
        self.history_X.clear()
        self.history_Y.clear()

        outcomes = {"X": 0, "Y": 0}
        votes_X_array = np.zeros(num_simulations, dtype=np.uint16)
        votes_Y_array = np.zeros(num_simulations, dtype=np.uint16)

        payoff_X_array = np.zeros(num_simulations, dtype=np.float32)
        payoff_Y_array = np.zeros(num_simulations, dtype=np.float32)

        for i in range(num_simulations):
            outcome, votes_X, votes_Y, avg_X_payoff, avg_Y_payoff = self.simulate_once()
            outcomes[outcome] += 1
            votes_X_array[i] = votes_X
            votes_Y_array[i] = votes_Y
            payoff_X_array[i] = avg_X_payoff
            payoff_Y_array[i] = avg_Y_payoff

            # Update progress bar and status
            if progress_bar and i % 100 == 0:
                progress_bar.progress((i + 1) / num_simulations)
            if status_text and i % 100 == 0:
                status_text.text(f"Running simulation {i + 1} / {num_simulations}")
        
        # Only convert to list once at the end
        self.history_X = votes_X_array.tolist()
        self.history_Y = votes_Y_array.tolist()

        # Calculate attack success rate if applicable
        attack_success_rate = None
        if self.attack:
            attack_success_rate = outcomes["Y"] / num_simulations
        # Calculate average votes for each option
        avg_votes_X = np.mean(votes_X_array)
        avg_votes_Y = np.mean(votes_Y_array)

        self.avg_payoff_X = payoff_X_array.tolist()
        self.avg_payoff_Y = payoff_Y_array.tolist()
        
        return {
            "total_runs": num_simulations,
            "outcome_counts": outcomes,
            "attack_success_rate": attack_success_rate,
            "average_votes_X": avg_votes_X,
            "average_votes_Y": avg_votes_Y
        }
