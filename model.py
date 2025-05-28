import math
import random
from statistics import mean
from typing import List, Tuple
from agents import Juror
from payoff_mechanisms import (compute_payoff_basic_attack, compute_payoff_basic_no_attack, 
                               compute_payoff_redistributive_attack, compute_payoff_redistributive_no_attack, 
                               compute_payoff_symbiotic_attack, compute_payoff_symbiotic_no_attack)

class OracleModel:
    """
    Agent-based model of the Schelling point oracle. Simulates a single dispute resolution round 
    with a panel of jurors voting on outcome "A" vs "B", including payoff mechanism and optional attack.
    """
    def __init__(self, num_jurors: int, honesty: float, rationality: float, noise: float,
                 p: float, d: float, epsilon: float, bribed_fraction: float,
                 payoff_type: str, attack: bool, x_guess_noise: float):
        # Store model parameters
        self.num_jurors = num_jurors              # Number of jurors in the panel
        self.honesty = honesty                    # Honesty probability for all jurors (stick to their belief)
        self.rationality = rationality            # Rationality probability for all jurors (maximise their payoff)
        self.noise = noise                        # Noise standard deviation for payoff estimation (juror's imperfect perception of the payoff by adding a noise - simulate human decision - making)
        self.p = p                                # Base reward (p)
        self.d = d                                # Deposit (stake)
        self.epsilon = epsilon                    # Bonus payoff (epsilon) given by attacker for bribery
        self.bribed_fraction = bribed_fraction    # Fraction of jurors that can be bribed by attacker
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

        exp_payoff_A = 0.0
        exp_payoff_B = 0.0
        majority_needed = self.num_jurors // 2 + 1
        probabilities = []
        for k in range(other_count + 1):
            prob_k = math.comb(other_count, k) * (q ** k) * ((1 - q) ** (other_count - k))
            probabilities.append(prob_k)

        for k in range(other_count + 1):
            prob_k = probabilities[k]

            votes_A_ifA = k + 1
            votes_B_ifA = other_count - k
            votes_A_ifB = k
            votes_B_ifB = other_count - k + 1

            # Compute payoffs for A and B votes under different mechanisms
            if self.attack:
                if self.payoff_type.lower() == "basic":
                    payoff_A = compute_payoff_basic_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", self.p, self.d, self.epsilon)
                    payoff_B = compute_payoff_basic_attack("Y", "X" if votes_B_ifB < majority_needed else "Y", self.p, self.d, self.epsilon)
                    if self.jurors[juror_index].bribed:
                        payoff_B = compute_payoff_basic_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", self.p, self.d, self.epsilon) + self.epsilon
                elif self.payoff_type.lower() == "redistributive":
                    payoff_A = compute_payoff_redistributive_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    payoff_B = compute_payoff_redistributive_attack("Y", "X" if votes_B_ifB < majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    if self.jurors[juror_index].bribed:
                        payoff_B = compute_payoff_redistributive_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon) + self.epsilon
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_A = compute_payoff_symbiotic_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    payoff_B = compute_payoff_symbiotic_attack("Y", "X" if votes_B_ifB < majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon)
                    if self.jurors[juror_index].bribed:
                        payoff_B = compute_payoff_symbiotic_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", k, self.num_jurors, self.p, self.d, self.epsilon) + self.epsilon
                else:
                    payoff_A = payoff_B = 0.0
            else:
                if self.payoff_type.lower() == "basic":
                    payoff_A = compute_payoff_basic_no_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", self.p, self.d)
                    payoff_B = compute_payoff_basic_no_attack("Y", "X" if votes_B_ifB < majority_needed else "Y", self.p, self.d)
                elif self.payoff_type.lower() == "redistributive":
                    payoff_A = compute_payoff_redistributive_no_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                    payoff_B = compute_payoff_redistributive_no_attack("Y", "X" if votes_B_ifB < majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_A = compute_payoff_symbiotic_no_attack("X", "X" if votes_A_ifA >= majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                    payoff_B = compute_payoff_symbiotic_no_attack("Y", "X" if votes_B_ifB < majority_needed else "Y", k, self.num_jurors, self.p, self.d)
                else:
                    payoff_A = payoff_B = 0.0

            exp_payoff_A += prob_k * payoff_A
            exp_payoff_B += prob_k * payoff_B

        return exp_payoff_A, exp_payoff_B

    def simulate_once(self) -> Tuple[str, int, int]:
        """
        Run a single simulation round: assign juror beliefs, apply bribery if attack is enabled, 
        collect votes from all jurors, and determine the outcome.
        
        Returns:
            outcome (str): "A" or "B" (winning outcome of this round)
            votes_for_A (int): Number of jurors who voted "A"
            votes_for_B (int): Number of jurors who voted "B"
        """
        # 1. Assign random beliefs to jurors ("A" belief with probability p, else "B")
        for juror in self.jurors:
            juror.bribed = False  # reset bribery status each round
            juror.belief = "A" if random.random() < self.p else "B"
        # 2. Mark a fraction of jurors as bribed (if attack is enabled)
        if self.attack and self.bribed_fraction > 0:
            num_to_bribe = math.ceil(self.bribed_fraction * self.num_jurors)
            # Randomly select jurors to bribe
            bribed_indices = random.sample(range(self.num_jurors), min(num_to_bribe, self.num_jurors))
            for idx in bribed_indices:
                self.jurors[idx].bribed = True
        # 3. Each juror makes a voting decision
        votes = []
        for i, juror in enumerate(self.jurors):
            expA, expB = self._expected_payoffs(i)       # compute expected payoff of voting A vs B
            vote = juror.decide_vote(expA, expB)         # juror decides vote based on expectations
            votes.append(vote)
        # 4. Count votes
        votes_for_A = votes.count("A")
        votes_for_B = votes.count("B")
        # 5. Determine winning outcome (tie-break goes to "A")
        outcome = "A" if votes_for_A >= votes_for_B else "B"
        # Store votes in a dictionary for potential payoff analysis (using "X"/"Y" to correspond to "A"/"B")
        self.votes = {"X": votes_for_A, "Y": votes_for_B}
        return outcome, votes_for_A, votes_for_B

    def run_simulations(self, num_simulations: int) -> dict:
        """
        Run the simulation for a given number of rounds and aggregate the results.
        
        Returns:
            results (dict) with keys:
              - "total_runs": number of rounds simulated
              - "outcome_counts": {'A': count_A_wins, 'B': count_B_wins}
              - "attack_success_rate": fraction of rounds outcome was "B" (None if attack is False)
              - "average_votes_A": average number of "A" votes per round
              - "average_votes_B": average number of "B" votes per round
        """

        # in case function gets reused, clear the data for proper CVS file results
        self.history_X.clear()
        self.history_Y.clear()

        outcomes = {"A": 0, "B": 0}
        total_votes_A = []
        total_votes_B = []

        for _ in range(num_simulations):
            outcome, votes_A, votes_B = self.simulate_once()
            outcomes[outcome] += 1
            total_votes_A.append(votes_A)
            total_votes_B.append(votes_B)

            # append list with results for CVS file
            self.history_X.append(votes_A)
            self.history_Y.append(votes_B)

        # Calculate attack success rate if applicable
        attack_success_rate = None
        if self.attack:
            attack_success_rate = outcomes["B"] / num_simulations
        # Calculate average votes for each option
        avg_votes_A = mean(total_votes_A) if total_votes_A else 0.0
        avg_votes_B = mean(total_votes_B) if total_votes_B else 0.0
        
        return {
            "total_runs": num_simulations,
            "outcome_counts": outcomes,
            "attack_success_rate": attack_success_rate,
            "average_votes_A": avg_votes_A,
            "average_votes_B": avg_votes_B
        }
