import math
import random
from statistics import mean
from typing import List, Tuple
from agents import Juror

class OracleModel:
    """
    Agent-based model of the Schelling point oracle. Simulates a single dispute resolution round 
    with a panel of jurors voting on outcome "A" vs "B", including payoff mechanism and optional attack.
    """
    def __init__(self, num_jurors: int, honesty: float, rationality: float, noise: float,
                 p: float, d: float, epsilon: float, bribed_fraction: float,
                 payoff_type: str, attack: bool):
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
        # Initialise the jurors
        self.jurors: List[Juror] = [Juror(honesty, rationality, noise) for _ in range(num_jurors)]
        # Treat all jurors as the selected panel for voting
        self.selected_jurors: List[Juror] = self.jurors
        # For symbiotic payoff, define parameters beta and external_reward
        if payoff_type.lower() == "symbiotic":
            self.beta = 0.5                   # Fraction of losers' deposit redistributed to winners
            self.external_reward = p * d      # External reward to each winner (e.g., fraction of deposit)
        else:
            self.beta = 1.0                   # Not used in Basic/Redistributive
            self.external_reward = 0.0        # Not used in Basic/Redistributive
        # Alias for attack bribe amount (used in payoff mechanism calculations if needed)
        self.bribe_amount = epsilon

    def _expected_payoffs(self, juror_index: int) -> Tuple[float, float]:
        """
        Compute the expected payoff for a given juror if they vote "A" vs if they vote "B".
        This uses the current model parameters (payoff mechanism, attack settings) and 
        assumes other jurors' votes are uncertain (modeled via probability q for another juror voting "A").
        """
        other_count = self.num_jurors - 1  # Number of other jurors aside from this one
        # Determine probability q that any given other juror votes "A"
        if other_count > 0:
            if self.attack:
                # Under attack: assume bribed_fraction of others will vote "B" (with 0% chance for "A"),
                # and the remaining (1 - bribed_fraction) vote "A" with probability p.
                q = (1 - self.bribed_fraction) * self.p
            else:
                # No attack: each other juror votes "A" with probability p (i.e., fraction of jurors with belief "A").
                q = self.p
        else:
            # If there are no other jurors, this juror's vote solely decides outcome (handle trivially).
            q = 0.0
        # Compute expected payoffs for voting "A" and voting "B" by summing over all possible numbers of others voting "A"
        exp_payoff_A = 0.0
        exp_payoff_B = 0.0
        majority_needed = self.num_jurors // 2 + 1  # Votes needed for a majority (>50%). If tie, "A" wins by rule.
        # Calculate binomial probabilities for k = 0...other_count (where k is number of other jurors voting "A")
        probabilities = []
        for k in range(other_count + 1):
            # Binomial probability P(k out of other_count vote A) = C(other_count,k) * q^k * (1-q)^(other_count-k)
            prob_k = math.comb(other_count, k) * (q ** k) * ((1 - q) ** (other_count - k))
            probabilities.append(prob_k)
        # Iterate over all possible k values to accumulate expected payoff
        for k in range(other_count + 1):
            prob_k = probabilities[k]
            # Scenario: this juror votes "A"
            votes_A_ifA = k + 1   # k others vote A, plus this juror's A vote
            votes_B_ifA = other_count - k
            if votes_A_ifA >= majority_needed:
                # Outcome: "A" wins if this juror votes A
                if self.payoff_type.lower() == "basic":
                    # Winner gets back deposit + p*d reward
                    payoff_A = self.d + self.p * self.d
                elif self.payoff_type.lower() == "redistributive":
                    # Winners share losers' deposits equally
                    winners = votes_A_ifA
                    losers = votes_B_ifA
                    if losers == 0 or winners == 0:
                        payoff_A = self.d
                    else:
                        payoff_A = self.d + (losers * self.d) / winners
                elif self.payoff_type.lower() == "symbiotic":
                    winners = votes_A_ifA
                    losers = votes_B_ifA
                    if losers == 0:
                        payoff_A = self.d + self.external_reward
                    else:
                        # Losers lose beta*d each; redistribute that plus external reward to winners
                        payoff_A = self.d + (losers * self.beta * self.d) / winners + self.external_reward
                else:
                    payoff_A = self.d  # default fallback (should not happen if payoff_type is valid)
            else:
                # Outcome: "B" wins and this juror voted A (losing side)
                if self.payoff_type.lower() == "basic":
                    payoff_A = 0.0  # lost deposit
                elif self.payoff_type.lower() == "redistributive":
                    payoff_A = 0.0  # lost deposit
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_A = (1 - self.beta) * self.d  # keeps portion of deposit (the rest lost)
                else:
                    payoff_A = 0.0
                # If attack is on and B wins, attacker does NOT compensate jurors who voted A (no additional payoff)
            # Scenario: this juror votes "B"
            votes_A_ifB = k        # k others vote A, plus this juror votes B
            votes_B_ifB = other_count - k + 1
            if votes_B_ifB >= majority_needed:
                # Outcome: "B" wins if this juror votes B
                if self.payoff_type.lower() == "basic":
                    payoff_B = self.d + self.p * self.d
                elif self.payoff_type.lower() == "redistributive":
                    winners = votes_B_ifB
                    losers = votes_A_ifB
                    if losers == 0 or winners == 0:
                        payoff_B = self.d
                    else:
                        payoff_B = self.d + (losers * self.d) / winners
                elif self.payoff_type.lower() == "symbiotic":
                    winners = votes_B_ifB
                    losers = votes_A_ifB
                    if losers == 0:
                        payoff_B = self.d + self.external_reward
                    else:
                        payoff_B = self.d + (losers * self.beta * self.d) / winners + self.external_reward
                else:
                    payoff_B = self.d
                # If attack is on and B wins, the attack succeeds (no special compensation needed for B voters)
            else:
                # Outcome: "A" wins and this juror voted B (losing side)
                if self.payoff_type.lower() == "basic":
                    payoff_B = 0.0
                elif self.payoff_type.lower() == "redistributive":
                    payoff_B = 0.0
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_B = (1 - self.beta) * self.d
                else:
                    payoff_B = 0.0
                # If attack is on and B loses, attacker compensates B voters with their deposit + epsilon bonus
                if self.attack:
                    payoff_B = self.d + self.epsilon
            # Add weighted payoff for this scenario (k others vote A) to expectations
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
            num_to_bribe = round(self.bribed_fraction * self.num_jurors)
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
        outcomes = {"A": 0, "B": 0}
        total_votes_A = []
        total_votes_B = []
        for _ in range(num_simulations):
            outcome, votes_A, votes_B = self.simulate_once()
            outcomes[outcome] += 1
            total_votes_A.append(votes_A)
            total_votes_B.append(votes_B)
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
