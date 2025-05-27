import math
import random
from statistics import mean
from typing import List, Tuple
from agents import Juror

class OracleModel:
    """
    Simulates the Schelling oracle (single-round) with a panel of jurors.
    """
    def __init__(self, num_jurors: int, honesty: float, rationality: float, noise: float,
                 p: float, d: float, epsilon: float,
                 bribed_fraction: float, payoff_type: str, attack: bool):
        self.N = num_jurors
        self.honesty = honesty
        self.rationality = rationality
        self.noise = noise
        self.p = p               # Could be used as reward factor or belief fraction depending on context
        self.d = d               # deposit each juror stakes
        self.epsilon = epsilon   # attack bribe bonus
        self.bribed_fraction = bribed_fraction
        self.payoff_type = payoff_type  # "Basic", "Redistributive", or "Symbiotic"
        self.attack = attack
        # Initialize jurors list
        self.jurors: List[Juror] = [Juror(honesty, rationality, noise) for _ in range(num_jurors)]
        # Determine symbiotic parameters if needed
        if payoff_type.lower() == "symbiotic":
            # define fraction of deposit to redistribute (beta) and external reward factor
            self.beta = 0.5    # e.g., 50% of losers' stake is redistributed
            self.external_reward = p * d  # reward given to each winner (p fraction of deposit)
        else:
            self.beta = 1.0    # not used in non-symbiotic modes
            self.external_reward = 0.0  # not used in basic/redistributive (could be interpreted differently for Basic)
    
    def _expected_payoffs(self, juror_index: int) -> Tuple[float, float]:
        """
        Compute the expected payoff for a specific juror voting A vs voting B.
        This accounts for the current model parameters, including the attack scenario.
        """
        # Get effective probability q that any other given juror votes A.
        # We exclude the current juror from this calculation (focus on others).
        if self.attack:
            # Under attack: assume bribed_fraction * (N-1) of others definitely vote B.
            # So they have 0 chance to vote A. The remaining (1 - bribed_fraction) can vote A with probability p.
            q = (1 - self.bribed_fraction) * self.p
        else:
            # No attack: each other juror votes A with probability p (basically fraction with belief A).
            q = self.p
        other_count = self.N - 1  # number of other jurors
        # We will compute expected payoff by summing over possible k (others voting A).
        # Use binomial probabilities for k.
        # To avoid computing large combinations repeatedly, we can calculate probabilities iteratively or use math.comb.
        # For clarity and given moderate N, we'll just compute directly.
        exp_payoff_A = 0.0
        exp_payoff_B = 0.0
        # Determine threshold for A winning. For simplicity, let's define majority threshold.
        majority_needed = self.N // 2 + 1  # smallest number of votes to have >50% (works for N odd; if N even, this treats >50%)
        # Precompute binomial probability distribution for k = 0,...,other_count
        # We'll use a simple approach to compute P(X=k) for X ~ Binomial(other_count, q)
        # (In practice, could use math.comb and q^k etc., but careful with large N and floating precision.)
        # For clarity, we'll iteratively build probabilities using the relationship: P(k) = P(k-1) * (remaining terms).
        probabilities = [0.0] * (other_count + 1)
        for k in range(other_count + 1):
            # Compute binomial probability P(X=k)
            # We can compute manually: C(n, k) * q^k * (1-q)^(n-k)
            # For stability, use math.comb for combination.
            prob = math.comb(other_count, k) * (q ** k) * ((1 - q) ** (other_count - k))
            probabilities[k] = prob
        # Now sum over all k cases:
        for k in range(other_count + 1):
            prob_k = probabilities[k]
            # Determine outcome if current juror votes A:
            # Current juror votes A, so total A votes = k (others voting A) + 1 (this juror).
            votes_A_ifA = k + 1
            votes_B_ifA = other_count - k  # since out of others, k voted A, rest voted B
            # Outcome if juror votes A:
            if votes_A_ifA >= majority_needed:  # A wins
                # If attack on and outcome is A wins, attacker pays those who voted B (not this juror, since juror voted A, so irrelevant for this juror's payoff).
                # Juror is on winning side.
                if self.payoff_type.lower() == "basic":
                    # deposit back + p fraction reward
                    payoff_A = self.d + (self.p * self.d)
                elif self.payoff_type.lower() == "redistributive":
                    # winners share losers' deposits equally
                    winners = votes_A_ifA
                    losers = votes_B_ifA
                    # If losers = 0 (everyone voted A), each gets just their deposit (no extra, since no losers to take from).
                    if losers == 0 or winners == 0:
                        payoff_A = self.d
                    else:
                        payoff_A = self.d + (losers * self.d) / winners
                elif self.payoff_type.lower() == "symbiotic":
                    winners = votes_A_ifA
                    losers = votes_B_ifA
                    # If unanimous (losers=0), everyone gets deposit + external reward (no redistribution needed).
                    if losers == 0:
                        payoff_A = self.d + self.external_reward
                    else:
                        # Losers lose beta*d each. Total redistributed = losers * beta*d.
                        payoff_A = self.d + (losers * self.beta * self.d) / winners + self.external_reward
                # (If needed, ensure payoff_A is at least d in cases where losers=0 etc., but above covers it.)
            else:  # B wins, juror voted A and is in minority
                if self.payoff_type.lower() == "basic":
                    payoff_A = 0.0  # lost deposit
                elif self.payoff_type.lower() == "redistributive":
                    payoff_A = 0.0  # lost deposit
                elif self.payoff_type.lower() == "symbiotic":
                    # Loser keeps (1-beta)*d perhaps.
                    payoff_A = (1 - self.beta) * self.d
                # If attack is on and B wins, the juror voted A so they get no bribe (attacker doesn't pay those who voted A).
                # payoff_A as above already reflects losing scenario.
            # Determine outcome if current juror votes B:
            # Total A votes = k (since juror votes B), total B votes = (other_count - k) + 1.
            votes_A_ifB = k
            votes_B_ifB = other_count - k + 1
            if votes_B_ifB >= majority_needed:  # B wins
                # Juror voted B and is on winning side.
                if self.payoff_type.lower() == "basic":
                    payoff_B = self.d + (self.p * self.d)
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
                # Attacker doesn't pay if B wins (no failed attack).
            else:  # A wins, juror voted B and is minority
                if self.payoff_type.lower() == "basic":
                    payoff_B = 0.0  # normally they'd lose deposit
                elif self.payoff_type.lower() == "redistributive":
                    payoff_B = 0.0  # they'd lose deposit
                elif self.payoff_type.lower() == "symbiotic":
                    payoff_B = (1 - self.beta) * self.d  # they'd lose part of deposit
                # But if attack is on and B loses, attacker pays deposit + epsilon to those who voted B.
                if self.attack:
                    payoff_B = self.d + self.epsilon  # compensate their loss and add bonus
            # Accumulate expectation
            exp_payoff_A += prob_k * payoff_A
            exp_payoff_B += prob_k * payoff_B
        return exp_payoff_A, exp_payoff_B

    def simulate_once(self) -> Tuple[str, int, int]:
        """
        Simulate a single round. Assign beliefs, mark bribed jurors, collect votes, and determine the outcome.
        Returns:
            outcome (str): Winning outcome ("A" or "B").
            votes_for_A (int)
            votes_for_B (int)
        """
        # 1. Assign beliefs to each juror (based on probability p for believing A).
        for juror in self.jurors:
            juror.bribed = False  # reset bribed status each round
            juror.belief = "A" if random.random() < self.p else "B"
        # 2. Mark a fraction of jurors as bribed (if attack on).
        if self.attack and self.bribed_fraction > 0:
            num_bribed = round(self.bribed_fraction * self.N)
            # Randomly choose that many jurors to be bribed
            bribed_indices = random.sample(range(self.N), min(num_bribed, self.N))
            for idx in bribed_indices:
                self.jurors[idx].bribed = True
        # 3. Each juror decides their vote.
        votes = []
        for i, juror in enumerate(self.jurors):
            # If juror is bribed, they'll vote B without needing payoff calc (our Juror.decide_vote handles it too).
            # Otherwise, compute expected payoffs for voting A vs B for this juror.
            expA, expB = self._expected_payoffs(i)
            vote = juror.decide_vote(expA, expB)
            votes.append(vote)
        # 4. Count votes for A and B.
        votes_for_A = votes.count("A")
        votes_for_B = votes.count("B")
        # 5. Determine outcome.
        outcome = "A" if votes_for_A >= votes_for_B else "B"  # if tie, A wins by this rule; adjust if needed.
        return outcome, votes_for_A, votes_for_B

    def run_simulations(self, num_simulations: int) -> dict:
        """
        Run multiple simulation rounds and compute aggregate statistics.
        Returns a dictionary with results summary.
        """
        outcomes = {"A": 0, "B": 0}
        votes_A_list = []
        votes_B_list = []
        for _ in range(num_simulations):
            outcome, votes_A, votes_B = self.simulate_once()
            outcomes[outcome] += 1
            votes_A_list.append(votes_A)
            votes_B_list.append(votes_B)
        # Compute attack success rate if applicable
        attack_success_rate = None
        if self.attack:
            # Attack succeeds when outcome is B (the attacker’s desired outcome)
            attack_success_rate = outcomes["B"] / num_simulations
        # Average votes for A and B
        avg_votes_for_A = mean(votes_A_list)
        avg_votes_for_B = mean(votes_B_list)
        return {
            "total_runs": num_simulations,
            "outcome_counts": outcomes,
            "attack_success_rate": attack_success_rate,
            "average_votes_A": avg_votes_for_A,
            "average_votes_B": avg_votes_for_B
        }