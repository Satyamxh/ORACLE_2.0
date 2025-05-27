import math
import random
from statistics import mean
from typing import List, Tuple

class Juror:
    """
    Represents a juror in the Schelling oracle simulation.
    Attributes:
        honesty (float): Probability of voting sincerely (according to own belief).
        rationality (float): Likelihood of choosing the payoff-maximizing action.
        noise (float): Standard deviation of noise in payoff estimation.
        belief (str): The juror's belief about the true outcome ("A" or "B").
        bribed (bool): Whether this juror is bribed/controlled by the attacker to vote "B".
    """
    def __init__(self, honesty: float, rationality: float, noise: float):
        self.honesty = honesty
        self.rationality = rationality
        self.noise = noise
        self.belief = None  # will be set each round
        self.bribed = False  # can be set each round
    
    def decide_vote(self, exp_payoff_A: float, exp_payoff_B: float) -> str:
        """
        Decide the vote ("A" or "B") for this juror given the expected payoffs for voting A vs B.
        The decision accounts for the juror's honesty, rationality, and noise in payoff perception.
        """
        # If juror is bribed (attack-controlled), they will vote "B" no matter what.
        if self.bribed:
            return "B"
        # With probability equal to honesty, vote according to own belief (sincere vote).
        if random.random() < self.honesty:
            return self.belief  # vote what they believe is true
        # Otherwise, the juror is considering payoffs:
        # Add some noise to the expected payoffs to simulate misperception.
        perceived_A = exp_payoff_A + random.gauss(0, self.noise)
        perceived_B = exp_payoff_B + random.gauss(0, self.noise)
        # Determine which vote seems better financially.
        best_vote = "A" if perceived_A > perceived_B else "B"
        # If perceived payoffs are exactly equal (which is rare), we can break tie by favoring their belief or random.
        if math.isclose(perceived_A, perceived_B, rel_tol=1e-9):
            best_vote = self.belief  # default to their belief in a tie scenario
        # Decide whether to follow the calculated best vote or deviate, based on rationality.
        if random.random() < self.rationality:
            # Follow the best perceived payoff option.
            return best_vote
        else:
            # With (1 - rationality) chance, the juror deviates (could vote the opposite or randomly).
            # We'll choose the opposite of best_vote as an "error" decision.
            return "A" if best_vote == "B" else "B"