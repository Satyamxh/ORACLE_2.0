import random
import math

class Juror:
    """Represents a juror in the Schelling oracle simulation.
    
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
        self.belief = None    # Will be set each round
        self.bribed = False   # Will be set each round if this juror is bribed
        self.vote = None      # The juror's vote in the current round (set in decide_vote)

    def decide_vote(self, exp_payoff_X: float, exp_payoff_Y: float) -> str:
        """
        Decide the vote ("X" or "Y") for this juror given the expected payoffs for voting X vs Y.
        Accounts for honesty (voting true belief), rationality (voting best perceived payoff), 
        and noise (random payoff misperception).
        """
        # With probability equal to honesty, vote sincerely (according to own belief)
        if random.random() < self.honesty:
            self.vote = self.belief
            return self.belief
        # Otherwise, consider payoff-maximizing vote with noise in perception
        perceived_X = exp_payoff_X + random.gauss(0, self.noise)
        perceived_Y = exp_payoff_Y + random.gauss(0, self.noise)
        # Determine which vote has higher perceived payoff
        if math.isclose(perceived_X, perceived_Y, rel_tol=1e-9):
            # If payoffs are essentially equal, default to voting own belief
            best_vote = self.belief
        else:
            best_vote = "X" if perceived_X > perceived_Y else "Y"
        # Follow the best perceived option with probability = rationality, otherwise deviate
        if random.random() < self.rationality:
            self.vote = best_vote
            return best_vote
        else:
            # With (1 - rationality) chance, vote the opposite of the perceived best (irrational choice)
            self.vote = "X" if best_vote == "Y" else "Y"
            return self.vote
