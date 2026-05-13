import numpy as np


class QAgent:
    """
    Q-Learning agent with epsilon-greedy action selection, state bucketing,
    and epsilon decay functionality.
    """

    def __init__(
        self,
        n_actions,
        lr=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.999995,
        min_epsilon=0.01,
        use_decay=True,
    ):
        """
        Initialize the Q-Learning agent.

        :param n_actions: Number of possible actions (planes).
        :param lr: Learning rate (alpha).
        :param gamma: Discount factor for future rewards.
        :param epsilon: Initial exploration rate.
        :param epsilon_decay: Multiplicative factor to reduce epsilon after each episode/step.
        :param min_epsilon: The floor for epsilon (agent will always explore at least this much).
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = n_actions
        self.use_decay = use_decay
        self.q_table = {}

    def _bucket_state(self, state):
        """
        Discretizes continuous time into 10min blocks to limit the state space size.

        :param state: Raw state from the environment.
        :return: A hashed, bucketed version of the state.
        """
        times, locs, f_idx = state
        return (tuple(round(t, -1) for t in times), locs, f_idx)

    def choose_action(self, state, use_epsilon=True):
        """
        Selects an action using epsilon-greedy logic.
        """
        s = self._bucket_state(state)
        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.n_actions)

        if use_epsilon and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[s])

    def decay_epsilon(self):
        """
        Reduces the exploration rate according to epsilon_decay.
        Call this at the end of each episode.
        """
        if self.use_decay:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Bellman equation.
        """
        s, s_next = self._bucket_state(state), self._bucket_state(next_state)

        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.n_actions)
        if s_next not in self.q_table:
            self.q_table[s_next] = np.zeros(self.n_actions)

        # Q-Learning update rule
        target = reward + (0 if done else self.gamma * np.max(self.q_table[s_next]))
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])
