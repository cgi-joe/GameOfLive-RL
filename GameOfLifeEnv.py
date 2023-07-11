import gym
import numpy as np
from stable_baselines3 import PPO
import pygame
import random  

class GameOfLifeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = 20
        self.board = np.zeros((self.board_size, self.board_size))
        self.action_space = gym.spaces.MultiDiscrete([2] * (self.board_size * self.board_size))
        self.observation_space = gym.spaces.MultiDiscrete([2] * (self.board_size * self.board_size))
        self.t = 0
        self.max_time = 30#random.randint(5, 30)


        # Pygame setup
        self.cell_size = 20
        self.screen = pygame.display.set_mode((self.board_size * self.cell_size, self.board_size * self.cell_size))
        self.random_action_prob = 1.0  # Initial random action probability

        # Define epsilon and related attributes
        self.epsilon = 1.0  # Initial epsilon value
        self.epsilon_decrease = 0.01  # Amount to decrease epsilon each step
        self.epsilon_increase = 0.01  # Amount to decrease epsilon each step
        self.min_epsilon = 0.1  # Minimum epsilon value
        self.convergence_threshold = 0.01  # Threshold for determining if observations are converging
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.t = 0
        return self.board.flatten()

    def step(self, action):
        done = self.t >= self.max_time
        
        # if self.t > 0 and np.allclose(self.board.flatten(), self.prev_observation, atol=self.convergence_threshold):
        #     self.epsilon = min(self.epsilon + self.epsilon_increase, 1.0)
        # else:
        #     self.epsilon = max(self.epsilon - self.epsilon_decrease, self.min_epsilon)
            
        self.prev_observation = self.board.flatten()
        if self.t == 0:
            action = action.reshape((self.board_size, self.board_size))  # Reshape the action
            self.board = action  # Set the board to the agent's action at the start of the episode
        else:
            self.board = self.update_board(self.board)

        reward = np.sum(self.board)  # Negative reward for each step

        self.t += 1
        return self.board.flatten(), reward, done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        for i in range(self.board_size):
            for j in range(self.board_size):
                pygame.draw.rect(
                    self.screen,
                    (255, 255, 255) if self.board[i, j] == 1 else (0, 0, 0),
                    pygame.Rect(i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size),
                )

        pygame.display.flip()
        pygame.time.delay(100)  # Delay in milliseconds

    def update_board(self, board):
        new_board = np.zeros_like(board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                # Calculate indices for the neighbors with wraparound
                i_values = [(i-1)%self.board_size, i, (i+1)%self.board_size]
                j_values = [(j-1)%self.board_size, j, (j+1)%self.board_size]

                # Count the number of live neighbors
                live_neighbors = np.sum(board[i_values][:, j_values]) - board[i, j]

                # Apply the Game of Life rules
                if board[i, j] == 1 and (live_neighbors == 2 or live_neighbors == 3):
                    new_board[i, j] = 1
                elif board[i, j] == 0 and live_neighbors == 3:
                    new_board[i, j] = 1
                elif board[i, j] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                    new_board[i, j] = 0
        return new_board


env = GameOfLifeEnv()

max_eps = 6000
render_int = max_eps//500
n_step_learn_rate = max_eps//100
print_int = max_eps//250

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    n_steps=n_step_learn_rate,
    batch_size=10,
    n_epochs=10,
)

rewards = []

for i_episode in range(max_eps):
    observation = env.reset()
    if i_episode % render_int == 0:
        env.render()
    done = False
    while not done:
        action, _states = model.predict(observation.reshape(1, -1))
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if i_episode % render_int == 0:
            env.render()
        if done and (i_episode % render_int == 0):
            avg_reward = np.mean(rewards[-100:])  # Calculate rolling average of last 100 rewards
            print(f"Episode {i_episode} finished after {env.t} timesteps with reward {reward}, average reward: {avg_reward}")

    # Call the learn() method to update the policy
    if i_episode % 50 == 0 and i_episode != 0:
        model.learn(total_timesteps=(n_step_learn_rate))

    decayed_random_action_prob = max(1.0 - ((i_episode % max_eps) / max_eps), 0)
    env.random_action_prob = decayed_random_action_prob

    if i_episode == max_eps -1:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            pygame.display.flip()
            pygame.time.delay(200)  # Delay in milliseconds