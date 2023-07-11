import gym
import numpy as np
from stable_baselines3 import PPO
import pygame
import os
import numpy as np
import pickle
import random  

class GameOfLifeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = 30
        self.board = np.zeros((self.board_size, self.board_size))
        self.action_space = gym.spaces.MultiDiscrete([2] * (self.board_size * self.board_size))
        self.observation_space = gym.spaces.MultiDiscrete([2] * (self.board_size * self.board_size))
        self.t = 0
        self.max_time = np.inf
        self.save_state = None
        self.last_states = []
        self.done = False

        # Pygame setup
        self.cell_size = 10
        self.screen = pygame.display.set_mode((self.board_size * self.cell_size, self.board_size * self.cell_size))

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.t = 0
        self.done = False
        return self.board.flatten()

    def step(self, action):
        # print(self.board)
        
        action = action.reshape((self.board_size, self.board_size))  # Reshape the action
        self.last_states.append(self.board.copy())  # Save the previous state

        if self.t == 0:
            self.board = action  # Set the board to the agent's action at the start of the episode
            self.save_state = action
        elif self.t > 0:
            self.board = self.update_board(self.board)


        # reward = np.sum(self.board)  # Reward for each step
        if self.t > 1:
            self.last_states.pop(0)
            self.done = np.array_equal(self.board, self.last_states[-1]) or np.array_equal(self.board, self.last_states[-2])  # End episode if the state is the same as the previous state
        reward = np.sum(self.board) if self.done else 0
            
        
        self.t += 1

        return self.board.flatten(), reward, self.done, {}

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
        pygame.time.delay(1)  # Delay in milliseconds
        
    # def update_board(self, board): # without wrapping
    #     new_board = np.zeros_like(board)
    #     for i in range(board.shape[0]):
    #         for j in range(board.shape[1]):
    #             # Count the number of live neighbors
    #             live_neighbors = np.sum(
    #                 board[max(i - 1, 0) : min(i + 2, board.shape[0]), max(j - 1, 0) : min(j + 2, board.shape[1])]
    #             ) - board[i, j]

    #             # Apply the Game of Life rules
    #             if board[i, j] == 1 and (live_neighbors == 2 or live_neighbors == 3):
    #                 new_board[i, j] = 1
    #             elif board[i, j] == 0 and live_neighbors == 3:
    #                 new_board[i, j] = 1
    #     return new_board

    def update_board(self, board): # with wrapping
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
    
    def save_initial_state(self, path):
        # Save the initial state to a file
        with open(path, 'wb') as f:
            pickle.dump(self.save_state, f)

    def load_initial_state(self, path):
        # Load the initial state from a file
        with open(path, 'rb') as f:
            self.initial_state = pickle.load(f)
        self.board = self.initial_state
        self.t = 0


env = GameOfLifeEnv()

max_eps = 6000
render_int = max_eps//450
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
highest_reward = -np.inf 

for i_episode in range(max_eps):
    observation = env.reset()
    if i_episode % render_int == 0:
        env.render()
    done = False
    episode_reward = 0  # Initialize the reward for this episode
    while not done:
        action, _states = model.predict(observation.reshape(1, -1))
        observation, reward, done, info = env.step(action)
        episode_reward += reward  # Accumulate the reward for this episode
        env.render()
        
        if env.t == 1:
            run_id = f"run_{i_episode}"  # Unique run ID based on the episode number
            filename = f"{i_episode}_initial_state.pkl"  # Filename for the pickle file
            path = f"./saved_runs/int_inf_attempt_1/{filename}"  # Full path to the file
            env.save_initial_state(path)  # Save the initial state

        if done:
            if episode_reward > highest_reward:  # If this episode achieved a new high score
                highest_reward = episode_reward  # Update the highest reward               
                os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
                rewards.append(reward)
                print(f"New best episode: episode {i_episode} had a reward of {highest_reward}")

            
            avg_reward = np.mean(rewards[-100:])  # Calculate rolling average of last 100 rewards
            print(f"Episode {i_episode} finished after {env.t} timesteps with reward {reward}, average reward: {avg_reward}")


    # Call the learn() method to update the policy
    if i_episode % 5 == 0 and i_episode != 0:
        model.learn(total_timesteps=(n_step_learn_rate))

    if i_episode == max_eps -1:
        run_id = "first_run_5k"  # Replace with your unique run ID
        filename = "initial_state.pkl"  # Choose a filename for the pickle file
        path = f"./saved_runs/{run_id}"
        env.save_initial_state(path)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            pygame.display.flip()
            pygame.time.delay(200)  # Delay in milliseconds
