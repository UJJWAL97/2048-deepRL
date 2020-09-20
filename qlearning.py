import random
from tkinter import Frame, Label, CENTER
from collections import namedtuple
import logic
import constants as c
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
import numpy as np
import time
import math
key_map = {
    0: "'w'",
    1: "'s'",
    2: "'a'",
    3: "'d'"

}

class DQN(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.fc1=nn.Linear(in_features=16,out_features=128)
        self.fc2=nn.Linear(in_features=128,out_features=128)
        self.fc3=nn.Linear(in_features=128,out_features=64)
        self.out=nn.Linear(in_features=64,out_features=4)
        self.device=device
    def forward(self,t):
        t=torch.tensor(t).to(self.device)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                temp = (policy_net(state).argmax(dim=0).to(self.device))
                temp=temp.cpu()
                temp = temp.numpy()
                temp = int(temp)
                return torch.tensor([temp]).to(self.device)  # exploit

class GameGrid(Frame):

    def __init__(self):
        # Frame.__init__(self)
        self.score = 0
        self.game = Frame()
        self.game.grid()
        self.game.master.title('2048')
        # self.master.bind("<Key>", self.key_down)

        # self.gamelogic = gamelogic
        self.game.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                              c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
                              c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
                              c.KEY_LEFT_ALT: logic.left, c.KEY_RIGHT_ALT: logic.right,
                              c.KEY_H: logic.left, c.KEY_L: logic.right,
                              c.KEY_K: logic.up, c.KEY_J: logic.down}

        self.game.grid_cells = []
        self.init_grid(self.game)
        self.init_matrix()
        self.update_grid_cells(self.game)

    def render(self):

        self.game.update_idletasks()
        self.game.update()
        time.sleep(0.01)

        time.sleep(0.01)

    def init_grid(self, game):
        background = Frame(game, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            game.grid_cells.append(grid_row)

    def gen(self):
        return random.randint(0, c.GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = list()
        self.matrix = logic.add_two(self.matrix)
        self.matrix = logic.add_two(self.matrix)

    def update_grid_cells(self, game):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    game.grid_cells[i][j].configure(
                        text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    game.grid_cells[i][j].configure(text=str(
                        new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number])
        game.update_idletasks()

    def num_actions_available(self):
        return 4

    def get_state(self):
        flat = np.array(self.matrix).flatten().astype(np.float32())
        return torch.from_numpy(flat)

    def key_down(self, event):
        key = event
        game_done = False
        game_result = False
        temp = 0
        current_state = self.matrix

        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))

        elif key in self.game.commands:
            self.matrix, done = self.game.commands[key](self.matrix)

            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells(self.game)

                done = False
                if logic.game_state(self.matrix) == 'win':
                    game_done = True
                    game_result = True
                    print("WON")
                    self.game.grid_cells[1][1].configure(
                        text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.game.grid_cells[1][2].configure(
                        text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    game_done = True
                    game_result = False
                    self.game.grid_cells[1][1].configure(
                        text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.game.grid_cells[1][2].configure(
                        text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        if (game_done and game_result):
            self.score = sum(np.array(self.matrix).flatten())
        elif (game_done and not (game_result)):
            self.score = -1 * sum(np.array(self.matrix).flatten())
        else:
            if (self.score != sum(np.array(self.matrix).flatten())):

                for i in range(4):
                    for j in range(4):
                        if (self.matrix[i][j] == 2 * current_state[i][j]):
                            temp += self.matrix[i][j]
                self.score = sum(np.array(self.matrix).flatten())
                return self.matrix, game_done, temp

            else:

                # print(0)
                return self.matrix, game_done, -1

        return self.matrix, game_done, self.score

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.stack(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.next_state)

    return (t1, t2, t3, t4)
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        temp = policy_net(states)
        print("c111111111111111111111111111111")
        print(actions.shape)
        print(temp)
        return temp.gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        batch_size = next_states.shape[0]
        non_final_state_locations = torch.zeros(batch_size, dtype=torch.bool).to(QValues.device)
        for i in range(len(next_states)):
            if (1024 in next_states[i]):
                non_final_state_locations[i] = False
            else:
                non_final_state_locations[i] = True
        non_final_states = next_states[non_final_state_locations]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()

        return values



batch_size = 1
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 100
memory_size = 100000
lr = 0.1
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

agent = Agent(strategy, 4, device)
memory = ReplayMemory(memory_size)

policy_net = DQN(device).to(device)
target_net = DQN(device).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []

for episode in range(num_episodes):
    em = GameGrid()
    state = em.get_state()
    for timestep in count():
        em.render()
        action = agent.select_action(state, policy_net)
        next_state, done, reward = em.key_down(key_map[action.item()])
        next_state=em.get_state()
        memory.push(Experience(state, action, next_state, torch.tensor([reward], device=device,dtype=torch.float32)))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done: 
            em.game.destroy()
            episode_durations.append(timestep)
            print(timestep)

            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())






