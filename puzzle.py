import random
from tkinter import Frame, Label, CENTER

import logic
import constants as c
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
import numpy as np
import time
key_map={
0: "'w'",
1: "'s'",
2: "'a'",
3: "'d'"

}
class PolicyNet(nn.Module):
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

class Agent:
    def __init__(self,device):
        self.policy=PolicyNet(device).to(device)
        self.action_history=[]
        self.reward_history=[]
        self.device = device
    def get_action(self,state):
        prob=F.softmax(self.policy.forward(state))
        action_prob=torch.distributions.Categorical(prob)
        action=action_prob.sample()
        log_probs=action_prob.log_prob(action)
        self.action_history.append(log_probs)
        return action.item()
    def rewards_history(self,reward):
        self.reward_history.append(reward)


class GameGrid(Frame):


    def __init__(self):
        # Frame.__init__(self)
        self.score=0
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
    def init_grid(self,game):
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

    def update_grid_cells(self,game):
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
        flat=np.array(self.matrix).flatten().astype(np.float32())
        return  torch.from_numpy(flat)

    def key_down(self, event):
        key = event
        game_done=False
        game_result=False
        temp=0
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
        if(game_done and game_result ):
            self.score=sum(np.array(self.matrix).flatten())
        elif(game_done and  not(game_result)):
            self.score=-1*sum(np.array(self.matrix).flatten())
        else:
            if(self.score != sum(np.array(self.matrix).flatten()) ):

                for i in range(4):
                    for j in range(4):
                        if(self.matrix[i][j]== 2*current_state[i][j]):
                            temp+=self.matrix[i][j]
                self.score=sum(np.array(self.matrix).flatten())
                return self.matrix,game_done,temp

            else:

                # print(0)
                return self.matrix, game_done, -100

        return self.matrix,game_done,self.score
    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent=Agent(device)
lr=0.001
gamma=0.9999
optimizer = optim.Adam(params=agent.policy.parameters(), lr=lr)



score=0
episode=10000
result=[]

for i in range(episode):

    gamegrid = GameGrid()
    gamegrid.render()
    step=0
    done = False
    for timestamp in count():

        current_state=gamegrid.get_state()

        action=agent.get_action(current_state)

        next_state,done ,reward=gamegrid.key_down(key_map[action])
        print(reward)
        agent.rewards_history(reward)

        score+=reward

        gamegrid.render()
        if(timestamp==2000):
            print("******************************FORCEED*******************************")
            done=True
        if done:
            gamegrid.game.     destroy()
            optimizer.zero_grad()
            G=np.zeros_like(agent.reward_history,dtype=np.float32)
            for t in range(len(agent.reward_history)):
                total_G=0
                dis=1
                for k in range(t,len(agent.reward_history)):
                    total_G+=agent.reward_history[k]*dis
                    dis*=gamma
                G[t]=total_G
            mean=np.mean(G)
            std=np.std(G)
            if(std==0):
                std=1
            G=(G-mean)/std

            G=torch.tensor(G).to(agent.device)
            loss=0
            for g,logprob in zip(G,agent.action_history):
                loss+= -g * logprob
            loss.backward()
            optimizer.step()
            agent.action_history=[]
            agent.reward_history=[]
            result.append(score)
            if(score>0):
              print("Episode ",i,"Score %.3f " %score,step)
            if((i % 500)==0):
                print("Episode ", i, "Score %.3f " % score, step)
            score=0
            break

