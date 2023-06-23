# goal is to use teacher forcing with the trained model. I'll be using my previous SNS implementation and starting with cartpole
import argparse
import os
import gymnasium as gym

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from sns_utils import SNSCell, Agent

print(os.path.dirname(os.path.realpath(__file__)))
class Actor(nn.Module):
    def __init__(self, inp_size,hid_size,out_size):
        super(Actor, self).__init__()
        self.hid_size = hid_size
        self.inp = nn.Linear(inp_size,hid_size)
        self.sig = nn.Tanh()
        self.sns = SNSCell(hid_size)
        self.out = nn.Linear(hid_size,out_size)
        self.hidden = torch.zeros(self.hid_size)
        self.reset()

    def forward(self, x, hidden=None):
        x = self.inp(x)
        x = self.sig(x)
        if hidden is None:
            x, hidden = self.sns(x, self.hidden)
            self.hidden = hidden.detach()
        else:
            x, hidden = self.sns(x, hidden)
        x = self.out(x)
        return x, hidden.detach()
    def reset(self):
        self.hidden = torch.zeros(1,self.hid_size)


def train_with_teacher_forcing(model, teacher, env, optimizer, criterion, max_episodes, losses = []):
    for episode in range(max_episodes):
        next_obs, _ = env.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        model.reset()
        hidden = None
        done = False
        term = False
        with tqdm(total = 500) as pbar:
            while not done and not term:
                # Perform a forward pass through the model
                output, hidden = model(next_obs, hidden)
                

                # Convert the output into a probability distribution
                probs = torch.softmax(output, dim=-1)
                
                # Choose the action with the highest probability
                action = probs.squeeze().argmax().item()
                action = output
                # Use the feed_forward model to get the action from the observation
                feed_forward_action = teacher(next_obs)
                
                
                # Use teacher forcing to update the action
                action = feed_forward_action
                
                # Perform the action
                next_obs, _, done, term , _ = env.step(action.detach().numpy())
                
                # save the observation for next time
                next_obs = torch.tensor(next_obs, dtype=torch.float32)


                # Perform a backward pass and update the model
                sqzd = output[0]
                loss = criterion(sqzd, action.clone().detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                pbar.update(1)
        

        if (episode + 1) % 2 == 0:
            print(f"Episode: {episode+1}, Loss: {loss.item()}")
            
if __name__ == "__main__":            
    # Initialize the environment
    env = gym.make('CartPole-v1')

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Set the hyperparameters
    input_size = env.observation_space.shape[0]
    hidden_size = 1
    output_size = env.action_space.n
    learning_rate = 0.001
    max_episodes = 100

    # Create an instance of both models
    model = Actor(input_size, hidden_size, output_size)

    teacher = Agent(env,hidden_size=8)
    teacher.load_state_dict(torch.load("SNS-Toolbox-Optimization/cloned_cartpole/CartPole-v1_ppo_100_1685605595.pth"))
    teacher.eval()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set up tensorboard
    losses = []

    # Train the model using teacher forcing
    train_with_teacher_forcing(model, teacher, env, optimizer, criterion, max_episodes,losses = losses)
    env.close()
    torch.save(model.state_dict(), f'sns_trained.pth')
