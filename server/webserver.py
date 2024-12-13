from flask import Flask, render_template
import os
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from gymnasium.utils.save_video import save_video

import copy
import random
import time


class Breakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array_list', repeat=4, device='cpu'):
        env = gym.make('ALE/Breakout-v5', render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        super(Breakout, self).__init__(env)
        
        self.image_shape = (84,84)
        self.repeat = repeat
        self.lives = 5
        self.frame_buffer = []
        self.device = device
        
    def step(self, action):
        total_reward = 0
        done = False
        
        for i in range(self.repeat):
            observation, reward, done, truncacted, info = self.env.step(action)
            
            total_reward += reward
            
            current_lives = info['lives']
            
            if current_lives < self.lives:
                total_reward = total_reward - 1
                self.lives = current_lives
                
            self.frame_buffer.append(observation)
            
            if done:
                break
        
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)
        
        total_reward = torch.tensor(total_reward).view(1,-1).float()
        total_reward = total_reward.to(self.device)
        
        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)
        
        return max_frame, total_reward, done, info, observation
    
    def reset(self):
        self.frame_buffer = []
        
        observation, _ = self.env.reset()
        image = observation.copy()
        
        self.lives = 5
        
        observation = self.process_observation(observation)

        return observation, image
    
    def process_observation(self, observation):
        img = Image.fromarray(observation).resize(self.image_shape).convert("L")
        img = torch.from_numpy(np.array(img))
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(self.device)
        
        return img/255.0
    
class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.device = device
        
    def insert(self, transition):
        transition = [item.to('cpu') for item in transition]
        
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)
    
    def sample(self, batch_size=64):
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10
    
class Model(nn.Module):
    def __init__(self, nb_action=4):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1))
        
        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_action)
        
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        
        s = self.dropout(self.relu(self.state_value1(x)))
        s = self.dropout(self.relu(self.state_value2(s)))
        s = self.relu(self.state_value3(s))

        a = self.dropout(self.relu(self.action_value1(x)))
        a = self.dropout(self.relu(self.action_value2(a)))
        a = self.relu(self.action_value3(a))

        output = s + (a - a.mean())
        
        return output
    
    def save_the_model(self, weights_filename='models/latest.pt'):
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(self.state_dict(), weights_filename)
        
    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename, map_location=device))
            print("Loaded weights file")
        except:
            print("No weights file")

def f(episode_id: int) -> bool:
    return True

class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, action_size=None, learning_rate=1e-5):
        self.memory = ReplayMemory(device=device, capacity=600000)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / 5000) * 2)
        self.batch_size = 64
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.action_size = action_size
        
        self.optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
        
    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.action_size, (1,1)), None
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True), av
        
    def train(self, env, epochs):
        
        for epoch in range(1,epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0
            
            while not done:
                action = self.get_action(state)
                
                next_state, reward, done, _ = env.step(action)
                
                self.memory.insert([state, action, reward, done, next_state])
                
                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1,action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                state = next_state
                ep_return += reward.item()
            
            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay
            
            if epoch  % 10 == 0:
                self.model.save_the_model()

            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

                save_video(
                env.render(),
                "videos",
                episode_trigger=f,
                fps=24,
                step_starting_index=0,
                episode_index=epoch)                
                
            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")
    
    def test(self, env):
        for epoch in range(1,3):
            state = env.reset()
            done = False
            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break

os.environ['KMP_DUPLICATE_OK'] = 'TRUE'

device = torch.device('cpu')

model = Model(nb_action=4).to(device)

model.load_the_model()

user_agent = Agent(model=model,
              device=device,
              epsilon=0,
              action_size=4,
              learning_rate=1e-5)

ai_agent = Agent(model=model,
              device=device,
              epsilon=0.05,
              action_size=4,
              learning_rate=1e-5)

app = Flask(__name__)

user_environment = Breakout(device=device)
state = [0]
image = [0]
state[0],image[0] = user_environment.reset()
image[0] = Image.fromarray(image[0]).resize((320,420))
image[0].save('static/images/img.png','JPEG')
timestep = [0]
next_state = [0]
ai_environment = [copy.deepcopy(user_environment)]
Q_list = []
before_state = [state[0]]
first_done = [False]

@app.route('/')
def home():
    goodAt = []
    Q_list_clone = Q_list.copy()
    for i in range(5):
        first = Q_list_clone.index(sorted(Q_list_clone[20:-20])[-1])
        Q_list_clone = Q_list_clone[:first - 10] + Q_list_clone[first + 10:]
        goodAt.append(first)
    goodAt = np.array(sorted(goodAt))
    return render_template('index2.html', a='You did good at '+ str(goodAt*(14/114)) + ' seconds!!') 
@app.route('/<name>')
def user(name):
    if int(name) <=3:
        action, Q = user_agent.get_action(before_state[0])
        state[0],reward,done,info,image[0] = user_environment.step(int(name))
        if done == True:
            first_done[0] = True
        if first_done[0] == False:
            before_state[0] = state[0]
            Q_list.append(Q[0,int(name)])
        timestep[0] += 1
        if timestep[0] <= 100:
            _0,_1,_2,_3,_4 = ai_environment[0].step(int(name))
        if timestep[0] == 100:
            next_state[0] = state[0]

        image[0] = Image.fromarray(image[0]).resize((320,420))
        image[0].save('static/images/img.png','JPEG')
        return render_template('index.html', a='hi') 
    
    elif int(name) == 4:
        save_video(
        user_environment.render(),
        "static/videos",
        episode_trigger=f,
        fps=32,
        step_starting_index=0,
        episode_index=0)
        
        done = False
        while done == False:
            action, Q = ai_agent.get_action(next_state[0])
            state[0], reward, done, info,image[0] = ai_environment[0].step(action)
            next_state[0] = state[0]

        save_video(
        ai_environment[0].render(),
        "static/videos",
        episode_trigger=f,
        fps=32,
        step_starting_index=0,
        episode_index=1)
        
        return render_template('index.html', a='hi') 
    elif int(name) == 5:
        return render_template('index.html', a='hi') 
    elif int(name) == 6:
        state[0],image[0] = user_environment.reset()
        image[0] = Image.fromarray(image[0]).resize((320,420))
        image[0].save('static/images/img.png','JPEG')
        timestep[0] = 0
        next_state[0] = 0
        ai_environment[0] = copy.deepcopy(user_environment)
        Q_list.clear()
        before_state[0] = state[0]
        first_done[0] = False
        return render_template('index.html', a='hi') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)