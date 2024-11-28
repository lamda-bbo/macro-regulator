import torch as th
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import time
from torchvision.models import resnet18
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from model.cnn import MyCNN, MyCNNCoarse
from model.actor import Actor
from model.critic import Critic


from utils.debug import *

class PPOAgent():
    def __init__(self, args):
        self.args = args
        self.resnet = resnet18(pretrained=True)
        self.cnn = MyCNN().to(args.device)
        self.cnn_coarse = MyCNNCoarse(args=args, res_net=self.resnet).to(args.device)

        self.actor = Actor(args=args, cnn=self.cnn, cnn_coarse=self.cnn_coarse).float().to(args.device)
        self.critic = Critic(args=args, cnn_coarse=self.cnn_coarse).float().to(args.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), args.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), args.lr)

        self.buffer = []
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_size * args.n_macro
        self.counter = 0
        self.training_step = 0

        self.last_log_episode = -args.log_interval - 1

        self.training = False
    
    def select_action(self, state:np.array):
        state = th.from_numpy(state).float().to(self.args.device).unsqueeze(0)
        with th.no_grad():
            action_prob = self.actor(state)
        dist = Categorical(action_prob)
        if self.training:
            action = dist.sample()
        else:
            action = dist.probs.argmax()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def update(self, t_env):
        if self.counter % self.buffer_capacity == 0:
            t_start = time.time()
            state = th.tensor(np.array([t.state for t in self.buffer]), dtype=th.float32)
            action = th.tensor(np.array([t.action for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            reward = th.tensor(np.array([t.reward for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            old_action_log_prob = th.tensor(np.array([t.action_log_prob for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            done = th.tensor(np.array([t.done for t in self.buffer], dtype=np.int32), dtype=th.int32).view(-1, 1).to(self.args.device)
            del self.buffer[:]
            
            target_list = []
            target = 0
            for i in range(reward.shape[0]-1, -1, -1):
                if done[i, 0] == 1:
                    target = 0
                r = reward[i, 0].item()
                target = r + self.args.gamma * target
                target_list.append(target)
            target_list.reverse()
            target_v_all = th.tensor(np.array([t for t in target_list]), dtype=th.float32).view(-1, 1).to(self.args.device)
            
            actor_loss_lst  = []
            critic_loss_lst = []
            for _ in range(self.args.epoch):
                for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                    disable = self.args.disable_tqdm):
                    self.training_step += 1

                    action_prob = self.actor(state[index].to(self.args.device))
                    dist = Categorical(action_prob)
                    action_log_prob = dist.log_prob(action[index].squeeze())
                    ratio = th.exp(action_log_prob - old_action_log_prob[index].squeeze())
                    target_v = target_v_all[index]
                    critic_output = self.critic(state[index].to(self.args.device))
                    advantage = (target_v - critic_output).detach()

                    L1 = ratio * advantage.squeeze() 
                    L2 = th.clamp(ratio, 1-self.args.clip_param, 1+self.args.clip_param) * advantage.squeeze() 
                    action_loss = -th.min(L1, L2).mean() 

                    self.actor_opt.zero_grad()
                    action_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                    self.actor_opt.step()

                    value_loss = nn.functional.smooth_l1_loss(self.critic(state[index].to(self.args.device)), target_v)
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                    self.critic_opt.step()

                    actor_loss_lst.append(action_loss.cpu().item())
                    critic_loss_lst.append(value_loss.cpu().item())
            
            t_end = time.time()

            if self.args.i_episode - self.last_log_episode >= self.args.log_interval:
                self.args.logger.add('actor_loss', np.mean(actor_loss_lst), t_env)
                self.args.logger.add('critic_loss', np.mean(critic_loss_lst), t_env)
                self.last_log_episode = self.args.i_episode
            
            return t_end - t_start
        else:
            return 0

    def store_transition(self, transition):
        self.counter += 1
        self.buffer.append(transition)

    def save_model(self, gp_hpwl, benchmark=None):
        if benchmark is None:
            base_path = os.path.join(os.path.dirname(os.path.abspath('.')), 'results', 'save_model', 
                                    self.args.name, self.args.unique_token, "average_best")
        else:
            base_path = os.path.join(os.path.dirname(os.path.abspath('.')), 'results', 'save_model', 
                                    self.args.name, self.args.unique_token, benchmark)
        file_name = f'{self.args.i_episode}_{self.args.t_env}_{gp_hpwl}.pkl'
        os.makedirs(base_path, exist_ok=True)
        th.save(
            {
                'actor' : self.actor.state_dict(),
                'critic' : self.critic.state_dict()
            },
            os.path.join(base_path, file_name)
        )
    
    def load_model(self, checkpoint_path):
        checkpoint = th.load(checkpoint_path, map_location=th.device(self.args.device))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
    
    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

