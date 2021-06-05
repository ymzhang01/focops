import numpy as np
import torch
from utils import torch_to_numpy


class DataGenerator:
    """
    A data generator used to collect trajectories for on-policy RL with GAE
    References:
        https://github.com/Khrylx/PyTorch-RL
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, obs_dim, act_dim, batch_size, max_eps_len):

        # Hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len

        # Batch buffer
        self.obs_buf = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((batch_size, act_dim),  dtype=np.float32)
        self.rew_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cost_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.vtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cvtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cadv_buf = np.zeros((batch_size, 1), dtype=np.float32)

        self.eps_info = {'eps_len': [], 'terminal': [], 'cost_ret': []}


    def run_traj(self, env, policy, value_net, cvalue_net, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device, constraint):

        batch_idx = 0

        while batch_idx < self.batch_size:
            obs = env.reset()
            if running_stat is not None:
                obs = running_stat.normalize(obs)
            ret_eps = 0
            cost_ret_eps = 0
            num_eps = 0

            for t in range(self.max_eps_len):
                act = policy.get_act(torch.Tensor(obs).to(dtype).to(device))
                act = torch_to_numpy(act).squeeze()
                next_obs, rew, done, info = env.step(act)


                if constraint == 'velocity':
                    if 'y_velocity' not in info:
                        cost = np.abs(info['x_velocity'])
                    else:
                        cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                elif constraint == 'circle':
                    cost = info['cost']

                ret_eps += rew
                cost_ret_eps += (c_gamma ** t) * cost

                if running_stat is not None:
                    next_obs = running_stat.normalize(next_obs)

                obs = next_obs

                batch_idx += 1

                if done or t == self.max_eps_len - 1:
                    if done:
                        self.eps_info['terminal'].append(1)
                    else:
                        self.eps_info['terminal'].append(0)
                    score_queue.append(ret_eps)
                    cscore_queue.append(cost_ret_eps)
                    self.eps_info['cost_ret'].append(cost_ret_eps)

                if done or batch_idx == self.batch_size:
                    if batch_idx == self.batch_size:
                        self.eps_info['terminal'].append(0)
                    self.eps_info['eps_len'].append(t + 1)
                    num_eps += 1
                    break

        # Get advantage values
        start_idx, end_idx = 0, 0
        for eps in range(num_eps):
            eps_len = self.eps_info['eps_len'][eps]
            terminal = self.eps_info['terminal'][eps]
            end_idx = start_idx + eps_len
            obs_eps = self.obs_buf[start_idx:end_idx]
            next_obs_eps = self.next_obs_buf[start_idx:end_idx]
            rew_eps = self.rew_buf[start_idx:end_idx]
            cost_eps = self.cost_buf[start_idx:end_idx]
            self.adv_buf[start_idx:end_idx], self.vtarg_buf[start_idx:end_idx] \
                = self.get_advantage(obs_eps, next_obs_eps, rew_eps, terminal,
                                     value_net, gamma, gae_lam, dtype, device)
            self.cadv_buf[start_idx:end_idx], self.cvtarg_buf[start_idx:end_idx] \
                = self.get_advantage(obs_eps, next_obs_eps, cost_eps, terminal,
                                     cvalue_net, c_gamma, c_gae_lam, dtype, device)

            start_idx = end_idx

        # Normalize advantage
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean()) / (self.cadv_buf.std() + 1e-6)



        avg_cost = np.mean(self.eps_info['cost_ret'])


        # Normalize advantage functions
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean()) / (self.cadv_buf.std() + 1e-6)

        return {'states':self.obs_buf, 'actions':self.act_buf,
                'v_targets': self.vtarg_buf,'advantages': self.adv_buf,
                'cv_targets': self.cvtarg_buf, 'c_advantages': self.cadv_buf, 'avg_cost': avg_cost}


    def get_advantage(self, obs_eps, next_obs_eps, rew_eps, terminal, value_net, gamma, gae_lam, dtype, device, mode='reward'):
        eps_len = obs_eps.shape[0]
        gae_delta = np.zeros((eps_len, 1))
        adv_eps =  np.zeros((eps_len, 1))
        # Check if terminal state, if terminal V(S_T) = 0, else V(S_T)
        status = np.ones((eps_len, 1))
        status[-1] = 1 - terminal
        prev_adv = 0

        for t in reversed(range(eps_len)):
            # Get value for current and next state
            obs_tensor = torch.Tensor(obs_eps[t]).to(dtype).to(device)
            next_obs_tensor = torch.Tensor(next_obs_eps[t]).to(dtype).to(device)
            current_val, next_val = torch_to_numpy(value_net(obs_tensor), value_net(next_obs_tensor))

            # Calculate delta and advantage
            gae_delta[t] = rew_eps[t] + gamma * next_val * status[t] - current_val
            adv_eps[t] = gae_delta[t] + gamma * gae_lam * prev_adv

            # Update previous advantage
            prev_adv = adv_eps[t]

        # Get target for value function
        obs_eps_tensor = torch.Tensor(obs_eps).to(dtype).to(device)
        vtarg_eps = torch_to_numpy(value_net(obs_eps_tensor)) + adv_eps


        return adv_eps, vtarg_eps

