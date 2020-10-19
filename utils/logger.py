import os
import pickle
import torch
from utils.misc import println

class Logger:

    def __init__(self, hyperparams):

        self.log_data = {'time': 0,
                         'MinR': [],
                         'MaxR': [],
                         'AvgR': [],
                         'MinC': [],
                         'MaxC': [],
                         'AvgC': [],
                         'nu': [],
                         'running_stat': None}

        self.models = {'iter': None,
                       'policy_params': None,
                       'value_params': None,
                       'cvalue_params': None,
                       'pi_optimizer': None,
                       'vf_optimizer': None,
                       'cvf_optimizer': None,
                       'pi_loss': None,
                       'vf_loss': None,
                       'cvf_loss': None}

        self.hyperparams = hyperparams
        self.iter = 0



    def update(self, key, value):
        if type(self.log_data[key]) is list:
            self.log_data[key].append(value)
        else:
            self.log_data[key] = value

    def save_model(self, component, params):
        self.models[component] = params


    def dump(self):
        batch_size = self.hyperparams['batch_size']
        # Print results
        println('Results for Iteration:', self.iter + 1)
        println('Number of Samples:', (self.iter + 1) * batch_size)
        println('Time: {:.2f}'.format(self.log_data['time']))
        println('MinR: {:.2f}| MaxR: {:.2f}| AvgR: {:.2f}'.format(self.log_data['MinR'][-1],
                                                                  self.log_data['MaxR'][-1],
                                                                  self.log_data['AvgR'][-1]))
        println('MinC: {:.2f}| MaxC: {:.2f}| AvgC: {:.2f}'.format(self.log_data['MinC'][-1],
                                                                  self.log_data['MaxC'][-1],
                                                                  self.log_data['AvgC'][-1]))
        println('Nu: {:.3f}'.format(self.log_data['nu'][-1]))
        println('--------------------------------------------------------------------')


        # Save Logger
        env_id = self.hyperparams['env_id']
        constraint = self.hyperparams['constraint']
        seed = self.hyperparams['seed']
        envname = env_id.partition(':')[-1] if ':' in env_id else env_id


        directory = '_'.join(['focops', 'results'])
        filename1 = '_'.join(['focops', constraint, envname, 'log_data_seed', str(seed)]) + '.pkl'
        filename2 = '_'.join(['focops', constraint, envname, 'hyperparams_seed', str(seed)]) + '.pkl'
        filename3 = '_'.join(['focops', constraint, envname, 'models_seed', str(seed)]) + '.pth'

        if not os.path.exists(directory):
            os.mkdir(directory)

        pickle.dump(self.log_data, open(os.path.join(directory, filename1), 'wb'))
        pickle.dump(self.hyperparams, open(os.path.join(directory, filename2), 'wb'))
        torch.save(self.models, os.path.join(directory, filename3))

        # Advance iteration by 1
        self.iter += 1
