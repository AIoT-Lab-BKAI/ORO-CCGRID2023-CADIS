from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule
import torch
import wandb
import time
import json

time_records = {"server_aggregation": {"overhead": [], "aggregation": []}, "local_training": {}}

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['alpha']
        self.alpha = option['alpha']
        self.h  = self.model.zeros_like()
        
    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: return
        self.model = self.aggregate(models)
        return
    
    def run(self):
        super().run()
        json.dump(time_records, open(f"./measures/{self.option['algorithm']}.json", "w"))
        return

    def aggregate(self, models):
        
        start = time.time()
        self.h = self.h - self.alpha * (1.0 / self.num_clients * fmodule._model_sum(models) - self.model)
        end = time.time()
        time_records['server_aggregation']["overhead"].append(end - start)
        
        start = time.time()
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        end = time.time()
        time_records['server_aggregation']["aggregation"].append(end - start)
        
        return new_model

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.gradL = None
        self.alpha = option['alpha']
        time_records['local_training'][self.name] = []
        return
    
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        
        start = time.time()
        self.train(model)
        end = time.time()
        time_records['local_training'][self.name].append(end - start)
        
        cpkg = self.pack(model, loss)
        return cpkg

    def train(self, model):
        if self.gradL == None:
            self.gradL = model.zeros_like().to('cuda')
        # global parameters
        src_model = copy.deepcopy(model).to('cuda')
        src_model.freeze_grad()
        model = model.to('cuda')
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                l1 = self.calculator.get_loss(model, batch_data, device='cuda')
                l2 = 0
                l3 = 0
                for pgl, pm, ps in zip(self.gradL.parameters(), model.parameters(), src_model.parameters()):
                    l2 += torch.dot(pgl.view(-1), pm.view(-1))
                    l3 += torch.sum(torch.pow(pm-ps,2))
                loss = l1 - l2 + 0.5 * self.alpha * l3
                loss.backward()
                optimizer.step()
        # update grad_L
        self.gradL = self.gradL - self.alpha * (model-src_model)
        return

