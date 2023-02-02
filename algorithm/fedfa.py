from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np
import copy
import time, wandb, json

time_records = {"server_aggregation": {"overhead": [], "aggregation": []}, "local_training": {}}

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # self.m = fmodule._modeldict_zeroslike(self.model.state_dict())
        self.m = copy.deepcopy(self.model) * 0.0
        self.beta = option['beta']
        self.alpha = 1.0 - self.beta
        self.gamma = option['gamma']
        self.eta = option['learning_rate']
        self.paras_name=['beta','gamma']
        
    def run(self):
        super().run()
        json.dump(time_records, open(f"./measures/{self.option['algorithm']}.json", "w"))
        return

    def unpack(self, pkgs):
        ws = [p["model"] for p in pkgs]
        losses = [p["train_loss"] for p in pkgs]
        ACC = [p["acc"] for p in pkgs]
        freq = [p["freq"] for p in pkgs]
        return ws, losses, ACC, freq

    def iterate(self, t):
        # sample clients
        self.selected_clients = self.sample()
        # training
        ws, losses, ACC, F = self.communicate(self.selected_clients)
        if self.selected_clients == []: return
        
        start = time.time()
        # aggregate
        # calculate ACCi_inf, fi_inf
        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha*accinf+self.beta*finf for accinf,finf in zip(ACCinf,Finf)]
        wnew = self.aggregate(ws, p)
        dw = wnew - self.model
        # calculate m = γm+(1-γ)dw
        self.m = self.gamma * self.m + (1 - self.gamma) * dw
        end = time.time()
        time_records['server_aggregation']["overhead"].append(end - start)
        
        start = time.time()
        self.model = wnew - self.m * self.eta
        end = time.time()
        time_records['server_aggregation']["aggregation"].append(end - start)
        
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.frequency = 0
        self.momentum = option['gamma']
        time_records['local_training'][self.name] = []

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        acc, loss = self.test(model,'train')
        
        start = time.time()
        self.train(model)
        end = time.time()
        time_records['local_training'][self.name].append(end - start)
        
        cpkg = self.pack(model, loss, acc)
        return cpkg

    def pack(self, model, loss, acc):
        self.frequency += 1

        return {
            "model":model,
            "train_loss":loss,
            "acc":acc,
            "freq":self.frequency,
        }
        