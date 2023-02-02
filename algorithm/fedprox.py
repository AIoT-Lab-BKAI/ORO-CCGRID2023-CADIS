from .fedbase import BasicServer, BasicClient
import copy
import torch
import time, wandb, json

time_records = {"server_aggregation": {"overhead": [], "aggregation": []}, "local_training": {}}

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['mu']
    
    def iterate(self, t):
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: return
        
        time_records['server_aggregation']["overhead"].append(0)
        
        start = time.time()
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        end = time.time()
        time_records['server_aggregation']["aggregation"].append(end - start)
        return
    
    def run(self):
        super().run()
        json.dump(time_records, open(f"./measures/{self.option['algorithm']}.json", "w"))
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.mu = option['mu']
        time_records['local_training'][self.name] = []

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        
        start = time.time()
        self.train(model)
        end = time.time()
        time_records['local_training'][self.name].append(end - start)
        
        cpkg = self.pack(model, loss)
        return cpkg
    
    def train(self, model, device='cuda'):
        # global parameters
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                original_loss = self.calculator.get_loss(model, batch_data, device='cuda')
                # proximal term
                loss_proximal = 0
                for pm, ps in zip(model.parameters(), src_model.parameters()):
                    loss_proximal += torch.sum(torch.pow(pm-ps,2))
                loss = original_loss + 0.5 * self.mu * loss_proximal                #
                loss.backward()
                optimizer.step()
        return

