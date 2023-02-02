from .fedbase import BasicServer, BasicClient
import time, wandb, json
from algorithm.cfmtx.cfmtx import cfmtx_test

time_records = {"server_aggregation": {"overhead": [], "aggregation": []}, "local_training": {}}

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

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
        
        acc, cfmtx = cfmtx_test(self.model, self.test_data, "cuda")
        json.dump(cfmtx, open(f"./measures/{self.option['algorithm']}_cfmtx.json", "w"))
        return
    
class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
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
