from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import os, json, numpy as np
from algorithm.cfmtx.cfmtx import cfmtx_test

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def run(self):
        super().run()
        acc, cfmtx = cfmtx_test(self.model, self.test_data, "cuda")
        np.savetxt(f"./measures/{self.option['algorithm']}_cfmtx.json", cfmtx, fmt="%.3f", delimiter=",")
        return
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


