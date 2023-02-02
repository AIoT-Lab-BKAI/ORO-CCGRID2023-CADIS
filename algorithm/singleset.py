from pathlib import Path
from .mp_fedbase import BasicServer, BasicClient
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch.optim as optim
import json


class PillDataset(Dataset):
    # def __init__(self, user_idx, img_folder_path="", idx_dict=None, label_dict=None, map_label_dict=None):
    #     super().__init__()
    #     self.user_idx = user_idx
    #     self.idx = idx_dict[str(user_idx)]
    #     self.img_folder_path = img_folder_path
    #     self.label_dict = label_dict
    #     self.map_label_dict = map_label_dict
    #     self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        
    def __init__(self, user_idxes=[], img_folder_path="", idx_dict=None, label_dict=None, map_label_dict=None):
        super().__init__()
        
        self.idx = []
        for user_idx in user_idxes:
            self.idx += idx_dict[str(user_idx)]
            
        self.img_folder_path = img_folder_path
        self.label_dict = label_dict
        self.map_label_dict = map_label_dict
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        img_name = self.idx[item]
        img_path = os.path.join(self.img_folder_path,img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        pill_name = self.label_dict[img_name]
        label = self.map_label_dict[pill_name]
        return img,label
    
    @classmethod
    def combine(datasets=[]):
        pass
    

def read_json_idx(filename):
    idxs = []
    file_idxes = json.load(open(filename, "r"))
    for client_id in file_idxes:
        idxs += file_idxes[client_id]
    return idxs


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        single_set_idx = 'dataset_idx/' + option['dataidx_filename']
        data_folder = option['data_folder']
        
        dataset_name = option['task'].split('_')[0]
        
        if dataset_name == 'mnist': 
            train_dataset = datasets.MNIST(data_folder, 
                                           train=True, 
                                           download=True, 
                                           transform=transforms.Compose([
                                               transforms.ToTensor(), 
                                               transforms.Normalize((0.1307,), (0.3081,))
                                               ]))
            self.train_dataset = CustomDataset(train_dataset, read_json_idx(single_set_idx))
            
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(data_folder, 
                                            train=True, 
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                    (0.2023, 0.1994, 0.2010))
                                                ]))
            self.train_dataset = CustomDataset(train_dataset, read_json_idx(single_set_idx))
            
        elif dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(data_folder, 
                                            train=True, 
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                    (0.2023, 0.1994, 0.2010))
                                                ]))
            self.train_dataset = CustomDataset(train_dataset, read_json_idx(single_set_idx))
        
        elif dataset_name == 'pilldataset':
            with open(option['dataidx_path'] +"/client_dataset/user_group_img.json",'r') as f:
                user_group_img = json.load(f)
            with open(option['dataidx_path'] +"/client_dataset/img_label_dict.json",'r') as f:
                img_label_dict = json.load(f)
            with open(option['dataidx_path'] +"/client_dataset/label_hash.json",'r') as f:
                label_hash = json.load(f)
            with open(option['dataidx_path'] +"/server_dataset/user_group_img.json",'r') as f:
                server_user_group_img = json.load(f)
            with open(option['dataidx_path'] +"/server_dataset/img_label_dict.json",'r') as f:
                server_img_label_dict = json.load(f)  
            
            self.test_data = PillDataset([0], f"{option['data_folder']}/server_dataset/pill_cropped", server_user_group_img, server_img_label_dict, label_hash)
            n_clients = len(user_group_img)
            self.train_dataset = PillDataset([idx for idx in range(n_clients)], f"{option['data_folder']}/client_dataset/pill_cropped", user_group_img, img_label_dict, label_hash)
        
        self.optimizer = optim.Adam(self.model.parameters())
        
    def iterate(self, t):
        self.train()
        return
    
    def train(self, model=None):
        if model == None:
            model = self.model
            
        model.train()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        data_loader = DataLoader(self.train_dataset, batch_size=self.clients_per_round*self.option['batch_size'], shuffle=True)
        
        for iter in range(self.option['num_epochs']):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                self.optimizer.step()


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


