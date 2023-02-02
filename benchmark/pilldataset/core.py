
from benchmark.toolkits import BasicTaskReader,PillDataset
from benchmark.toolkits import ClassifyCalculator

import json


class TaskReader(BasicTaskReader):
    def __init__(self,taskpath="", data_folder="./dataset_idx/pill_dataset", dataidx_path=""):
        super(TaskReader,self).__init__(taskpath)
        self.dataidx_path = dataidx_path
        self.data_folder = data_folder
        
    def read_data(self):
        with open(self.dataidx_path +"/client_dataset/user_group_img.json",'r') as f:
            user_group_img = json.load(f)
        with open(self.dataidx_path +"/client_dataset/img_label_dict.json",'r') as f:
            img_label_dict = json.load(f)
        with open(self.dataidx_path +"/client_dataset/label_hash.json",'r') as f:
            label_hash = json.load(f)
        with open(self.dataidx_path +"/server_dataset/user_group_img.json",'r') as f:
            server_user_group_img = json.load(f)
        with open(self.dataidx_path +"/server_dataset/img_label_dict.json",'r') as f:
            server_img_label_dict = json.load(f)  
        
        test_dataset = PillDataset(0, f"{self.data_folder}/server_dataset/pill_cropped",server_user_group_img,server_img_label_dict,label_hash)
        n_clients = len(user_group_img)
        train_data = [PillDataset(idx, f"{self.data_folder}/client_dataset/pill_cropped", user_group_img, img_label_dict, label_hash) for idx in range(n_clients)]
        return train_data,test_dataset,n_clients

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
