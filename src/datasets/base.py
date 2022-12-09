import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, total_data):
        self.total_data = total_data
        
    def __len__(self):
        return len(self.total_data)
    
    def __getitem__(self, idx):
        return self.total_data[idx]