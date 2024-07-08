import torch
import pickle

class ModelData():
    def __init__(self, data_path, batch_size, n, train_test_ratio, device):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.text_len = len(self.text)
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {}
        self.itos = {}
        self.create_mapping()
        # self.save_map_dicts()
        self.data = torch.tensor(self.encode(self.text), dtype = torch.long)
        self.data_len = len(self.data)
        self.train_data, self.val_data = None, None
        self.batch_size = batch_size
        self.n = n
        self.ttr = train_test_ratio
        self.split_data()
        self.device = device

    def encode(self, str):
        return [self.stoi[c] for c in str]

    def decode(self, lst):
        return ''.join([self.itos[i] for i in lst])
    
    def create_mapping(self):
        for [i, ch] in enumerate(self.chars):
            self.stoi.update({ch:i})
            self.itos.update({i:ch})

    def split_data(self):
        k = int(self.ttr * len(self.data))
        self.train_data = self.data[:k]
        self.val_data = self.data[k:]

    def get_batch(self, type, log=False):
        xb,yb = None, None
        if type == "train":   
            ix = torch.randint((len(self.train_data) - self.n), (self.batch_size,))
            xb = torch.stack([self.train_data[i : i+self.n] for i in ix])
            yb = torch.stack([self.train_data[i+1 : i+self.n+1] for i in ix])
        
        else:
            ix = torch.randint((len(self.val_data) - self.n), (self.batch_size,))
            xb = torch.stack([self.val_data[i : i+self.n] for i in ix])
            yb = torch.stack([self.val_data[i+1 : i+self.n+1] for i in ix])
        
        if log == True:
            self.log_io_relation(xb, yb)
        # return xb.to("cuda"), yb.to("cuda")
        return xb.to(self.device), yb.to(self.device)
    
    def log_io_relation(self, xb, yb):
        print("xb.shape: ", xb.shape)
        print("yb.shape: ", yb.shape)
        print("xb: ", xb)
        print("yb: ", yb)
        for ba in range(self.batch_size):
            for t in range(self.n):
                context = xb[ba, :t+1]
                target = yb[ba, t]
                print(f"when input is {context.tolist()} the target: {target}")
    
    def save_map_dicts(self):
        with open('/workspace/work_dir/dicts/itos.pkl', 'wb') as f:
            pickle.dump(self.itos, f)
        with open('/workspace/work_dir/dicts/stoi.pkl', 'wb') as f:
            pickle.dump(self.stoi, f)
