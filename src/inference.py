import torch
import pickle
from ModelData import ModelData as MD 
from BigramLanguageModel import BigramLanguageModel as BLM
from torch.nn import functional as F

def decode(itos, lst):
    return ''.join([itos[i] for i in lst])

data_path = 'data/dataset/tiny_shakespeare/input.txt'
batch_size = 64
n = 256
train_test_ratio = 0.9
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_new_tokens = 800

modelData = MD(data_path, batch_size, n, train_test_ratio, device)

model = BLM(modelData).to(device)
model.load_state_dict(torch.load('inference_models/model_weights.pth'))

###### MODEL SUMMARY ######
print("Pretrained model weights are loading on GPU...")
print("\nModel Summary: \n")
print(model)
print("\nNumber of total model parameters: ", sum(p.numel() for p in model.parameters()))

idx = torch.zeros((1,256), dtype=torch.long).to(device)

print("\nModel is generating an output of size", max_new_tokens,"-->\n\n")

for _ in range(max_new_tokens):
    idx_forward = idx[:, -n:]
    logits, loss = model(idx_forward)
    logits = logits[:,-1,:]
    probs = F.softmax(logits, dim =-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    print(decode(modelData.itos, idx_next[0].tolist()), end='')
    idx = torch.cat((idx, idx_next), dim = 1)
