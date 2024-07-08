import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, d_model, d_k, d_v, n):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n = n
        
        self.query = nn.Linear(self.d_model, self.d_k, bias = False)
        self.key = nn.Linear(self.d_model, self.d_k, bias = False)
        self.value = nn.Linear(self.d_model, self.d_v, bias = False)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer('tril', torch.tril(torch.ones(self.n, self.n), diagonal = 0))

    def forward(self, emb, bool_mask=True):
        # emb : (batch_size,n,d_model)
        q = self.query(emb) # (batch_size,n,d_k)
        # print("q.shape: ", q.shape)
        k = self.key(emb) # (batch_size,n,d_k)
        # print("k.shape: ", k.shape)
        v = self.value(emb) # (batch_size,n,d_v)
        # print("v.shape: ", v.shape)

        similarity_scores = q @ torch.transpose(k, -2, -1) * (self.d_k ** -0.5) # (batch_size,n,d_k) @ (batch_size,d_k,n) --> (batch_size,n,n)
        # print("similarity_scores.shape: ", similarity_scores.shape)
        if (bool_mask == True):
            masked_similarity_scores = similarity_scores.masked_fill(self.tril[:self.n, :self.n] == 0, float("-inf")) # (batch_size,n,n) 
        attention_matrix = self.softmax(masked_similarity_scores) # (batch_size,n,n)
        # print("attention_matrix.shape: ", attention_matrix.shape)
        # print("attention_matrix: ", attention_matrix)
        # print("----------------------------")
        # print("v.shape: ", v.shape)
        # print("v: ", v)
        out = attention_matrix @ v # (batch_size,n,n) @ (batch_size,n,d_v) --> (batch_size,n,d_v)
        return out

