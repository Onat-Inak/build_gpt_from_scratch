
import torch 
import torch.nn as nn
from Head import Head as Head
from MultiHeadAttention import MultiHeadAttention as MultiHeadAttention
from FeedForward import FeedForward as FeedForward
from Block import Block as Block
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module): 
    def __init__(self, modelData):
        super().__init__()
        self.vocab_size = modelData.vocab_size
        self.n = modelData.n
        self.batch_size = modelData.batch_size
        self.device = modelData.device
        self.seq_len_train = modelData.n

        self.d_model = 512
        self.num_heads = 8
        self.d_k = self.d_model//self.num_heads
        self.d_v = self.d_model//self.num_heads
        self.mlp_dim = 4 * self.d_model
        self.n_blocks = 6
        self.dropout = 0.2

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding_table = nn.Embedding(self.n, self.d_model)

        self.blocks = nn.Sequential(*[Block(self.num_heads, self.d_model, self.d_k, self.d_v, self.n, self.mlp_dim, self.dropout) for _ in range(self.n_blocks)])
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)


    def forward(self, idx, targets=None):
        # idx: (batch_size,n)

        token_emb = self.token_embedding_table(idx) # (batch_size,n,d_model)
        
        pos_idx = torch.arange(self.n, device = self.device) # (n)
        pos_emb = self.pos_embedding_table(pos_idx) # (n,d_model) 

        # print("token_emb.shape: ", token_emb.shape, "   |   pos_emb.shape:", pos_emb.shape)

        emb = token_emb + pos_emb # (batch_size,n,d_model) --> pos_emb is broadcasted along batch_size
        # print("emb.shape :", emb.shape)

        x = self.blocks(emb)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (batch_size,n,vocab_size)

        if targets == None:
            loss = None
        else:
            [B,T,C] = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_forward = idx[:, -self.seq_len_train:]
            # [self.batch_size, self.n] = idx_forward.shape
            # for head in self.block.sa_head.heads:
            #     head.n = self.n
            # print("idx_forward.shape: ", idx_forward.shape)
            logits, loss = self.forward(idx_forward)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim =-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
