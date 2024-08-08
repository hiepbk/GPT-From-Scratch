import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Union, Tuple


class TransformerBlock(nn.Module):
    def __init__(
            self, 
            num_heads: int, # số head có trong multihead attention
            n_embed: int, # kích thước của không gian vector nhúng 
            block_size: int # số lượng token đầu vào 
        ):
        super(TransformerBlock, self).__init__()
        hidden_dim = n_embed // num_heads # tính số chiều không gian embed của các self-attention block 
        self.mhsa = MultiHeadSelfAttention(num_heads, hidden_dim, n_embed, block_size) 
        self.feed_forward = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # kích thước đầu vào [batch, block_size, n_embed]
        x = x + self.mhsa(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class FeedForward(nn.Module):
    def __init__(
            self, 
            n_embed: int, # kích thước không gian nhúng embed
            extend_width: int=4, # hệ số mở rộng
            dropout: float=0.2
        ):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, extend_width*n_embed), # khởi tạo lớp linear projection kích thước [n_embed, extend_width*n_embed]
            nn.ReLU(),
            nn.Linear(extend_width*n_embed, n_embed), # khởi tạo lớp linear projection kích thước [extend_width*n_embed, n_embed]
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # đầu vào kích thước []
        return self.layer(x) # -> đầu ra kích thước []


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            num_heads: int, # số lượng Self-attention block 
            hidden_dim: int, # số chiều không gian ẩn 
            n_embed: int, # số chiều không gian nhúng vector đầu vào 
            block_size: int, # số lượng token đầu vào 
            dropout: float=0.2
        ):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads # số lượng Self-attention block 
        self.heads = nn.ModuleList([SingleHead(hidden_dim, n_embed, block_size) for _ in range(self.num_heads)])
        self.project = nn.Linear(n_embed, n_embed) # [n_embed, n_embed]
        self.drop = nn.Dropout(dropout) # [n_embed, n_embed]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = torch.cat([sh(x) for sh in self.heads], dim=-1)
        out = self.project(out)
        out = self.drop(out)
        return out # [batch, block_size, n_embed]


class SingleHead(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, # kích thước không gian ẩn 
            n_embed: int, # kích thước không gian vector nhúng đầu vào 
            block_size: int, # số lượng token đầu vào 
            dropout: float=0.2
        ):
        super(SingleHead, self).__init__()
        self.key = nn.Linear(n_embed, hidden_dim, bias=False) # khởi tạo lớp linear projection để ánh xạ đầu vào thành ma trận key
        self.query = nn.Linear(n_embed, hidden_dim, bias=False) # khởi tạo lớp linear projection để ánh xạ đầu vào thành ma trận query
        self.value = nn.Linear(n_embed, hidden_dim, bias=False) # khởi tạo lớp linear projection để ánh xạ đầu vào thành ma trận value
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # tạo và lưu trữ một ma trận tam giác dưới kích thước [block_size, block_size]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x [batch, block_size, n_embed]
        B, T, C = x.shape
        k = self.key(x) # đầu ra kích thước [batch, block_size, hidden_dim]
        q = self.query(x) # đầu ra kích thước [batch, block_size, hidden_dim]
        weights = q @ k.transpose(-2, -1) * C**(-0.5) # đầu ra kích thước [batch, block_size, block_size]
        masked_weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        masked_probs = F.softmax(masked_weights, dim=-1)
        masked_probs = self.drop(masked_probs)
        v = self.value(x)
        out = masked_probs @ v
        return out # [batch, block_size, hidden_dim]


class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size: int, # kích thước từ điển 
            block_size: int, # xác định số lượng từ tối đa mà mô hình có thể xử lý (max_len)
            n_embed: int, # kích thước không gian nhúng từ 
            num_heads: int, # số lượng head trong multihead attention
            n_layers: int # số lượng TransformerEncode block 
        ):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size # kích thước từ điển 
        self.block_size = block_size # khởi tạo độ dài tối đa cho chuỗi đầu vào được xử lý ~ tương đương với kích thước chuỗi đầu vào (num_example)
        self.embedding = nn.Embedding(vocab_size, n_embed) # khởi tạo đối tượng mã hóa input đầu vào
        self.positional_embedding_table = nn.Embedding(block_size, n_embed) # khởi tạo đối tượng nhúng vị trí 
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_heads, n_embed, block_size) for _ in range(n_layers)],
        )
        self.norm = nn.LayerNorm(n_embed)        
        self.fc = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        token_embeddings = self.embedding(x) # B, T -> B, T, N_EMB
        positional_embedding = self.positional_embedding_table(torch.arange(T, device=x.device)) # T -> T, C
        token_embeddings = token_embeddings + positional_embedding # B, T, C + T, C -> B, T, C
        blocks_out = self.blocks(token_embeddings)
        blocks_out = self.norm(blocks_out)
        logits = self.fc(blocks_out) # B, T, N_EMB -> B, T, C
        logits = logits.reshape(B*T, self.vocab_size)
        return logits

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        t = idx.shape[1]
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits.reshape(1, t, -1)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if t < self.block_size:
                t += 1
        return idx


if __name__ == "__main__":
    vocab_size = 65
    block_size = 256
    n_embed = 384
    num_heads = 6
    n_layers = 6

    model = GPT(vocab_size, block_size, n_embed, num_heads, n_layers)
    inp = torch.ones((1,256), dtype=torch.long)
    out = model(inp)
    print(out.shape)