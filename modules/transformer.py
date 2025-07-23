import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2).float() / d_model) * (-torch.log(torch.Tensor([10000.0]))))

        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # (1, max_len, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, value=-torch.inf)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads # 每个头处理的维度数
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()  # (B, L, d_model)
        q = self.w_q(q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)    # (B, num_heads, L, d_k)
        k = self.w_k(k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)    # (B, num_heads, L, d_k)
        v = self.w_v(v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)    # (B, num_heads, L, d_k)

        # B, L_q, _ = q.shape
        # B, L_k, _ = k.shape
        # B, L_v, _ = v.shape

        # q = self.w_q(q).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        # k = self.w_k(k).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        # v = self.w_v(v).view(B, L_v, self.num_heads, self.d_k).transpose(1, 2)


        scores, _ = scaled_dot_product_attention(q, k, v, mask)     # (B, num_heads, L, d_k)
        concat = scores.transpose(1, 2).contiguous().view(B, L, -1) # (B, L, d_model)

        return self.out(concat)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.linear2(x)
        return out

class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multiattn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.multiattn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x

    

class Encoder(nn.Module):
    def __init__(self, vocal_size, d_model, N, num_heads, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocal_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
    
    def forward(self, src, mask=None):
        x = self.embedding(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocal_size, d_model, N, num_heads, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocal_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.fc_out = nn.Linear(d_model, vocal_size)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        # out = F.softmax(x, dim=-1)
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, N, num_heads, d_ff, dropout)

    def make_pad_mask(self, seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones(size, size)).bool().to(next(self.parameters()).device)

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = self.make_pad_mask(src, src_pad_idx)
        tgt_mask = self.make_pad_mask(tgt, tgt_pad_idx) & self.make_subsequent_mask(tgt.size(1))
        enc_out = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return output


# 小词表（中英共用）
word2idx = {
    'PAD': 0, 'SOS': 1, 'EOS': 2,
    'I': 3, 'am': 4, 'a': 5, 'student': 6,
    'You':7, 'are':8, 'teacher':9,
    '我': 10, '是': 11, '学生': 12,
    '你': 13, '老师':14, '猪': 15,
    "爱": 16, "倪梦晴": 17, "love": 18,
    "nmq": 19, "you": 20
}

idx2word = {v:k for k,v in word2idx.items()}
vocab_size = len(word2idx)

# 训练数据：简单英译中对
pairs = [
    (["I", "am", "a", "student"], ["我", "是", "学生"]),
    (["You", "are", "a", "teacher"], ["你", "是", "老师"]),
    (["I", "love", "nmq"], ["我", "爱", "倪梦晴"])
]

# 超参数
max_len = 6  # max sequence length (include SOS, EOS)
batch_size = len(pairs)
d_model = 32
nhead = 4
num_layers = 2

# 编码函数，添加SOS, EOS，pad到max_len
def encode(tokens, vocab, max_len):
    seq = [vocab['SOS']] + [vocab[t] for t in tokens] + [vocab['EOS']]
    seq += [vocab['PAD']] * (max_len - len(seq))
    return seq[:max_len]

# 准备数据tensor
src_seqs = [encode(src, word2idx, max_len) for src, _ in pairs]
tgt_seqs = [encode(tgt, word2idx, max_len) for _, tgt in pairs]


src = torch.tensor(src_seqs)  # (B, T)
tgt = torch.tensor(tgt_seqs)  # (B, T)


# tgt输入是去掉最后一个token的序列，输出是去掉第一个token的序列
tgt_input = tgt[:, :-1]
tgt_output = tgt[:, 1:]

# 生成mask
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))

# 创建PyTorch官方Transformer模型
model = nn.Transformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    dim_feedforward=64,
    dropout=0.1,
    batch_first=True,
)

# 词嵌入层
src_embedding = nn.Embedding(vocab_size, d_model)
tgt_embedding = nn.Embedding(vocab_size, d_model)

# 输出线性层
output_linear = nn.Linear(d_model, vocab_size)

# 优化器和损失函数
optimizer = torch.optim.Adam(list(model.parameters()) + list(src_embedding.parameters()) + list(tgt_embedding.parameters()) + list(output_linear.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['PAD'])

# 训练循环
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    src_emb = src_embedding(src)
    tgt_emb = tgt_embedding(tgt_input)
    out = model(src_emb, tgt_emb, tgt_mask=tgt_mask)
    out = output_linear(out)  # (B, T, vocab_size)
    loss = criterion(out.view(-1, vocab_size), tgt_output.reshape(-1))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 推理：贪婪解码（简易版）
def greedy_decode(src_seq, max_len=6):
    model.eval()
    src_seq = torch.tensor([encode(src_seq, word2idx, max_len)])
    src_emb = src_embedding(src_seq)
    memory = model.encoder(src_emb)

    ys = torch.tensor([[word2idx['SOS']]])
    for i in range(max_len - 1):
        tgt_emb = tgt_embedding(ys)
        tgt_mask = generate_square_subsequent_mask(tgt_emb.size(1))
        out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out = output_linear(out)
        prob = out[:, -1, :]
        next_word = prob.argmax(dim=-1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        if next_word.item() == word2idx['EOS']:
            break
    return [idx2word[idx.item()] for idx in ys[0][1:-1]]  # skip SOS and EOS

# 测试
for src_sent, tgt_sent in pairs:
    pred = greedy_decode(src_sent)
    print(f"{src_sent} -> {''.join(pred)}")
