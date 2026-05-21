"""
字符级语言模型训练脚本，使用 Transformer 模型，含 PPL 计算和文本生成功能。
用法:
    python language_model.py --epochs 20  # 训练模型
    python language_model.py --generate --load best_model.pt --prompt "你好"  # 生成文本
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据处理 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Transformer 模型 ───────────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                 hidden_dim, dropout, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 嵌入层（字符嵌入 + 位置编码）
        self.char_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过模型最大长度 {self.max_seq_len}")
        
        # 字符嵌入 + 位置编码
        char_emb = self.char_emb(x)  # (B, T, D)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.pos_emb(pos)  # (B, T, D)
        x_emb = self.drop(char_emb + pos_emb)

        # 生成自注意力掩码（防止看到未来字符）
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Transformer 前向传播
        out = self.transformer(x_emb, mask)  # (B, T, D)
        out = self.drop(out)
        logits = self.fc(out)  # (B, T, V)
        
        return logits

    @torch.no_grad()
    def generate(self, char2idx, idx2char, prompt, max_len=200, temperature=1.0, top_k=0):
        """
        文本生成函数
        :param char2idx: 字符到索引的映射
        :param idx2char: 索引到字符的映射
        :param prompt: 生成起始提示文本
        :param max_len: 生成文本最大长度
        :param temperature: 温度系数（越高越随机）
        :param top_k: Top-K 采样（0 表示不使用）
        :return: 生成的文本
        """
        self.eval()
        device = next(self.parameters()).device
        
        # 将提示文本转换为索引
        input_ids = [char2idx[c] for c in prompt if c in char2idx]
        if not input_ids:
            raise ValueError("提示文本中无有效字符")
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 生成文本
        for _ in range(max_len):
            # 截断过长序列（防止超出max_seq_len）
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # 前向传播获取预测
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature  # (1, V)
            
            # Top-K 采样
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # 计算概率并采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接结果
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 终止条件（可根据需要添加结束符）
            if next_token.item() == char2idx.get("\n", -1) and len(input_ids[0]) > len(prompt) + 10:
                break
        
        # 转换为文本
        generated_ids = input_ids[0].cpu().tolist()
        generated_text = "".join([idx2char[idx] for idx in generated_ids])
        
        return generated_text


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=256)  # Transformer 建议更大的嵌入维度
    parser.add_argument("--num_heads",  type=int,   default=8)    # Transformer 注意力头数
    parser.add_argument("--num_layers", type=int,   default=4)    # Transformer 层数
    parser.add_argument("--hidden_dim", type=int,   default=1024) # FFN 隐藏层维度
    parser.add_argument("--dropout",    type=float, default=0.1)  # Transformer 建议更小的dropout
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer.pt")
    
    # 生成参数
    parser.add_argument("--generate",   action="store_true", help="是否生成文本")
    parser.add_argument("--load",       default="best_transformer.pt", help="加载模型路径")
    parser.add_argument("--prompt",     default="", help="生成提示文本")
    parser.add_argument("--max_len",    type=int,   default=200, help="生成文本最大长度")
    parser.add_argument("--temperature",type=float, default=1.0, help="生成温度")
    parser.add_argument("--top_k",      type=int,   default=10, help="Top-K 采样数")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer")

    # 文本生成模式
    if args.generate:
        if not args.prompt:
            raise ValueError("生成模式必须指定 --prompt 提示文本")
        
        # 加载模型
        checkpoint = torch.load(args.load, map_location=device)
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        model_args = checkpoint["args"]
        
        # 构建模型
        model = TransformerLM(
            vocab_size=len(char2idx),
            embed_dim=model_args["embed_dim"],
            num_heads=model_args["num_heads"],
            num_layers=model_args["num_layers"],
            hidden_dim=model_args["hidden_dim"],
            dropout=model_args["dropout"],
            max_seq_len=model_args["seq_len"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        
        # 生成文本
        print(f"\n提示文本: {args.prompt}")
        print("-" * 50)
        generated_text = model.generate(
            char2idx=char2idx,
            idx2char=idx2char,
            prompt=args.prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"生成文本: {generated_text}")
        return

    # 训练模式
    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 构建Transformer模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_seq_len=args.seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 学习率调度器（Transformer常用）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        # 学习率调度
        scheduler.step(va_loss)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()