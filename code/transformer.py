import torch
import torch.nn as nn
        

class TfmEncoderDecoder(nn.Module):
    def __init__(self, xdim, seq_len, d_model, nhead, dim_feedforward, dropout, activation, bias, num_layers):
        super().__init__()
        # modules
        self.linear = nn.Linear(xdim, d_model, bias=bias)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation),
            num_layers=num_layers,
            norm=None,
        )
        self.tgt = nn.Parameter(torch.randn(size=(seq_len, d_model)))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation),
            num_layers=num_layers,
            norm=None,
        )
        self.fc_out = nn.Linear(d_model, 1, bias=bias)

    def forward(self, x):
        x = self.linear(x)          # (bz, seq_len, X_DIM -> d_model)
        src = x.permute(1, 0, 2)    # (seq_len, bz, d_model)
        mem = self.encoder(src)     # (seq_len, bz, d_model)
        seq_len, bz, d = mem.shape
        tgt = self.tgt[:, None, :].expand(seq_len, bz, d)
        out = self.decoder(tgt, mem)  # (seq_len, bz, d_model)
        yhat = self.fc_out(out)       # (seq_len, bz, 1)
        return yhat.squeeze(-1).T     # (bz, seq_len)

x = torch.randn(100, 10, 128)
model = TfmEncoderDecoder(
    xdim=128,
    seq_len=10,
    d_model=16,
    nhead=4,
    dim_feedforward=16,
    dropout=0.1,
    activation="gelu",
    bias=False,
    num_layers=3,
)
y = model(x)
print(y.shape)