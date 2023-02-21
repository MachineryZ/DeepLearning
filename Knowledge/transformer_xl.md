# Transformer XL

transformer xl 的代码， https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py:

~~~python
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class MemTransformerLM(nn.Module):
    def __init__(self, n_tocken, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, tie_weight=True, d_embed=None, div_val=1, tie_projs=[False], pre_norm=False, tgt_len=)
~~~