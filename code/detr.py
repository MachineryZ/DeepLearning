import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads,
    num_encoder_layers, num_decoder_layers):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
        self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        print(f"h.shape = {h.shape}")
        print(f"decoder input.shape = {(pos + h.flatten(2).permute(2,0,1)).shape}")
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
        self.query_pos.unsqueeze(1))
        print(f"decoder output.shape = {h.shape}")
        print(f"query_pos.shape = {self.query_pos.shape}")
        print(f"query_pos.unsqueeze(1).shape = {self.query_pos.unsqueeze(1).shape}")
        return self.linear_class(h), self.linear_bbox(h).sigmoid()

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(10, 3, 800, 1200)
logits, bboxes = detr(inputs)

print(f"logits.shape = {logits.shape}")
print(f"bboxes.shape = {bboxes.shape}")

"""
h.shape = torch.Size([1, 256, 25, 38])
decoder input.shape = torch.Size([950, 1, 256]) [seq_len, bs, feature]
decoder output.shape = torch.Size([100, 1, 256]) [seq_len, bs, feature]
query_pos.shape = torch.Size([100, 256]) 
query_pos.unsqueeze(1).shape = torch.Size([100, 1, 256]) [seq_len, bs, feature]
logits.shape = torch.Size([100, 1, 92])
bboxes.shape = torch.Size([100, 1, 4])
"""


