import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r 

        # self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # self.weight.requires_grad = False
        self.weight = torch.randn(out_features, in_features)
        # self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        orig_out = torch.matmul(x, self.weight.T)

        lora_out = self.dropout(torch.matmul(x, self.lora_A.T))  # A*x
        lora_out = torch.matmul(lora_out, self.lora_B.T) * self.scaling  # B*(A*x)

        return orig_out + lora_out
    
    
def apply_lora(model, LoRALayer, r=8, alpha=32):

    # for name, module in model.named_modules():
    #     # 查找每个 transformer 层的 QKV 权重
    #     if isinstance(module, nn.Linear) and 'c_attn' in name:
    #         # 替换原始权重矩阵为 LoRA 适配层
    #         lora_layer = LoRALayer(module.weight, r, alpha)
    #         module.weight = lora_layer.forward()
    #         print(f"LoRA applied to {name}")

    for name, module in model.named_modules():
        if "q_proj" in name or "v_proj" in name:  # 仅作用于 Query 和 Value 层
            setattr(model, name, LoRALinear(module.in_features, module.out_features))
