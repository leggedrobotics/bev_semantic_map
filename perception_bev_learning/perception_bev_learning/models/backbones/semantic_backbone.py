import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.models import BACKBONES

__all__ = ["SegmenterDino"]


@BACKBONES.register_module()
class SegmenterDino(nn.Module):
    def __init__(self, backbone, decoder, ckpt_path, *arg, **kwarg):
        super().__init__()
        self.semantic_net = DINOv2_Segmenter(backbone, decoder)
        self.semantic_net.load_state_dict(torch.load(ckpt_path, map_location="cuda"))

    def init_weights(self):
        for name, param in self.semantic_net.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.semantic_net(x)
        return x


class DINOv2_Segmenter(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.dinov2 = DinoV2Wrapper(backbone)

        self.decoder = SegmenterDecoderHead(decoder)

    def forward(self, x):
        out, og_shape = self.dinov2(x)
        og_shape = [og_shape[0] // 8, og_shape[1] // 8]
        out_dino = F.interpolate(
            out[0], size=og_shape, mode="bilinear", align_corners=False
        )
        semantic_features = self.decoder(out, og_shape)
        semantic_features = F.softmax(semantic_features)
        
        return [semantic_features]
        # return [torch.cat((out_dino, semantic_features), dim=1)]


class DinoV2Wrapper(nn.Module):
    def __init__(
        self,
        config={
            "unfreeze_layers": [],
            "layer_output": [12],
            "return_cls_token": False,
            "size": "small",
        },
    ):
        super().__init__()
        if config["size"] == "small":
            size = "dinov2_vits14"
        elif config["size"] == "base":
            size = "dinov2_vitb14"
        self.model = torch.hub.load("facebookresearch/dinov2", size)

        # which layers to get output from
        self.layer_output = [i - 1 for i in config["layer_output"]]
        self.unfreeze_layers = [i - 1 for i in config["unfreeze_layers"]]

        for name, param in self.model.named_parameters():
            # print(name)
            layer_vals = [
                name.startswith("blocks.{}".format(i - 1)) for i in self.unfreeze_layers
            ]

            if any(layer_vals):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.patch_size = 14
        self.return_cls_token = config["return_cls_token"]

    def forward(self, x):
        # convert the input to be divisible by patch size
        b, c, h, w = x.shape

        x = F.interpolate(
            x,
            size=(
                h // self.patch_size * self.patch_size,
                w // self.patch_size * self.patch_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        return (
            self.model.get_intermediate_layers(
                x,
                self.layer_output,
                reshape=True,
                return_class_token=self.return_cls_token,
            ),
            [h, w],
        )


class SegmenterDecoderHead(nn.Module):
    def __init__(
        self,
        config={
            "num_output_channels": 16,
            "d_encoder": 384,
            "n_layers": 2,
            "n_heads": 6,
            "d_model": 384,
            "d_ff": 4 * 384,
            "dropout": 0.0,
        },
    ):
        super().__init__()
        self.d_encoder = config["d_encoder"]
        self.n_layers = config["n_layers"]
        self.n_cls = config["num_output_channels"]
        self.d_model = config["d_model"]
        self.d_ff = config["d_ff"]
        self.scale = self.d_model**-0.5

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model, config["n_heads"], self.d_ff, config["dropout"]
                )
                for i in range(self.n_layers)
            ]
        )

        self.cls_emb = nn.Parameter(
            torch.randn(1, self.n_cls, self.d_model)
        )  # These are the tokens that are added to the encoder output,
        self.proj_dec = nn.Linear(self.d_encoder, self.d_model)

        self.patch_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.classes_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.mask_norm = nn.LayerNorm(self.n_cls)

        self.num_classes = self.n_cls

        # self.init_weights()

        # print out number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of parameters in SegmenterDecoderHead: %d" % num_params)

    # def init_weights(self):
    #     # trunc_normal_(self.pos_embed, std=0.02)
    #     # nn.init.normal_(self.cls_token, std=1e-6)
    #     # if self.register_tokens is not None:
    #     #     nn.init.normal_(self.register_tokens, std=1e-6)
    #     named_apply(init_weights_vit_timm, self)

    def forward(self, x, og_shape):
        # if coming from output of dinov2, x is a list of tensors
        if len(x) == 1:
            x = x[-1]
            # x is of shape (b, inner_dim, n_patches_h, n_patches_w)
            b, inner_dim, num_patches_h, num_patches_w = x.shape
            x = x.reshape(b, inner_dim, -1).transpose(1, 2)  # (b, n, d)
        else:
            # x is a list of tensors from the different layers of dinov2
            # concatenate them along the dimension of the number of patches
            x = torch.cat(x, dim=1)
            # print(x.shape)
            b, _, num_patches_h, num_patches_w = x.shape
            x = x.reshape(b, self.d_encoder, -1).transpose(1, 2)  # (b, n, d++)

        x = self.proj_dec(x)  # project the encoder output to the model dimension
        cls_emb = self.cls_emb.expand(
            b, -1, -1
        )  # expand the cls_emb to the same size as x, along batch dimension
        x = torch.cat(
            (x, cls_emb), 1
        )  # concatenate x and cls_emb along the second dimension
        for blk in self.blocks:
            x = blk(x)
            # self.intermediate_layer_outputs.append(x)
        x = self.decoder_norm(x)

        # split x into patch embeddings and class segmentation features embeddings
        patches = self.patch_proj(
            x[:, : -self.num_classes]
        )  # (b, n, d) @ (d, d) -> (b, n, d)
        cls_seg_feat = self.classes_proj(
            x[:, -self.num_classes :]
        )  # (b, n_cls, d) @ (d, d) -> (b, n_cls, d)

        # normalize the patch embeddings
        patches = F.normalize(patches, dim=2, p=2)
        # normalize the class segmentation features embeddings
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        # we are doing some funky stuff here

        masks = patches @ cls_seg_feat.transpose(
            1, 2
        )  # (b, n, d) @ (b, d, n_cls) -> (b, n, n_cls)
        masks = self.mask_norm(masks)

        masks = (
            masks.permute(0, 2, 1)
            .contiguous()
            .view(b, -1, num_patches_h, num_patches_w)
        )

        # ####### added afterwards
        # # print(og_shape[2:])
        # # w, h
        masks = torch.nn.functional.interpolate(
            masks, size=og_shape, mode="bilinear"
        )  # , align_corners=False)

        return masks


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        print("Using Regular Attention")
        self.attn = Attention(dim, heads, dropout)

        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None, return_attention=False):
        y = self.attn(self.norm1(x), mask)
        x = x + y  # residual connection, bypassing attention
        x = x + self.mlp(self.norm2(x))  # residual connection, bypassing MLP
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


if __name__ == "__main__":
    ckpt_path = "/home/jonfrey/Downloads/model.pt"

    backbone_dict = {
        "unfreeze_layers": [],
        "layer_output": [12],
        "return_cls_token": False,
        "size": "small",
    }
    decoder_dict = {
        "num_output_channels": 10,
        "d_encoder": 384,
        "n_layers": 1,
        "n_heads": 6,
        "d_model": 192,
        "d_ff": 768,
        "dropout": 0.0,
    }
    model = DINOv2_Segmenter(backbone_dict, decoder_dict)
    model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
