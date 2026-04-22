# -*- coding: utf-8 -*-
# Source adapted from official AST repository:
# https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py

import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
import wget
from timm.models.layers import to_2tuple, trunc_normal_
from torch.cuda.amp import autocast

_PRETRAIN_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "pretrained_ast"
_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(_PRETRAIN_DIR))


class PatchEmbed(nn.Module):
    """Override timm PatchEmbed to relax input shape constraints."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """Official AST model from YuanGongND/ast with small path adjustments."""

    def __init__(
        self,
        label_dim=527,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        audioset_pretrain=False,
        model_size="base384",
        verbose=True,
    ):
        super().__init__()
        assert timm.__version__ == "0.4.5", "Please use timm==0.4.5 for official AST compatibility."

        if verbose:
            print("---------------AST Model Summary---------------")
            print(
                "ImageNet pretraining: {:s}, AudioSet pretraining: {:s}".format(
                    str(imagenet_pretrain), str(audioset_pretrain)
                )
            )

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if not audioset_pretrain:
            if model_size == "tiny224":
                self.v = timm.create_model("vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "small224":
                self.v = timm.create_model("vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "base224":
                self.v = timm.create_model("vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "base384":
                self.v = timm.create_model("vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain)
            else:
                raise ValueError("model_size must be one of tiny224, small224, base224, base384")

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print("frequncey stride={:d}, time stride={:d}".format(fstride, tstride))
                print("number of patches={:d}".format(num_patches))

            new_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(16, 16),
                stride=(fstride, tstride),
            )
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                )
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(t_dim / 2) : int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed,
                        size=(self.oringal_hw, t_dim),
                        mode="bilinear",
                    )

                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(f_dim / 2) : int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")

                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

        else:
            if not imagenet_pretrain:
                raise ValueError(
                    "audioset_pretrain=True requires imagenet_pretrain=True in official AST implementation."
                )
            if model_size != "base384":
                raise ValueError("audioset_pretrain=True only supports model_size='base384'.")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_path = _PRETRAIN_DIR / "audioset_10_10_0.4593.pth"
            if not ckpt_path.exists():
                audioset_mdl_url = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
                wget.download(audioset_mdl_url, out=str(ckpt_path))

            sd = torch.load(str(ckpt_path), map_location=device)
            audio_model = ASTModel(
                label_dim=527,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=1024,
                imagenet_pretrain=False,
                audioset_pretrain=False,
                model_size="base384",
                verbose=False,
            )
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print("frequncey stride={:d}, time stride={:d}".format(fstride, tstride))
                print("number of patches={:d}".format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2) : 6 - int(f_dim / 2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        # Input x shape: (batch_size, time_frames, frequency_bins)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        bsz = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(bsz, -1, -1)
        dist_token = self.v.dist_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        return x
