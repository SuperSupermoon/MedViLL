import torch
import torchvision
import torch.nn as nn
from einops import rearrange

class ImageEncoder_pool(nn.Module):
    def __init__(self, args):
        super(ImageEncoder_pool, self).__init__()
        self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.pool_func = (
            nn.AdaptiveMaxPool2d
            if args.img_embed_pool_type == 'max'
            else nn.AdaptiveAvgPool2d)

    def forward(self, x):
        out = self.model(x)
        model_out = out.size()[-2:]
        W = int(model_out[0] / 2)
        H = int(model_out[1] / 2)
        pool = self.pool_func((W, H))
        out = torch.flatten(pool(out), start_dim=2).transpose(1, 2).contiguous() 
        # random pixel sampling
        random_sampling = torch.randperm(out.size(1))[:self.args.num_image_embeds]
        random_sampling, _ = torch.sort(random_sampling)
        random_sample = out[:, random_sampling]
        return random_sample


class ImageEncoder_cnn(nn.Module):
    def __init__(self, args):
        super(ImageEncoder_cnn, self).__init__()
        self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)  # 512x512: torch.Size([16, 2048, 16, 16])
        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()
        
        vis_pe = torch.arange(out.size(1), dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), out.size(1))

        random_sampling = torch.randperm(out.size(1))[:self.args.num_image_embeds]
        random_sampling, _ = torch.sort(random_sampling)

        random_sample = out[:, random_sampling]
        random_position = vis_pe[:, random_sampling]
        return random_sample, random_position


class fully_use_cnn(nn.Module):
    def __init__(self):
        super(fully_use_cnn, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        pool_func = (nn.AdaptiveAvgPool2d)
        self.pool = pool_func((3, 1))

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()
        vis_pe = torch.arange(out.size()[1], dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size()[0], out.size()[1])
        return out, vis_pe


class Img_patch_embedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

    def forward(self, img, mask=None):
        img_size = img.size()
        p = self.patch_size
        out = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        out = self.patch_to_embedding(out)
        return out
