import torch
import torch.nn as nn
import torchvision



import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
from glob import glob

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048

        # out = self.pool(self.model(x))
        # out = torch.flatten(out, start_dim=2)
        # out = out.transpose(1, 2).contiguous()
        
        out = self.model(x)
        out = torch.flatten(out, start_dim=2) #out torch.Size([100, 2048, 3])
        out = out.transpose(1, 2).contiguous() #out torch.Size([100, 3, 2048])

        # print("out.size()",out.size())
        # input("STOP!!!")
        

        return out  # BxNx2048


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(args.img_hidden_sz * args.num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.clf(x)
        return out



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
            else nn.AdaptiveAvgPool2d
        )

    def forward(self, x):
        # B x 3 x W x H -> B x 2048 x M x M -> B x 2048 x N -> B x N x 2048
        out = self.model(x)
        model_out = out.size()[-2:]

        W = int(model_out[0] / 2)
        H = int(model_out[1] / 2)
        pool = self.pool_func((W, H))
        out = pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # B x N x 2048
        #print('out_size:', out.size())  # torch.Size([32, 9, 2048])

        # random pixel sampling
        # TODO: At each iteration, randomly sample pixels
        num_range = out.size()[1]
        random_sampling = torch.randperm(num_range)[:self.args.num_image_embeds]
        random_sampling, _ = torch.sort(random_sampling)
        #print('random_sampling:', random_sampling)
        random_sample = out[:, random_sampling]
        print('random_sample_size:', random_sample.size())

        return random_sample

class random_sample(nn.Module):
    def __init__(self, args):
        super(random_sample, self).__init__()
        self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # B x 3 x W x H -> B x 2048 x M x M -> B x 2048 x N -> B x N x 2048
        out = self.model(x)  # 512x512: torch.Size([16, 2048, 16, 16])

        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # B x N x 2048
        # print('out_size:', out.size())  # torch.Size([32, 256, 2048])
        # print('random_sample_size out:', out.size())

        

        # position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        # position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)

        vis_pe = torch.arange(out.size()[1], dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size()[0], out.size()[1])
            
        # random pixel sampling
        # TODO: At each iteration, randomly sample pixels
        num_range = out.size()[1]
        random_sampling = torch.randperm(num_range)[:self.args.num_image_embeds]
        random_sampling, _ = torch.sort(random_sampling)

        #print('random_sampling:', random_sampling)
        random_sample = out[:, random_sampling] #16, 0, 2048???
        random_position = vis_pe[:, random_sampling]
        
        # print('random_sample_size out:', random_sample.size())

        # input("img size check !   STOP!!!")
        return random_sample, random_position

class fully_use_cnn(nn.Module):
    def __init__(self):
        super(fully_use_cnn, self).__init__()
        # self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
        )
        self.pool = pool_func((3, 1))

    def forward(self, x):
        # print("Size of x: ", x.size())
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        out = torch.flatten(out, start_dim=2) #out torch.Size([100, 2048, 3])
        out = out.transpose(1, 2).contiguous() #out torch.Size([100, 3, 2048])
        
        # print("Size of x after passed to Model: ", out.size())
        # out = self.pool(self.model(x)) #out torch.Size([100, 2048, 3, 1])
        out = torch.flatten(out, start_dim=2) #out torch.Size([100, 2048, 3])
        out = out.transpose(1, 2).contiguous() #out torch.Size([100, 3, 2048])
        # print("fully_use_cnn",out.size())

        vis_pe = torch.arange(out.size()[1], dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size()[0], out.size()[1])
        
        return out, vis_pe  # BxNx2048  # torch.Size([1, 2048, 7, 7])

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
        # print("\n")
        # print(f'img_size :{img_size}')
        p = self.patch_size
        # print(f'patch size :{p}')
        out = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # print(f'Image patch token is (batch, hxw , patch x patch x channel) : {x.size()}')
        out = self.patch_to_embedding(out)
        # print(f'patch_to_embedding to each token : {out.size()}')
        return out

