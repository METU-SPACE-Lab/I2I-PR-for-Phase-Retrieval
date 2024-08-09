from models.denoisers import register_denoiser

import torch.nn as nn
import torch
from diffusers import UNet2DModel

hdd_mnt_pth = "mnt_hdd"


@register_denoiser("UNet2D")
class UNet2DModel256(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            layers_per_block=3,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0]

@register_denoiser("UNet2DMulti")
class UNet2DModel256Multi(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=10,
            out_channels=5,
            layers_per_block=3,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        state_dict = torch.load(f"/{hdd_mnt_pth}/onurcan/onurk/save_mololoed_betterscheduler_clamp_morenoise_moreiterations_notzx_new_yesyesyesyesyesyes_allimages_best_amax_newwithcorrectsnr.pth")
        del state_dict["lam"]
        state_dict = {k.replace("denoiser.", "").replace("model.", ""): v for k, v in state_dict.items()}
        state_dict['conv_in.weight'] = state_dict['conv_in.weight'].repeat(1, 10, 1, 1)
        state_dict['conv_out.weight'] = state_dict['conv_out.weight'].repeat(5, 1, 1, 1)
        state_dict['conv_out.bias'] = state_dict['conv_out.bias'].repeat(5)
        self.model.load_state_dict(state_dict, strict=False)
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0] + x[:, 0:5, :, :] 
    
    
from utils.utils import fft2d, ifft2d, zero_padding_twice, crop_center_half
@register_denoiser("UNet2DMultiLargeAdversarial")
class UNet2DMultiLargeAdversarial(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.freq_model = UNet2DModel(
            sample_size=512,
            in_channels=10,
            out_channels=2,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                # "DownBlock2D",
                # "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                # "UpBlock2D",
                # "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=6,
            out_channels=1,
            layers_per_block=4,#3,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        # state_dict = torch.load(f"/{hdd_mnt_pth}/onurcan/onurk/save_mololoed_betterscheduler_clamp_morenoise_moreiterations_notzx_new_yesyesyesyesyesyes_allimages_last_amax_newwithcorrectsnr_2__largenew_betterdc.pth")
        # del state_dict["lam"]
        # state_dict = {k.replace("denoiser.", "").replace("model.", ""): v for k, v in state_dict.items()}
        # # state_dict['conv_in.weight'] = state_dict['conv_in.weight'].repeat(1, 10, 1, 1)
        # # state_dict['conv_out.weight'] = state_dict['conv_out.weight'].repeat(5, 1, 1, 1)
        # # state_dict['conv_out.bias'] = state_dict['conv_out.bias'].repeat(5)
        # del state_dict['conv_out.weight']
        # del state_dict['conv_out.bias']
        # del state_dict['conv_in.weight']
        # self.model.load_state_dict(state_dict, strict=False)
    
    def forward(self, x):
        Fx = fft2d(zero_padding_twice(x))
        Fx_abs = Fx.abs()
        Fx_angle = Fx.angle()
        Fx_concat = torch.cat([Fx_abs, Fx_angle], dim=1)
        denoised_Fx = self.freq_model(Fx_concat, 0, return_dict=False)[0] + torch.cat([Fx_abs.mean(dim=1, keepdim=True), Fx_angle.mean(dim=1, keepdim=True)], dim=1)
        denoised_Fx = denoised_Fx[:, 0:1, :, :] * torch.exp(1j * denoised_Fx[:, 1:2, :, :])
        freq_denoised = crop_center_half(ifft2d(denoised_Fx).real)
        
        freq_denoised_x_concat = torch.cat([x, freq_denoised], dim=1)
        return self.model(freq_denoised_x_concat, 17, return_dict=False)[0] + x.mean(dim=1, keepdim=True), freq_denoised


@register_denoiser("UNet2DMultiLargeAdversarialNoFreq")
class UNet2DMultiLargeAdversarialNoFreq(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=5,
            out_channels=1,
            layers_per_block=4,#3,
            block_out_channels=(128, 128, 256, 512, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    
    def forward(self, x):
        return self.model(x, 0, return_dict=False)[0] + x.mean(dim=1, keepdim=True), 0




@register_denoiser("UNet2DDefault")
class UNet2DModel256Default(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            # block_out_channels=(256, 384, 512, 640)
        )
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0]
    
    
@register_denoiser("UNet2DAdv")
class UNet2DModel256Default(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0]

    
@register_denoiser("UNet2DCustom")
class UNet2DModel256Custom(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(224, 224, 448, 448, 896),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0]


@register_denoiser("UNet2DPretrained1")
class UNet2DModelPretrained1(nn.Module):
    def __init__(self):
        super().__init__()
        # "thanhtuit96/ddpm_flickr1024_256", subfolder="unet",
        # "xutongda/adm_imagenet_256x256_unconditional",
        # alkiskoudounas/sd-universe-256px
        # self.model = UNet2DModel().from_pretrained("google/ddpm-church-256", sample_size=256, in_channels=1, out_channels=1, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            layers_per_block=3,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        state_dict = torch.load(f"/{hdd_mnt_pth}/onurcan/onurk/indi_pretrained_new_best_noiseconst.pth")
        # state_dict = torch.load(f"/{hdd_mnt_pth}/onurcan/onurk/indi_pretrained_best.pth")
        # state_dict = torch.load(f"/{hdd_mnt_pth}/onurcan/onurk/save_mololoed_betterscheduler_clamp_morenoise_moreiterations_notzx_new_yesyesyesyesyesyes_allimages_best_amax_newwithcorrectsnr_fastertrain.pth")
        # state_dict = torch.load(f"/{hdd_mnt_pth}/onurcan/onurk/save_mololoed_betterscheduler_clamp_morenoise_moreiterations_notzx_new_yesyesyesyesyesyes_allimages_best_amax_newwithcorrectsnr.pth")
        del state_dict["lam"]
        state_dict = {k.replace("denoiser.", "").replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
    
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0] + x # bu mevcutta yoktu
    
    
    
@register_denoiser("UNet2DTwoChannels")
class UNet2DModelTwoChannels(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=2,
            out_channels=1,
            layers_per_block=4,
            block_out_channels=(128, 256, 256, 512, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0] + x.mean(dim=1, keepdim=True)
    
    



    
@register_denoiser("UNet2D11Channels")
class UNet2DModel11Channels(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=256,
            in_channels=11,
            out_channels=1,
            layers_per_block=4,
            block_out_channels=(128, 256, 256, 512, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
    def forward(self, x, timesteps):
        return self.model(x, timesteps, return_dict=False)[0] + x.mean(dim=1, keepdim=True)
    
    
    