import torch
import torch.nn as nn
import math

# pos_encoding and _pos_encoding
def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)
    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))
    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

# UNet
class UNet(nn.Module):
    def __init__(self, in_ch, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.down1 = ConvBlock(in_ch, 256, time_embed_dim)  #in_ch=256(たぶん)
        self.down2 = ConvBlock(256, 512, time_embed_dim)
        self.bot1 = ConvBlock(512, 1024, time_embed_dim)
        self.up2 = ConvBlock(512 + 1024, 512, time_embed_dim)
        self.up1 = ConvBlock(512 + 256, 256, time_embed_dim)
        self.out = nn.Conv2d(256, in_ch, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        x = self.bot1(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x

# Diffuser
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
    
        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std


#↑ここまで拡散モデルのサンプルそのまま
# Autoencoder with Diffusion
class AutoencoderWithDiffusion(nn.Module):
    def __init__(self, time_embed_dim=100, num_timesteps=1000, device='cpu'):
        super(AutoencoderWithDiffusion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),  # 1->(1.0)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0))    
        )

        self.unet = UNet(in_ch=256, time_embed_dim=time_embed_dim)
        self.diffuser = Diffuser(num_timesteps=num_timesteps, device=device)
        self.num_timesteps = num_timesteps
        self.device = device

    def forward(self, x, fixed_timestep=None):
        latent = self.encoder(x)
        
        # ランダムなステップ数（学習時）か,固定されたステップ数（テスト時）か
        if fixed_timestep is not None: #テスト時
            timesteps = torch.full((latent.shape[0],), fixed_timestep, device=self.device, dtype=torch.long)
            noisy_latent, noise = self.diffuser.add_noise(latent, timesteps)

            self.unet.eval()
            noise_pred = self.unet(noisy_latent, timesteps)
            self.unet.train()
            
            denoised_latent = self.diffuser.denoise(self.unet, noisy_latent, timesteps)
            reconstructed = self.decoder(denoised_latent)
        else: #学習時
            timesteps = torch.randint(1, self.num_timesteps + 1, (latent.shape[0],), device=self.device)
            noisy_latent, noise = self.diffuser.add_noise(latent, timesteps)
            noise_pred = self.unet(noisy_latent, timesteps)
            # denoised_latent = None
            # reconstructed = None
            denoised_latent = self.diffuser.denoise(self.unet, noisy_latent, timesteps)
            reconstructed = self.decoder(denoised_latent)

        return reconstructed, latent, noisy_latent, denoised_latent, noise, noise_pred

