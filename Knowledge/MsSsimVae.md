# MSSSIM VAE

Learning to Generate Images With Pereptual Similarity Metrics
https://arxiv.org/pdf/1511.06409.pdf
MS-SSIM VAE 也被称为 EL VAE


代码复现参考：
https://github.com/AntixK/PyTorch-VAE/blob/master/models/mssim_vae.py

~~~python
class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NoteImplementedError
    
    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError
    
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NoteImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        rase NoteImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *input: Any, **kwargs) -> Tensor:
        pass
~~~

~~~python
class MSSSIMVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        window_size: int = 11,
        size_average: bool = True,
        **kwargs,
    ) -> None:
        super(MSSSIMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dimes = [32, 64, 128, 256, 512]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*moodules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.mssim_loss = MSSSIM(self.in_channels, window_size, size_average)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args: Any, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]
        recons_loss = self.mssim_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    

class MSSIM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        window_size: int = 11,
        size_average: bool = True,
    ) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)
        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size: int, sigma: float) -> Tensor:
        kernel = torch.tensor([exp((x - window_size // 2) ** 2/(2 * simga **2)) for x in range(window_size)])

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1: Tensor, img2: Tensor, window_size: int, in_channels: int, size_averrage: bool) -> Tensor:
        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=in_channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = muu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=in_channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=in_channels) - mu1_mu2

        img_range = 1.0
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma_sq + sigma_sq + C2
        cs = torch.mean(v1 / v2)
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs
    
    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        # if normalize:
        #     mssim = (mssim + 1) / 2
        #     mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output


~~~

