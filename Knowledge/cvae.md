# CVAE

Learniing Structured Output Representation using Deep Conditional Generative Models
https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf

VAE 的变分下界为：
$$
L(\phi, \theta; x) = -KL(q_\phi(z|x)||p_\theta(z)) + E_{q_\phi(z|x)}[logp_\theta(z|x)] \leq log p_\theta(x)
$$

CVAE 的变分下界为：
$$
L(\phi, \theta; x, y) = -KL(q_\phi(z|x, y)||p_\theta(z|x)) + E_{q_\phi(z|x,y)}[logp_\theta(y|x, z)]\leq p_\theta(y|x)
$$

其中 (x, y) 是一般有监督学习中的数据对，可以看出 cvae 相当于一个有监督版本的 vae，他重构生成的是 y|x (vae 重构生成的是 x)。举个例子的话，x表示手写数字的类别标签，y表示手写数字图像，就可以通过采样z生成制定数字x对应的图像y。值得一提的是，vae中的关于z的先验项是 $p_\theta(z)$ 而 cvae 中的先验项 $p_\theta(z|x)$ 与 x 有关系

代码实现部分：
https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
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
class ConditionalVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_dim: int,
        hidden_dims: List = None,
        img_size: int = 64,
    ) -> None:
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        moduls = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        in_channels += 1 # for extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
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
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)
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
        self.decoder = nn.Sequential(*modules)
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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensr:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs["labels"].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, y], dim=1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]
        reoncs_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * toorch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}

    def sample(self, num_samples: int, current_device, **kwargs) -> Tensor:
        y = kwargs["labels"].float()
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]
        
~~~