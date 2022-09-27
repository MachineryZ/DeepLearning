# VAE

Auto-Encoding Variantional Bayes

直觉：对于一个 high dimensional 的 random variable $x$，我们想要从某一个条件概率分布来生成 $p_\theta(x|z)$。通常来说 z 的维度会比 x 的维度少很多。那么，得到 z 也需要一个分布 $p_\theta(z)$。所以完整的生成模型表达式应该是 $p_\theta(z) p_\theta(x|z)$

1. 想根据 $x$ 得到 $z$
    1. $p_\theta(x) = p_\theta(x|z)p_\theta(z)/p_\theta(x)$
2. 想得到 $x$ 的分布估计
    1. $p_\theta(x) = \int p_\theta(z) p_\theta(x|z) dz$
3. 使用一个神经网络 $q_\phi$ 来拟合
    1. $p_\theta(z|x) = q_\phi(z|x)$

那么我们这里就设置好了网络的输出，loss function 就用 kl divergence 来设计：
1. $D_{KL}(q_\phi(z|x)||p_\theta(z|x))=-\sum_z q_\phi (z|x)[\log(\frac{p_\theta(x,z)}{q_\phi(z|x)}-\log(p_\theta(x)))]$
2. $\log(p_\theta(x))=D_{KL}(q_\phi(z|x)||p_\theta(z|x))+L(\theta,\phi; x)$（variation lower bound）


通俗的解释就是：像 autoencoder 类的模型，在 hidden space 空间里，都是离散的点，来进行映射（因为数据有限，所以只能覆盖空间中有限点）那么，为了覆盖整个空间，我们会加上噪声，但是噪声范围又有限，所以加上了无限范围的高斯噪声。

接下来的代码是从这个 github repo 里 cp 下来的：
https://github.com/AntixK/PyTorch-VAE/blob/master/models/base.py
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
class VanillaVAE(Base):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        **kwargs,
    ) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        models = []
        if hidden_dime is None:
            hidden_dims = [32, 64, 128, 256, 512]
        # Build Encoder
        for h_dim in hidden_dims:
            models.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernels_size=3, stride=2, padding=1),
                    nn.BatchNoorm2d(h_dim),
                    nn.LeakyReLU()
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
                        output_paddings=1,
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
            nn.Conv2d(
                hidden_dims[-1],
                out_channels=3,
                kernel_size=3,
                padding=1,
            )
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        input: (Tensor) in shape of [N, C, H, W]
        return: List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        z in shape of [B, D]
        return in shape of [B, C, H, W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        mu: mean of latent Gaussian [B, D]
        logvar: standard deviation of the latent Gaussian [B, D]
        return: [B, D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0,
        )
        loss = recns_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction Loss": recons_loss.detach(), "kld": -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
~~~