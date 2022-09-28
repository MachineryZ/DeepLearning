# WAE

Wasserstein Auto-Encoder
https://arxiv.org/pdf/1711.01558.pdf

MMD-based $D_Z$ 是指一个替代 kl divergence 的，用 maximum mean discrepancy MMd 来当做loss
$$
MMD_k(P_Z, Q_Z) = ||\int_Z k(z,\cdot) dP_Z(z) - \int_Z k(z,\cdot )dQ_Z(z) ||_{H_k}
$$

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
class WAE_MMD(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dims: int,
        hidden_dims: List = None,
        reg_weight: int = 100,
        kernel_type: str = "imq",
        latent_var: float = 2.,
        **kwargs,
    ) -> None:
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
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
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, latent_dim)

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
    
    def encode(self, input: Tensor) -> Tensor:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z
    
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr
        recons_loss = F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z, reg_weight)
        loss = recons_loss + mmd_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "MMD": mmd_loss}

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)

        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type = "rbf":
            result = self.compute_rbf(x1, x2)
        elif self.kernal_type = "imq":
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError("Undefined kernel type.")

        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var
        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        

~~~
