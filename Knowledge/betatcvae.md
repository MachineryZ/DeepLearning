# Beta TC VAE

Isolating Sources of Disentanglement in VAEs
https://arxiv.org/pdf/1802.04942.pdf

参考复现代码如下：
https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py

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
class BetaTCVAE(BaseVAE):
    num_iter = 0
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        aneal_steps: int = 200,
        alpha: float = 1.,
        beta: float = 6.,
        gamma: float = 1.,
        **kwargs,
    ) -> None:
        self.latent_dim = latent_dim
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32]
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1] * 16, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, 256 * 2)
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
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    
    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.
~~~

