# Category VAE

Categorical Reparameterization with gumbel-softmax

https://arxiv.org/pdf/1611.01144.pdf

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
class CategoricalVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        categorical_dim: int = 40, # Num classes
        hidden_dims: List = None,
        temperature: float = 0.5,
        anneal_rate: float = 3e-5,
        anneal_interval: int = 100,
        alpha: float = 30.,
        **kwargs,
    ) -> None:
        super(CategoricalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
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
                    nn.BatchNorm(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, self.latent_dim * self.categorical_dim)

        modules = []
        self.decoder_input = nn.Linear(self.latent_dim * self.categorical_dim, hidden_dims[-1] * 4)
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
                
            )
        )


~~~


