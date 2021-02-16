import torch
from torch.distributions.normal import Normal
from torch.nn import Linear
from torch.nn import ReLU


def european_option_delta(log_moneyness, time_expiry, volatility) -> torch.Tensor:
    """
    Return Black-Scholes delta of European option.

    Parameters
    ----------
    - s : torch.Tensor
        log moneyness.
    - t : torch.Tensor
        Time to expiry.
    - v : torch.Tensor
        Volatility.

    Returns
    -------
    delta : torch.Tensor

    Examples
    --------
    >>> bs_delta([-0.01, 0.00, 0.01], 0.1, 0.2)
    tensor([0.4497, 0.5126, 0.5752])
    """
    s, t, v = map(torch.as_tensor, (log_moneyness, time_expiry, volatility))
    normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
    return normal.cdf((s + (v ** 2 / 2) * t) / (v * torch.sqrt(t)))


def clamp(x, min_value, max_value) -> torch.Tensor:
    """
    Clamp all elements in the input tensor into the range [`min_value`, `max_value`]
    and return a resulting tensor.
    The bounds `min_value` and `max_value` can be tensors.

    If min_value <= max_value:

        out = min_value if x < min_value
              x if min_value <= x <= max_value
              max_value if x > max_value

    If min_value > max_value:

        out = (min_value + max_value) / 2

    Parameters
    ----------
    - x : torch.Tensor, shape (*)
        The input tensor.
    - min_value : float or torch.Tensor, default None
        Lower-bound of the range to be clamped to.
    - max_value : float or torch.Tensor, default None
        Upper-bound of the range to be clamped to.

    Returns
    -------
    out : torch.tensor, shape (*)
        The output tensor.

    Examples
    --------
    >>> x = torch.linspace(-2, 12, 15) * 0.1
    >>> x
    tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
             0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
    >>> clamp(x, 0.0, 1.0)
    tensor([0.0000, 0.0000, 0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000,
            0.7000, 0.8000, 0.9000, 1.0000, 1.0000, 1.0000])

    >>> x = torch.tensor([1.0, 0.0])
    >>> clamp(x, [0.0, 1.0], 0.0)
    tensor([0.0000, 0.5000])
    """
    if min_value is not None:
        min_value = torch.as_tensor(min_value)
        x = torch.max(x, min_value)
    if max_value is not None:
        max_value = torch.as_tensor(max_value)
        x = torch.min(x, max_value)
    if min_value is not None and max_value is not None:
        x = torch.where(min_value < max_value, x, (min_value + max_value) / 2)
    return x


class MultiLayerPerceptron(torch.nn.ModuleList):
    """
    Feed-forward neural network.

    Parameters
    ----------
    - in_features : int
        Number of input features.
    - out_features : int
        Number of output features.
    - n_layers : int
        Number of hidden layers.
    - n_units : int, default
        Number of units in each hidden layer.

    Examples
    --------
    >>> net = Net(3, 1)
    >>> net
    Net(
      (0): Linear(in_features=3, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ReLU()
      (4): Linear(in_features=32, out_features=32, bias=True)
      (5): ReLU()
      (6): Linear(in_features=32, out_features=32, bias=True)
      (7): ReLU()
      (8): Linear(in_features=32, out_features=1, bias=True)
    )
    >>> net(torch.empty(2, 3))
    tensor([[...],
            [...]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, in_features, out_features, n_layers=4, n_units=32):
        super().__init__()
        for n in range(n_layers):
            i = in_features if n == 0 else n_units
            self.append(Linear(i, n_units))
            self.append(ReLU())
        self.append(Linear(n_units, out_features))

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


def generate_geometric_brownian_motion(
    n_paths, maturity=30 / 365, dt=1 / 365, volatility=0.2, device=None
) -> torch.Tensor:
    """
    Return geometric Brownian motion of shape (n_steps, n_paths).

    Parameters
    ----------
    - n_paths : int
        Number of price paths.
    - maturity : float, default 30 / 365
        Length of simulated period.
    - dt : float, default 1 / 365
        Interval of time steps.
    - volatility : float, default 0.2
        Volatility of asset price.

    Returns
    -------
    geometric_brownian_motion : torch.Tensor, shape (N_STEPS, N_PATHS)
        Here `N_PATH = int(maturity / dt)`.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> generate_geometric_brownian_motion(3, maturity=5 / 365)
    tensor([[1.0000, 1.0000, 1.0000],
            [1.0023, 0.9882, 0.9980],
            [1.0257, 0.9816, 1.0027],
            [1.0285, 0.9870, 1.0112],
            [1.0405, 0.9697, 1.0007]])
    """
    randn = torch.randn((int(maturity / dt), n_paths))
    randn[0, :] = 0.0
    bm = volatility * torch.sqrt(torch.tensor(dt)) * randn.cumsum(0)
    t = torch.linspace(0, maturity, int(maturity / dt)).reshape(-1, 1)
    geometric_brownian_motion = torch.exp(bm - (volatility ** 2) * t / 2)
    geometric_brownian_motion = geometric_brownian_motion.to(device=device)
    return geometric_brownian_motion


def entropic_loss(pnl) -> torch.Tensor:
    """
    Return entropic loss function, which is a negative of
    the expected entropic utility:

        loss(pnl) = -E[u(pnl)], u(x) = -exp(-x)

    Parameters
    ----------
    pnl : torch.Tensor, shape (*)
        Profit-loss distribution.

    Returns
    -------
    loss : torch.Tensor, shape (,)

    Examples
    --------
    >>> pnl = -torch.arange(4.0)
    >>> entropic_loss(pnl)
    tensor(7.7982)
    """
    return -torch.mean(-torch.exp(-pnl))


def cash_equivalent(pnl) -> torch.Tensor:
    """
    Return cash equivalent of profit-loss distribution
    with respect to entropic exponential utility.

    Parameters
    ----------
    pnl : torch.Tensor, shape (*)
        Profit-loss distribution.

    Returns
    -------
    cash_equivaluent : torch.Tensor, shape (,)

    Examples
    --------
    >>> pnl = -torch.arange(4.0)
    >>> cash_equivalent(pnl)
    tensor(-2.0539)
    """
    return -torch.log(entropic_loss(pnl))
