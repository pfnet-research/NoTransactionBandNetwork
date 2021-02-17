import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn.functional as fn
from torch.optim import Adam
from tqdm import tqdm

from utils import MultiLayerPerceptron
from utils import clamp
from utils import entropic_loss
from utils import european_option_delta
from utils import generate_geometric_brownian_motion
from utils import to_premium

seaborn.set_style("whitegrid")

FONTSIZE = 18
matplotlib.rcParams["figure.figsize"] = (10, 5)
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["figure.titlesize"] = FONTSIZE
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["legend.fontsize"] = FONTSIZE
matplotlib.rcParams["xtick.labelsize"] = FONTSIZE
matplotlib.rcParams["ytick.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.titlesize"] = FONTSIZE
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.1
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["axes.linewidth"] = 1.6


class NoTransactionBandNet(torch.nn.Module):
    """
    No-transaction band network.

    Parameters
    ----------
    - in_features : int, default 3
        Number of input features.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> m = NoTransactionBandNet()
    >>> x = torch.tensor([
    ...     [-0.01, 0.1, 0.2],
    ...     [ 0.00, 0.1, 0.2],
    ...     [ 0.01, 0.1, 0.2]])
    >>> prev = torch.full_like(x[:, 0], 0.5)
    >>> m(x, prev)
    tensor([..., ..., ...], grad_fn=<SWhereBackward>)
    """

    def __init__(self, in_features=3):
        super().__init__()
        self.net = MultiLayerPerceptron(in_features, 2)

    def forward(self, x, prev):
        no_cost_delta = european_option_delta(x[:, 0], x[:, 1], x[:, 2])
        band_width = self.net(x)

        lower = no_cost_delta - fn.leaky_relu(band_width[:, 0])
        upper = no_cost_delta + fn.leaky_relu(band_width[:, 1])

        return clamp(prev, lower, upper)


class FeedForwardNet(torch.nn.Module):
    """
    Feed-forward network with Black-Scholes delta.

    Parameters
    ----------
    - in_features : int, default 3
        Number of input features.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> m = FeedForwardNet(3)
    >>> x = torch.tensor([
    ...     [-0.01, 0.1, 0.2],
    ...     [ 0.00, 0.1, 0.2],
    ...     [ 0.01, 0.1, 0.2]])
    >>> prev = torch.full_like(x[:, 0], 0.5)
    >>> m(x, prev)
    tensor([..., ..., ...], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_features=3):
        super().__init__()
        self.net = MultiLayerPerceptron(in_features + 1, 1)

    def forward(self, x, prev):
        no_cost_delta = european_option_delta(x[:, 0], x[:, 1], x[:, 2])

        x = torch.cat((x, prev.reshape(-1, 1)), 1)
        x = self.net(x).reshape(-1)
        x = torch.tanh(x)

        return no_cost_delta + x


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---

    DEVICE

    # ---

    # In each epoch, N_PATHS brownian motion time-series are generated.
    N_PATHS = 50000
    # How many times a model is updated in the experiment.
    N_EPOCHS = 200

    # ---

    def to_numpy(tensor: torch.Tensor) -> np.array:
        return tensor.cpu().detach().numpy()

    # ---

    def european_option_payoff(prices: torch.Tensor, strike=1.0) -> torch.Tensor:
        """
        Return the payoff of a European option.

        Parameters
        ----------
        prices : torch.Tensor, shape (n_steps, n_paths)
            Prices of the underlying asset.

        Returns
        -------
        payoff : torch.Tensor, shape (n_paths, )
        """
        return fn.relu(prices[-1, :] - strike)

    def lookback_option_payoff(prices: torch.Tensor, strike=1.03) -> torch.Tensor:
        """
        Return the payoff of a lookback option.

        Parameters
        ----------
        prices : torch.Tensor, shape (n_steps, n_paths)
            Prices of the underlying asset.

        Returns
        -------
        payoff : torch.Tensor, shape (n_paths, )
        """
        return fn.relu(torch.max(prices, dim=0).values - strike)

    # ---

    def compute_profit_and_loss(
        hedging_model: torch.nn.Module,
        payoff: typing.Callable[[torch.Tensor], torch.Tensor],
        cost: float,
        n_paths=N_PATHS,
        maturity=30 / 365,
        dt=1 / 365,
        volatility=0.2,
    ) -> torch.Tensor:
        """
        Return profit-loss distribution after hedging.

        Parameters
        ----------
        - hedging_model : torch.nn.Module
            Hedging model to fit.
        - payoff : callable[[torch.Tensor], torch.Tensor]
            Payoff function of the derivative to hedege.
        - cost : float
            Transaction cost of underlying asset.

        Returns
        -------
        profit_and_loss : torch.Tensor, shape (n_paths,)
        """
        # prices: (time, batch)
        prices = generate_geometric_brownian_motion(
            n_paths, maturity=maturity, dt=dt, volatility=volatility, device=DEVICE
        )
        prev = torch.zeros_like(prices[0])

        pnl = -payoff(prices)

        for n in range(prices.shape[0] - 1):
            x_log_moneyness = prices[n, :, None].log()
            x_time_expiry = torch.full_like(x_log_moneyness, maturity - n * dt)
            x_volatility = torch.full_like(x_log_moneyness, volatility)
            x = torch.cat([x_log_moneyness, x_time_expiry, x_volatility], 1)

            hedge = hedging_model(x, prev)

            pnl += hedge * (prices[n + 1] - prices[n])
            pnl -= cost * torch.abs(hedge - prev) * prices[n]

            prev = hedge

        return pnl

    # ---

    # Definition of NoTransactionBandNet comes here

    # class NoTransactionBandNet(torch.nn.Module):
    #     ...

    # ---

    # Definition of FeedForwardNet comes here

    # class FeedForwardNet(torch.nn.Module):
    #     ...

    # ---

    torch.manual_seed(42)
    model_ntb = NoTransactionBandNet().to(DEVICE)
    torch.manual_seed(42)
    model_ffn = FeedForwardNet().to(DEVICE)

    # ---

    torch.manual_seed(42)
    pnl_ntb = compute_profit_and_loss(model_ntb, european_option_payoff, cost=1e-3)
    torch.manual_seed(42)
    pnl_ffn = compute_profit_and_loss(model_ffn, european_option_payoff, cost=1e-3)

    # ---

    plt.hist(
        to_numpy(pnl_ntb),
        bins=100,
        range=(-0.04, -0.01),
        alpha=0.6,
        label="No-transaction band Network",
    )
    plt.hist(
        to_numpy(pnl_ffn),
        bins=100,
        range=(-0.04, -0.01),
        alpha=0.6,
        label="Feed-forward Network",
    )
    plt.title(
        "Profit-loss histogram of 50000 price paths "
        "for a European option (before fit)"
    )
    plt.xlabel("Profit-loss")
    plt.ylabel("Number of events")
    plt.legend()
    plt.show()

    # ---

    def fit(
        hedging_model: torch.nn.Module,
        payoff: typing.Callable[[torch.Tensor], torch.Tensor],
        cost: float,
        n_epochs=N_EPOCHS,
    ) -> list:
        """
        Fit a hedging model to hedge the given derivative.

        Parameters
        ----------
        - hedging_model : torch.nn.Module
            Hedging model to fit.
        - payoff : callable[[torch.Tensor], torch.Tensor]
            Payoff function of the derivative to hedege.
        - cost : float, default 0.0
            Transaction cost of the underlying asset.
        - n_epochs : int, default N_EPOCHS
            How many times a hedging model is updated in the experiment.

        Returns
        -------
        history : list[float]
            Training history.
        """
        optim = Adam(hedging_model.parameters())

        history = []
        iterations = tqdm(range(n_epochs))

        for i in iterations:
            optim.zero_grad()
            pnl = compute_profit_and_loss(hedging_model, payoff, cost=cost)
            loss = entropic_loss(pnl)
            loss.backward()
            optim.step()

            iterations.desc = f"loss={loss:.5f}"
            history.append(loss.item())

        return history

    # ---

    torch.manual_seed(42)
    history_ntb = fit(model_ntb, european_option_payoff, cost=1e-3)
    torch.manual_seed(42)
    history_ffn = fit(model_ffn, european_option_payoff, cost=1e-3)

    # ---

    plt.figure()
    plt.plot(history_ntb, label="No-transaction band Network")
    plt.plot(history_ffn, label="Feed-forward Network")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss (Negative of expected utility)")
    plt.title("Learning history for a European option")
    plt.legend()
    plt.show()

    # ---

    torch.manual_seed(42)
    pnl_ntb = compute_profit_and_loss(model_ntb, european_option_payoff, cost=1e-3)
    torch.manual_seed(42)
    pnl_ffn = compute_profit_and_loss(model_ffn, european_option_payoff, cost=1e-3)

    # ---

    plt.figure()
    plt.hist(
        to_numpy(pnl_ntb),
        bins=100,
        range=(-0.12, 0.01),
        alpha=0.6,
        label="No-transaction band Network",
    )
    plt.hist(
        to_numpy(pnl_ffn),
        bins=100,
        range=(-0.12, 0.01),
        alpha=0.6,
        label="Feed-forward Network",
    )
    plt.title(
        "Profit-loss histogram of 50000 price paths "
        "for a European option (after fit)"
    )
    plt.xlabel("Profit-loss")
    plt.ylabel("Number of events")
    plt.legend()
    plt.show()

    # ---

    def evaluate_premium(
        hedging_model: torch.nn.Module,
        payoff: typing.Callable[[torch.Tensor], torch.Tensor],
        cost: float,
        n_times=20,
    ) -> float:
        """
        Evaluate the premium of the given derivative.

        Parameters
        ----------
        - hedging_model : torch.nn.Module
            Hedging model to fit.
        - payoff : callable[[torch.Tensor], torch.Tensor]
            Payoff function of the derivative to hedege.
        - cost : float, default 0.0
            Transaction cost of underlying asset.
        - n_times : int, default 20
            If `n_times > 1`, return ensemble mean of the results
            from multiple simulations.

        Returns
        -------
        premium : float
        """
        with torch.no_grad():
            p = lambda: -to_premium(
                compute_profit_and_loss(hedging_model, payoff, cost=cost)
            ).item()
            return float(np.mean([p() for _ in range(n_times)]))

    # ---

    torch.manual_seed(42)
    premium_ntb = evaluate_premium(model_ntb, european_option_payoff, cost=1e-3)
    torch.manual_seed(42)
    premium_ffn = evaluate_premium(model_ffn, european_option_payoff, cost=1e-3)

    # ---

    print("Premium evaluated by no-transaction band network :\t", premium_ntb)
    print("Premium evaluated by feed-forward band network   :\t", premium_ffn)

    # ---

    premium_reduced = (premium_ffn - premium_ntb) / premium_ffn * 100
    print("Reduced premium :\t", f"{premium_reduced:.4f} %")

    # ---

    torch.manual_seed(42)
    model_ntb = NoTransactionBandNet().to(DEVICE)
    torch.manual_seed(42)
    model_ffn = FeedForwardNet().to(DEVICE)

    # ---

    torch.manual_seed(42)
    history_ntb = fit(model_ntb, lookback_option_payoff, cost=1e-3)
    torch.manual_seed(42)
    history_ffn = fit(model_ffn, lookback_option_payoff, cost=1e-3)

    # ---

    plt.figure()
    plt.plot(history_ntb, label="No-transaction band Network")
    plt.plot(history_ffn, label="Feed-forward Network")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss (Negative of expected utility)")
    plt.title("Learning history for a lookback option")
    plt.legend()
    plt.show()

    # ---

    torch.manual_seed(42)
    premium_ntb = evaluate_premium(model_ntb, lookback_option_payoff, cost=1e-3)
    torch.manual_seed(42)
    premium_ffn = evaluate_premium(model_ffn, lookback_option_payoff, cost=1e-3)

    # ---

    print("Premium evaluated by no-transaction band network :\t", premium_ntb)
    print("Premium evaluated by feed-forward band network   :\t", premium_ffn)

    # ---

    premium_reduced = (premium_ffn - premium_ntb) / premium_ffn * 100
    print("Reduced premium :\t", f"{premium_reduced:.4f} %")
