# No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/pfnet-reseaarch/NoTransactionBandNetwork/main.ipynb)

Minimal implementation and experiments of "*No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging*".

## TL;DR

* [Deep Hedging](https://arxiv.org/abs/1802.03042) is a deep learning-based framework to compute the optimal hedging strategies of financial derivatives.
* However, this optimal strategy is hard to train due to the action dependence, *i.e.*, the appropriate hedging action at the next step depends on the current action.
* We propose a "*No-Transaction Band Network*" to overcome this issue.
* This network circumvents the complication of action-dependence and facilitate quick and precise computation of the optimal hedging.

The learning histories below demonstrate that the no-transaction band network can be trained much quicker than the ordinary feed-forward network (See [`main.ipynb`](main.ipynb) for details).

![loss_lookback](fig/loss_lookback.png)

## Proposed Architecture: No-Transaction Band Network

The following figures show the schematic diagrams of the neural network which was originally proposed in [Deep Hedging](https://arxiv.org/abs/1802.03042) (left) and the proposed network (right).

![nn](fig/nn.png)

* **The original architecture**:
  - The input of the neural network uses the current hedge ratio (`δ_ti`) as well as other information (`I_ti`) a human trader might use.
  - Since the input of the neural network includes on the current action, this architecture bears the complication of action-dependence.
* **The no-transaction band network**:
  - This architecture computes "no-transaction band" `[b_l, b_u]` by a neural network and then computes the next hedge ratio by `clamp`(https://pytorch.org/docs/stable/generated/torch.clamp.html?highlight=clamp#torch.clamp)ing the current hedge ratio inside this band.
  - Since the input of the neural network does not use the current action, this architecture can circumvent the action-dependence.

## Give it a Try!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/pfnet-reseaarch/NoTransactionBandNetwork/main.ipynb)

The minimal implementation and experiments are provided in [`main.ipynb`](main.ipynb).

More comprehensive library for Deep Hedging, `pfhedge`, is available on PyPI (See [pfnet-research/pfhedge](https://github.com/pfnet-research/pfhedge)).

## References

* Shota Imaki, Kentaro Imajo, Katsuya Ito, Kentaro Minami and Kei Nakagawa, "No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging".
* 今木翔太, 今城健太郎, 伊藤克哉, 南賢太郎, 中川慧, "[効率的な Deep Hedging のためのニューラルネットワーク構造](https://sigfin.org/026-15/)", 人工知能学 金融情報学研究会（SIG-FIN）第 26 回研究会.
* Hans Bühler, Lukas Gonon, Josef Teichmann and Ben Wood, "[Deep hedging](https://doi.org/10.1080/14697688.2019.1571683)". Quantitative Finance, 2019, 19, 1271–1291. arXiv:[1609.05213](https://arxiv.org/abs/1802.03042) [q-fin.CP].
