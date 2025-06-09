import torch
import torch.nn as nn
import torch.nn.functional as F

class Loc2VecModel(nn.Module):
    def __init__(self, input_channels: int = 3, embedding_dim: int = 16, dropout_rate: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 64, 1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(1024, 64, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, embedding_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoftmaxTripletLoss(nn.Module):
    """
    Triplet Soft-max ratio loss (Ailon et al.) with optional SoftPN variant.
    Minimises  MSE( (d_plus, d_minus), (0, 1) ).

    Args
    ----
    softpn : bool
        If True, use SoftPN (replace Δ(a,n) by min(Δ(a,n), Δ(p,n))).
    squared : bool
        If True, use squared Euclidean distance; else plain L2.
    reduction : str
        'mean' | 'sum' | 'none'   (mirrors PyTorch's reduction semantics)
    eps : float
        Numerical stabiliser added to denominator.
    """

    def __init__(self,
                 softpn: bool = False,
                 squared: bool = True,
                 reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.softpn = softpn
        self.squared = squared
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def _l2(a: torch.Tensor,
            b: torch.Tensor,
            squared: bool) -> torch.Tensor:
        out = (a - b).pow(2).sum(dim=1)
        return out if squared else out.clamp_min(1e-12).sqrt()

    def forward(self,
                anchor:   torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        anchor, positive, negative  : shape (B, embedding_dim)
        returns scalar loss (or per-sample loss if reduction='none')
        """
        delta_ap = self._l2(anchor, positive, self.squared)
        delta_an = self._l2(anchor, negative, self.squared)

        if self.softpn:
            delta_pn = self._l2(positive, negative, self.squared)
            delta_neg = torch.min(delta_an, delta_pn)
        else:
            delta_neg = delta_an

        exp_ap  = torch.exp(delta_ap)
        exp_neg = torch.exp(delta_neg)
        denom   = exp_ap + exp_neg + self.eps

        d_plus  = exp_ap  / denom               # expected → 0
        d_minus = exp_neg / denom               # expected → 1

        loss_vec = (d_plus ** 2) + ((d_minus - 1) ** 2)  # MSE vs (0,1)

        if self.reduction == 'mean':
            return loss_vec.mean()
        elif self.reduction == 'sum':
            return loss_vec.sum()
        else:                                   # 'none'
            return loss_vec

class Loc2VecTripletLoss(nn.Module):
    def __init__(self, pos_target=0.0, neg_target=1.0):
        super().__init__()
        self.pos_target = pos_target
        self.neg_target = neg_target

    def forward(self, anchor_i, anchor_p, anchor_n):
        distance_i_p = F.pairwise_distance(anchor_i, anchor_p)
        distance_i_n = F.pairwise_distance(anchor_i, anchor_n)

        loss = ((distance_i_p - self.pos_target)**2 + (distance_i_n - self.neg_target)**2).mean()

        np_distance_a_pos = distance_i_p.mean().item()
        np_distance_a_neg = distance_i_n.mean().item()

        loss_log = f'LOSS: {loss.item():.3f} | (+) DIST: {np_distance_a_pos:.3f} | (-) DIST: {np_distance_a_neg:.3f}'

        return loss #  loss_log # np_distance_a_pos, np_distance_a_neg, distance_i_n.min().item()


if __name__ == "__main__":
    # Example usage
    model = Loc2VecModel(input_channels=3, embedding_dim=64, dropout_rate=0.5)
    print(model)

    # Example forward pass
    dummy_input = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (1, 64) for embedding_dim=64
