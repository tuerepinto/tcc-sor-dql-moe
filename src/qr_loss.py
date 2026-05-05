import torch
import torch.nn.functional as F

def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    pred:   (B, NQ)  quantis previstos para uma ação
    target: (B, NQ)  quantis-alvo
    taus:   (NQ,)    frações dos quantis (0..1)
    """
    # pairwise TD errors: (B, NQ, NQ)
    td = target.unsqueeze(1) - pred.unsqueeze(2)

    huber = torch.where(
        td.abs() <= kappa,
        0.5 * td.pow(2),
        kappa * (td.abs() - 0.5 * kappa)
    )
    # quantile regression weighting
    tau = taus.view(1, -1, 1)
    loss = (torch.abs(tau - (td.detach() < 0).float()) * huber) / kappa
    return loss.mean()