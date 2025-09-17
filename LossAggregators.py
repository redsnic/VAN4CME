import torch

def _flat_std_from_grads(grads, device):
    # grads: list of tensors or None; returns std of concatenated grads
    flat_chunks = []
    for g in grads:
        if g is None:
            continue
        # Use .reshape(-1) to avoid copy; no need to clone
        flat_chunks.append(g.reshape(-1))
    if not flat_chunks:  # all None
        return torch.tensor(0.0, device=device)
    flat = torch.cat(flat_chunks)
    return flat.std()

class IVOLossAggregator:
    """
    Importance-variance optimizer for multi-loss weighting.
    Weights are nudged so losses with smaller grad STD get larger weights.
    """
    def __init__(self, model, losses, lambda_lr=0.2, logger=None, max_weight_per_loss=None, freeze_losses=None):
        self.model = model
        self.losses = list(losses)
        self.lambda_lr = float(lambda_lr)
        self.logger = logger
        self.max_weight_per_loss = max_weight_per_loss or {}
        self.freeze_losses = set(freeze_losses or [])

        # Put weights on the same device/dtype as model parameters
        device = next(model.parameters()).device
        self.weights = torch.ones(len(self.losses), device=device, requires_grad=False)

        # Cache params list once
        self._params = [p for p in model.parameters() if p.requires_grad]

        

    def __call__(self, losses_values: torch.Tensor):
        # losses_values: tensor shape [L]
        # Match device/dtype to losses_values to avoid casting surprises
        w = self.weights.to(device=losses_values.device, dtype=losses_values.dtype)
        return (losses_values * w).sum()

    @torch.no_grad()
    def reset(self):
        device = self.weights.device
        self.weights = torch.ones_like(self.weights, device=device)

    def step(self, losses_values, optimizer):
        """
        Update self.weights based on per-loss gradient std, WITHOUT touching .grad.
        `losses_values` is an iterable of scalar tensors that share the same graph
        as the final combined loss you will backprop later.
        """
        L = len(self.losses)
        if L <= 1:
            return

        # Compute per-loss grads w.r.t. model parameters without populating .grad
        stds = []
        for i, loss_scalar in enumerate(losses_values):
            grads = torch.autograd.grad(
                loss_scalar,
                self._params,
                retain_graph=True,     # keep graph for next losses and the final backward
                allow_unused=True,     # some params may not be used by a given loss
                create_graph=False     # we just need numbers for stds
            )
            std_i = _flat_std_from_grads(grads, device=self.weights.device)
            # Basic NAN/INF guard
            if torch.isnan(loss_scalar).any() or torch.isnan(std_i).any() or torch.isinf(std_i).any():
                raise ValueError(f"NAN/INF detected in loss '{self.losses[i]}' or its grad std.")
            stds.append(std_i)

        stds_tensor = torch.stack(stds)  # [L]
        max_std = stds_tensor.max()

        # Avoid divide-by-zero; keep things finite
        lambda_hats = (max_std / (stds_tensor + 1e-6)).detach()

        # freeze or cap certain losses
        with torch.no_grad():
            for i, name in enumerate(self.losses):
                if name in self.freeze_losses:
                    lambda_hats[i] = self.weights[i]            # keep weight fixed
                elif name in self.max_weight_per_loss:
                    lambda_hats[i] = torch.minimum(
                        lambda_hats[i],
                        torch.as_tensor(self.max_weight_per_loss[name], device=lambda_hats.device)
                    )

            # optional: renormalize weights to keep mean ~ 1
            # lambda_hats *= (len(lambda_hats) / lambda_hats.sum().clamp_min(1e-6))

            self.weights.mul_(1 - self.lambda_lr).add_(self.lambda_lr * lambda_hats)
