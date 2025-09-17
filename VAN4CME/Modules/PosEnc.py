import math, torch
import torch.nn as nn

# ---------------- positional encoding ------------------------------
# class PosEnc(nn.Module):
#     def __init__(self, n_freq=6, device="cpu"):
#         super().__init__()
#         self.register_buffer("f", 2 ** torch.arange(n_freq).float() * math.pi)
#         self.device = device
#         self.output_size = 2 * n_freq
#         self.to(device)
#         self.f.to(device)

#     def forward(self, t):
#         t = t[..., None].to(self.device)  # [B,1]
#         return torch.cat([torch.sin(t * self.f), torch.cos(t * self.f)], dim=-1)


import math, torch
import torch.nn as nn

class PosEnc(nn.Module):  # with decay, shape-safe & NaN-safe
    def __init__(self, n_freq=6, n_decay=4,
                 f_min=1/1024, f_max=1/4,
                 lam_min=1e-4, lam_max=1.0,
                 learn_params=True, device="cpu"):
        super().__init__()
        omega = 2*math.pi*torch.logspace(math.log10(f_min), math.log10(f_max), n_freq)
        lam = torch.logspace(math.log10(lam_min), math.log10(lam_max), n_decay)
        self.register_buffer("omega", omega)
        self.register_buffer("lam", lam)

        if learn_params:
            self.log_s   = nn.Parameter(torch.tensor(0.0))     # time stretch
            self.log_lam = nn.Parameter(torch.log(lam))        # learnable decay rates
        else:
            self.log_s   = None
            self.log_lam = None

        self.n_freq = n_freq
        self.n_decay = n_decay
        self.output_size = 1 + 2*n_freq + n_decay + 2*n_decay*n_freq
        self.to(device)

    def forward(self, t):  # t: scalar () or [B], [B,1], [B,1,1], ...
        # --- sanitize t -> [B,1], match dtype/device ---
        t = t.to(self.omega.device, self.omega.dtype).reshape(-1, 1)  # [B,1]

        # --- stable scaling ---
        if self.log_s is not None:
            log_s = torch.clamp(self.log_s, -10.0, 10.0)     # s in [e^-10, e^10]
            inv_scale = torch.exp(-log_s)                    # multiply instead of divide
        else:
            inv_scale = torch.tensor(1.0, device=t.device, dtype=t.dtype)

        # phase: [B,K], wrapped to avoid huge arguments
        x = t * (self.omega[None, :] * inv_scale)            # [B,K]
        x = torch.remainder(x, 2*math.pi)                    # wrap to [0, 2π)

        # --- stable decays ---
        if self.log_lam is not None:
            log_lam = torch.clamp(self.log_lam, -20.0, 10.0) # lam ~ [2e-9, 2e4]
            lam = torch.exp(log_lam)                         # [D]
        else:
            lam = self.lam                                   # [D]

        t_pos = torch.clamp(t, min=0.0)                      # no blow-up for negative times
        expo = (-t_pos * lam[None, :]).clamp(-60.0, 60.0)    # [B,D]
        decay = torch.exp(expo)                               # [B,D]

        # --- features (all 2-D) ---
        dc = torch.ones_like(t)                               # [B,1]
        s  = torch.sin(x)                                     # [B,K]
        c  = torch.cos(x)                                     # [B,K]
        ds = (decay[..., None] * s[:, None, :]).reshape(t.shape[0], -1)  # [B,D*K]
        dc_ = (decay[..., None] * c[:, None, :]).reshape(t.shape[0], -1) # [B,D*K]

        out = torch.cat([dc, s, c, decay, ds, dc_], dim=-1)   # [B, 1+2K + D + 2DK]
        # Uncomment to catch issues early:
        # assert out.dim() == 2 and out.shape[1] == self.output_size
        # if not torch.isfinite(out).all(): raise RuntimeError("NaN/Inf in PosEnc")
        return out
