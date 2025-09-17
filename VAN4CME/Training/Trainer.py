import torch 
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector

def train_minimal(
    crn, net, *, steps=10000, K=4096, lr=3e-4, t_max=1.5, report=200,
    mix_alpha=None, unbiased_is=True, w_clip=None, DEVICE="cpu", beta_time=1., use_w_time=True, gamma=2.0,
    selected_losses = ["residual", "ic", "entropy"], weights=[1.0, 1.0, 0.01]
):
    net.pe.t_max = t_max if hasattr(net.pe, "t_max") else t_max
    opt = torch.optim.Adam(net.parameters(), lr)
    eps = 5e-3
    ent_loss = None

    for step in range(1, steps + 1):
        # times
        t = (torch.rand(K, device=DEVICE, requires_grad=True)**beta_time) * t_max

        # sampling (pure NN vs mixture)
        if mix_alpha is None or float(mix_alpha) >= 0.9999:
            vals, ohs, ents, ps = net.sample_from_model(t)
            q_prob = ps  # q == pθ
            alpha_eff = 1.0
        else:
            alpha_eff = float(mix_alpha)
            vals, ohs, ents, ps, q_prob = net.sample_mixture(t, K, alpha_eff)

        # p and dp/dt at sampled (m,p,t)
        p_cur = net.get_probability(ohs, t)
        dpdt  = torch.autograd.grad(p_cur, t, grad_outputs=torch.ones_like(p_cur), create_graph=True)[0]

        # A p
        Ap = crn.ap_from_net(net, t, vals)

        losses = []
        # base integrand (your relative residual)
        if "residual" in selected_losses:
            integrand = ((dpdt - Ap)**2/(p_cur.detach().clamp_min(eps)**gamma))

            # importance weights to keep target = E_{pθ}[integrand]
            # w = pθ / q, detached to avoid grads through the sampler
            w = (p_cur.detach() / q_prob.clamp_min(1e-12).detach())
            if w_clip is not None:
                w = w.clamp(max=float(w_clip))

            if unbiased_is:
                res_loss = (w * integrand)
            else:
                w_norm = w / (w.mean().clamp_min(1e-12))
                res_loss = (w_norm * integrand)

            if use_w_time:
                w_time = torch.exp(-0.5 * t / t_max)
            else:
                w_time = torch.ones_like(t, device=DEVICE)
            res_loss = (res_loss * w_time).mean()  # time-weighted residual
            losses.append(res_loss)

        if "ic" in selected_losses:
            ic = crn.ic_loss(net)
            losses.append(ic)

        if "entropy" in selected_losses:
            ent = ents.mean() if ents is not None else torch.tensor(0.0, device=DEVICE)
            losses.append(ent)

        # grads = []
        # for loss in losses:
        #     opt.zero_grad()
        #     loss.backward(retain_graph=True)
            # grad = get_gradient_vector(net)
            # print("A")
            # grads.append(grad)
            # print(grad.shape)


        # g_config=ConFIG_update(grads)
        # apply_gradient_vector(net, g_config)
        # print(torch.stack(losses).shape, torch.tensor(weights, device=DEVICE).shape)
        loss = sum(torch.stack(losses) * torch.tensor(weights, device=DEVICE))
        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # tau decay (unchanged)
        if step >= 600 and step % 200 == 0:
            net.tau.mul_(0.995).clamp_(min=1.0)

        # logs
        if step % report == 0 or step == 1:
            with torch.no_grad():
                rmse = (dpdt - Ap).pow(2).mean().sqrt().item()
                d_mean = dpdt.abs().mean().item()
                a_mean = Ap.abs().mean().item()
                # simple ESS estimate (normalized IS)
                ess = w.sum().pow(2) / (w.pow(2).sum().clamp_min(1e-12)).item()

            print(
                f"step {step:5d} | loss {loss.item():.3e} | res {res_loss.item():.3e} | ic {ic.item():.3e} "
                f"| rmse {rmse:.3e} | |dpdt| {d_mean:.3e} | |Ap| {a_mean:.3e} "
                f"| entropy {0 if ent is None else ent.item():.3e} "
                f"| mixα {alpha_eff:.2f} | ESS ~ {ess:.0f}/{K} ({ess/K:.2f})"
            )


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # --- Critic φ_w(s): 1-Lipschitz via spectral normalization on Linear layers
# class Critic(nn.Module):
#     def __init__(self, dim_s, hidden=128, device="cpu"):
#         super().__init__()
#         self.device = device
#         def snlin(nin, nout):  # spectral norm on Linear
#             return nn.utils.parametrizations.spectral_norm(nn.Linear(nin, nout, bias=False))
#         self.net = nn.Sequential(
#             snlin(dim_s, hidden), nn.LeakyReLU(0.2, inplace=True),
#             snlin(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
#             snlin(hidden, 1)
#         )
#         self.to(device)

#     def forward(self, s_float):
#         # s_float: [K, S] float tensor (optionally normalized)
#         return self.net(s_float).squeeze(-1)  # [K]

# def _norm(s, normalize):
#     # optional simple per-species normalization for critic stability
#     if normalize is None:
#         return s.float()
#     return (s.float() - normalize["mean"]) / (normalize["std"].clamp_min(1.0))

# def train_minimal_wasserstein(
#     crn, net, *, steps=10000, K=4096, lr=3e-4, lr_critic=3e-4,
#     t_max=1.5, report=200, mix_alpha=None, unbiased_is=True,
#     w_clip=None, DEVICE="cpu", beta_time=1., use_w_time=True,
#     n_critic=5, gp_lambda=0.0,   # set gp_lambda>0 if you prefer GP over spectral norm
#     gamma_unused=2.0,            # kept for API symmetry; not used now
#     normalize_for_phi=None       # dict with {'mean':..., 'std':...} per species if you want
# ):
#     # time scaling for positional encoders, if any
#     net.pe.t_max = t_max if hasattr(net.pe, "t_max") else t_max

#     opt = torch.optim.Adam(net.parameters(), lr)

#     # --- Critic φ
#     # robustly infer #species
#     dim_s = len(crn.species)

#     phi = Critic(dim_s).to(DEVICE)
#     opt_phi = torch.optim.Adam(phi.parameters(), lr_critic)
#     eps = 5e-12

#     for step in range(1, steps + 1):
#         # --------- (A) Critic updates ----------
#         for _ in range(n_critic):
#             t = (torch.rand(K, device=DEVICE, requires_grad=True)**beta_time) * t_max
#             if mix_alpha is None or float(mix_alpha) >= 0.9999:
#                 vals, ohs, ents, ps = net.sample_from_model(t)
#                 q_prob = ps; alpha_eff = 1.0
#             else:
#                 alpha_eff = float(mix_alpha)
#                 vals, ohs, ents, ps, q_prob = net.sample_mixture(t, K, alpha_eff)

#             p_cur = net.prob_mp(t, vals).clamp_min(0.0)
#             logp  = (p_cur + eps).log()
#             dt_logp = torch.autograd.grad(logp, t, torch.ones_like(logp),
#                                           create_graph=False, retain_graph=False)[0].detach()

#             w = (p_cur.detach() / q_prob.clamp_min(1e-12).detach())
#             if w_clip is not None: w = w.clamp(max=float(w_clip))
#             if not unbiased_is: w = w / (w.mean().clamp_min(1e-12))
#             w_time = torch.exp(-0.5 * t / t_max) if use_w_time else torch.ones_like(t)

#             phi_s  = phi(_norm(vals, normalize_for_phi))
#             Aphi_s = A_of_phi(crn, phi, vals, t, normalize_for_phi)
#             g = phi_s * dt_logp - Aphi_s

#             critic_loss = -(w * w_time * g).mean()
#             if gp_lambda > 0.0:
#                 vals_f = vals.float().requires_grad_(True)
#                 phi_vals = phi(_norm(vals_f, normalize_for_phi)).sum()
#                 grad_s = torch.autograd.grad(phi_vals, vals_f, create_graph=True)[0]
#                 gp = ((grad_s.norm(dim=1) - 1.0)**2).mean()
#                 critic_loss = critic_loss + gp_lambda * gp

#             opt_phi.zero_grad()
#             critic_loss.backward()
#             opt_phi.step()

#         # --------- (B) Model update ----------
#         t = (torch.rand(K, device=DEVICE, requires_grad=True)**beta_time) * t_max
#         if mix_alpha is None or float(mix_alpha) >= 0.9999:
#             vals, ohs, ents, ps = net.sample_from_model(t)
#             q_prob = ps; alpha_eff = 1.0
#         else:
#             alpha_eff = float(mix_alpha)
#             vals, ohs, ents, ps, q_prob = net.sample_mixture(t, K, alpha_eff)

#         p_cur = net.prob_mp(t, vals).clamp_min(0.0)
#         logp  = (p_cur + eps).log()
#         dt_logp = torch.autograd.grad(logp, t, torch.ones_like(logp),
#                                       create_graph=True, retain_graph=True)[0]

#         with torch.no_grad():
#             phi_s  = phi(_norm(vals, normalize_for_phi))
#             Aphi_s = A_of_phi(crn, phi, vals, t, normalize_for_phi)

#         g = phi_s * dt_logp - Aphi_s

#         w = (p_cur.detach() / q_prob.clamp_min(1e-12).detach())
#         if w_clip is not None: w = w.clamp(max=float(w_clip))
#         if not unbiased_is: w = w / (w.mean().clamp_min(1e-12))
#         w_time = torch.exp(-0.5 * t / t_max) if use_w_time else torch.ones_like(t)

#         res_loss = (w * w_time * g).mean()
#         ic  = crn.ic_loss(net)
#         ent = ents.mean() if ents is not None else torch.tensor(0.0, device=DEVICE)
#         loss = res_loss + 5. * ic + 0.1*ent

#         opt.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
#         opt.step()

#         # if step >= 600 and step % 200 == 0 and hasattr(net, "tau"): # TODO enable tau decay
#         #     net.tau.mul_(0.995).clamp_(min=1.0)

#         # ----- logging: compute dp/dt with grads enabled -----
#         if step % report == 0 or step == 1:
#             t_log = t.detach().clone().requires_grad_(True)
#             p_cur_m = net.prob_mp(t_log, vals).clamp_min(0.0)
#             dpdt_m = torch.autograd.grad(p_cur_m, t_log, torch.ones_like(p_cur_m),
#                                          create_graph=False)[0]
#             with torch.no_grad():
#                 Ap_m = crn.ap_from_net(net, t_log, vals)
#                 rmse = (dpdt_m - Ap_m).pow(2).mean().sqrt().item()
#                 d_mean = dpdt_m.abs().mean().item()
#                 a_mean = Ap_m.abs().mean().item()
#                 ess = (w.sum().pow(2) / (w.pow(2).sum().clamp_min(1e-12))).item()
#                 J_val = (w * w_time * g).mean().item()

#             print(
#                 f"step {step:5d} | loss {loss.item():.3e} | J {J_val:.3e} | ic {ic.item():.3e} "
#                 f"| rmse {rmse:.3e} | |dpdt| {d_mean:.3e} | |Ap| {a_mean:.3e} "
#                 f"| entropy {ent.item():.3e} | mixα {alpha_eff:.2f} | ESS ~ {int(ess):d}/{K} ({ess/K:.2f})"
#             )

# # --- helper: turn nu into a list of reaction vectors (each [S])
# def _nu_to_list(nu, S, device):

#     if isinstance(nu, torch.Tensor):
#         if nu.ndim != 2:
#             raise RuntimeError(f"nu must be 2D, got shape {tuple(nu.shape)}")
#         # nu is [S, R]  -> take columns
#         if nu.shape[0] == S:
#             return [nu[:, r].to(device) for r in range(nu.shape[1])]
#         # nu is [R, S]  -> take rows
#         if nu.shape[1] == S:
#             return [nu[r, :].to(device) for r in range(nu.shape[0])]
#         raise RuntimeError(f"nu shape {tuple(nu.shape)} incompatible with S={S}")
#     elif isinstance(nu, (list, tuple)):
#         out = []
#         for v in nu:
#             v = torch.as_tensor(v, device=device)
#             if v.numel() != S:
#                 raise RuntimeError(f"each nu_r must have {S} entries, got {v.numel()}")
#             out.append(v.view(-1))
#         return out
#     else:
#         raise RuntimeError("Unsupported nu type; expected Tensor or list/tuple")

# @torch.no_grad()
# def _valid_mask_after_jump(vals, nu_r):
#     # vals: [K,S] (long/int), nu_r: [S]
#     s_plus = vals + nu_r.view(1, -1)  # <-- ensure correct broadcasting
#     return ((s_plus >= 0).all(dim=-1)), s_plus

# def A_of_phi(crn, phi_net, vals, t, normalize=None):
#     """
#     (A φ)(s) = Σ_r λ_r(s) [ φ(s+ν_r) - φ(s) ]
#       vals: [K,S] int counts
#       returns: [K] float
#     """
#     K, S = vals.shape
#     device = vals.device

#     # propensities: [K, R]
#     lambdas = crn.propensity(vals, t)            # must align with reaction order in nu
#     if lambdas.dim() != 2:
#         raise RuntimeError(f"propensity must return [K,R], got {lambdas.shape}")

#     # build list of ν_r vectors (each [S])
#     nu_list = _nu_to_list(crn.nu, S, device)
#     R = len(nu_list)
#     if lambdas.shape[1] != R:
#         raise RuntimeError(f"Mismatch: propensities have R={lambdas.shape[1]} but nu has R={R}")

#     phi_s = phi_net(_norm(vals, normalize))      # [K]
#     total = torch.zeros(K, device=device)

#     # loop reactions
#     for r, nu_r in enumerate(nu_list):
#         mask, s_plus = _valid_mask_after_jump(vals, nu_r)
#         phi_sp = torch.zeros(K, device=device)
#         if mask.any():
#             phi_sp[mask] = phi_net(_norm(s_plus[mask], normalize))
#         total = total + lambdas[:, r] * (phi_sp - phi_s)
#     return total  # [K]


import copy

def train_minimal_ppo(
    crn, net, *, steps=10000, K=4096, lr=3e-4, t_max=1.5, report=200,
    mix_alpha=None, unbiased_is=True, w_clip=None, DEVICE="cpu", beta_time=1., use_w_time=True, gamma=2.0,
    selected_losses = ["residual", "ic", "entropy"], weights=[1.0, 1.0, 0.01],
    # === PPO-style smoothing params ===
    ppo_clip=0.2,                # epsilon for ratio clipping
    use_ppo_clipping=True,       # turn clipping on/off
    kl_coef=0.0,                 # coefficient for KL(old || new) penalty (0 = off)
    update_old_every=1           # how often to sync behavior (in steps)
):
    net.pe.t_max = t_max if hasattr(net.pe, "t_max") else t_max
    opt = torch.optim.Adam(net.parameters(), lr)
    eps = 5e-12

    # frozen behavior policy (θ_old)
    net_old = copy.deepcopy(net).to(DEVICE)
    for p in net_old.parameters():
        p.requires_grad_(False)
    net_old.eval()

    def maybe_logprob_from_prob(prob):
        # some nets may have a direct logprob head; fallback to safe log
        return torch.log(prob.clamp_min(eps))

    for step in range(1, steps + 1):
        # times
        t = (torch.rand(K, device=DEVICE, requires_grad=True)**beta_time) * t_max

        # === SAMPLE FROM BEHAVIOR POLICY (θ_old) ===
        # If you want to keep your mixture idea, you could sample from net_old for q,
        # then optionally blend with net (advanced). The minimal PPO variant samples from θ_old.
        with torch.no_grad():
            vals_b, ohs_b, ents_b, ps_b = net_old.sample_from_model(t)  # behavior batch

        # Evaluate CURRENT policy on behavior batch
        p_cur = net.get_probability(ohs_b, t)                          # pθ(ohs_b | t)
        dpdt  = torch.autograd.grad(p_cur, t, grad_outputs=torch.ones_like(p_cur), create_graph=True)[0]
        Ap    = crn.ap_from_net(net, t, vals_b)

        with torch.no_grad():
            r = (p_cur / ps_b.clamp_min(eps))
            if use_ppo_clipping:
                r = r.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)

        # Optional hard clipping of raw IS weights too (separate from PPO clipping)
        if w_clip is not None:
            r = r.clamp(max=float(w_clip))

        losses = []

        # base integrand (your relative residual), nonnegative
        if "residual" in selected_losses:
            integrand = ((dpdt - Ap)**2 / (p_cur.detach().clamp_min(5e-3)**gamma))

            # Time weighting (as in your code)
            if use_w_time:
                w_time = torch.exp(-0.5 * t / t_max)
            else:
                w_time = torch.ones_like(t, device=DEVICE)

            w_sur = r                                # no grads through weights

            if not unbiased_is:
                # keep your normalized option if desired (still using clipped ratio before normalizing)
                w_sur = w_sur / (w_sur.mean().clamp_min(1e-12))

            res_loss = (w_sur * integrand * w_time).mean()
            losses.append(res_loss)

        if "ic" in selected_losses:
            ic = crn.ic_loss(net)
            losses.append(ic)

        if "entropy" in selected_losses:
            # ensure inputs are on the right device; ohs_b came from net_old (no grads)
            ohs_b = [x.to(DEVICE) for x in ohs_b]
            # joint entropy H[pθ(x|t)] ≈ sum_i H(pθ(x_i | x_<i>, t)) along behavior contexts
            ent_cur = net.entropy(ohs_b, t)      # shape [K]
            ent_loss = -ent_cur.mean()           # maximize entropy -> minimize (-H)
            losses.append(ent_loss)

        # Optional KL(old || new) penalty for extra smoothness
        if kl_coef and kl_coef > 0.0:
            with torch.no_grad():
                logp_old = torch.log(ps_b.clamp_min(eps))
            logp_new = maybe_logprob_from_prob(p_cur)
            # KL(old || new) = E_old[ log p_old - log p_new ]
            kl = (logp_old - logp_new).mean()
            losses.append(kl_coef * kl)

        loss = (torch.stack(losses) * torch.tensor(weights + ([1.0] if (kl_coef and kl_coef>0) else []), device=DEVICE)).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # sync θ_old periodically
        if (step % update_old_every) == 0:
            net_old.load_state_dict(net.state_dict())

        # tau decay (unchanged)
        if step >= 600 and step % 200 == 0:
            net.tau.mul_(0.995).clamp_(min=1.0)

        # logs
        if step % report == 0 or step == 1:
            with torch.no_grad():
                rmse = (dpdt - Ap).pow(2).mean().sqrt().item()
                d_mean = dpdt.abs().mean().item()
                a_mean = Ap.abs().mean().item()
                # Effective sample size under the *clipped* weights
                w_eff = r
                ess = (w_eff.sum().pow(2) / (w_eff.pow(2).sum().clamp_min(1e-12))).item()

            print(
                f"step {step:5d} | loss {loss.item():.3e} | res {res_loss.item():.3e} | ic {ic.item():.3e} "
                f"| rmse {rmse:.3e} | |dpdt| {d_mean:.3e} | |Ap| {a_mean:.3e} "
                f"| entropy {0 if ent_loss is None else ent_loss.item():.3e} "
                f"| clip {ppo_clip:.2f} | ESS~ {ess:.0f}/{K} ({ess/K:.2f})"
            )
