import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap
from torchrl.modules.tensordict_module.rnn import LSTMCell
from VAN4CME.Modules.PosEnc import PosEnc
from VAN4CME.Utils.Utils import remap, idx_to_onehot

def _entropy_categorical_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Stable categorical entropy from logits.
        logits: [..., C]
        returns: [...]   (entropy per item)
        """
        logp = F.log_softmax(logits, dim=-1)          # [..., C]
        p = logp.exp()                                 # [..., C]
        return -(p * logp).sum(dim=-1)                 # [...]

class SpeciesOperator(nn.Module):

    def __init__(self, max_species, hidden_size, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        max_species = int(max_species)
        self.species_layer = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
        )
        self.max_species = max_species
        self.device = device
        self.to(device)

    def forward(self):
        x = torch.arange(self.max_species, device=self.device).float().unsqueeze(1)
        out = self.species_layer(x)  # [max_species, hidden_size * max_species]
        return out.view(self.max_species, self.hidden_size)

# ---------------- autoregressive pθ(m,p | t) -----------------------
class GSVAN(nn.Module):
    def __init__(self, crn, hidden=128, n_freq=6, tau=1.0, batch_size=256, DEVICE="cpu", use_species_operator=True):
        super().__init__()
        self.crn = crn
        self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=False)
        self.batch_size = batch_size
        self.device = DEVICE
        self.use_species_operator = use_species_operator

        self.pe = PosEnc(n_freq, device=DEVICE)
        self.time_emb = nn.Sequential(
            nn.Linear(self.pe.output_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.lstm = LSTMCell(self.pe.output_size, hidden)

        self.species_projectors = nn.ModuleList([
            nn.Linear(self.crn.species_marginals_sizes[species], self.pe.output_size) for species in self.crn.species
        ])

        self.heads = nn.ModuleList([
            nn.Linear(hidden, self.crn.species_marginals_sizes[species]) for species in self.crn.species
        ])

        self.time_operator = nn.Sequential(
            # nn.Linear(2 * n_freq, hidden), # old pos enc
            nn.Linear(self.pe.output_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.species_operator = nn.ModuleList([
            SpeciesOperator(self.crn.species_marginals_sizes[species], hidden, device=DEVICE) for species in self.crn.species
        ])

        # TODO consider the option to have h0, c0 as parameters
        # self.h0 = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        # self.c0 = nn.Parameter(torch.zeros(hidden), requires_grad=False)

        # buffers
        for species in self.crn.species:
            size = self.crn.species_marginals_sizes[species]
            self.register_buffer(f"{species}_norm", torch.linspace(0, 1, size))
            self.register_buffer(f"{species}_grid", torch.arange(size, device=DEVICE).float())

        self.to(DEVICE)

    def forward(self, t): # used in vmap
        # --- encode time
        x0 = remap(t).to(self.device)
        pe_t = self.pe(x0)
        h = self.time_emb(pe_t); c = h.clone() # initialize the LSTM hidden state

        # autoregressive sampling
        cond = None

        ohs = []
        scalars = []
        entropies = []
        logPs = []

        # t_branch = self.time_operator(pe_t)

        for i, species in enumerate(self.crn.species):
            if cond is not None:
                h, c = self.lstm(cond + pe_t, (h, c))
            else:
                h, c = self.lstm(pe_t, (h, c))

            logits = self._species_logits(h, i)

            dist = torch.distributions.RelaxedOneHotCategorical(temperature=self.tau, logits=logits)
            oh_soft = dist.rsample().squeeze(0)
            oh_hard = F.one_hot(oh_soft.argmax(dim=-1), logits.size(-1)).float()
            oh = oh_hard + (oh_soft - oh_soft.detach())
            scalar = (oh_hard * getattr(self, f"{species}_grid")).sum(-1, keepdim=True)
            cond = self.species_projectors[i](oh_hard)
            ent = torch.distributions.Categorical(logits=logits).entropy()
            logp = torch.log_softmax(logits, dim=-1)
            logp_scalar = (logp * oh_hard).sum(-1, keepdim=True)
            ohs.append(oh)
            scalars.append(scalar)
            entropies.append(ent)
            logPs.append(logp_scalar)

        # gather probabilities

        logPs = torch.stack(logPs, dim=-1).squeeze(1).squeeze(0)   # [S]
        total_prob = logPs.sum().exp() # []
        scalars = torch.stack(scalars).squeeze(-1)       # [S]
        ohs = ohs                            # list(S, [C])
        entropies = torch.stack(entropies).squeeze(-1)   # [S]

        # shapes:  torch.Size([2]) torch.Size([2]) torch.Size([2]) torch.Size([])
        # print("shapes: ", scalars.shape, entropies.shape, logPs.shape, total_prob.shape)

        return scalars, ohs, entropies, total_prob

    def _species_logits(self, h: torch.Tensor, i: int) -> torch.Tensor:
        """
        Get logits for species i given hidden state h.
        h: [B, hidden] or [hidden] (broadcasted)
        returns logits: [B, C_i]
        """
        if self.use_species_operator:
            so = self.species_operator[i]()           # [C_i, hidden]
            logits = h @ so.T                          # [B, C_i] (or [C_i] if h is [hidden])
        else:
            logits = self.heads[i](h)                  # [B, C_i] (or [C_i])
        # Ensure batch dimension for scalar t
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits
    
    def entropy(self, ohs, t, return_components: bool = False):
        """
        Joint entropy H[p_theta(x|t)] estimated under provided behavior contexts.
        - ohs: list length S with one-hots [B, C_i] for species 0..S-1 from behavior policy
        - t:   [B] times (or scalar); gradients flow only through current network
        Returns:
            total_entropy: [B]
            (optional) per_species_entropies: [B, S]
        """
        # Make t a batch
        if torch.is_tensor(t):
            t_vec = t if t.ndim == 1 else t.unsqueeze(0)
        else:
            t_vec = torch.tensor([float(t)], device=self.device)

        # Time encoding and initial state
        x0   = remap(t_vec).to(self.device)
        pe_t = self.pe(x0)
        h    = self.time_emb(pe_t)          # [B, hidden]
        c    = h.clone()

        ents = []                           # list of [B]
        for i, _ in enumerate(self.crn.species):
            # LSTM step with teacher-forced context from behavior ohs
            if i == 0:
                h, c = self.lstm(pe_t, (h, c))
            else:
                m_cond = self.species_projectors[i - 1](ohs[i - 1])  # [B, 2*n_freq]
                h, c = self.lstm(pe_t + m_cond, (h, c))

            logits_i = self._species_logits(h, i)        # [B, C_i]
            H_i = _entropy_categorical_from_logits(logits_i)   # [B]
            ents.append(H_i)

        ents = torch.stack(ents, dim=-1)   # [B, S]
        total = ents.sum(dim=-1)           # [B]
        return (total, ents) if return_components else total


    def get_probability(self, ohs, t):
        x0 = remap(t).to(self.device)
        pe_t = self.pe(x0)
        h = self.time_emb(pe_t); c = h.clone()

        probs = []
        for i, species in enumerate(self.crn.species):
            # p(x|x_prev, t)
            if i == 0:
                h, c = self.lstm(pe_t, (h, c))
            else:
                m_cond = self.species_projectors[i - 1](ohs[i - 1])
                cond_p = pe_t + m_cond
                h, c = self.lstm(cond_p, (h, c))

            logits = self._species_logits(h, i)

            logp = torch.log_softmax(logits, dim=-1)
            # print(f"logp shape: {logp.shape}, ohs shape: {ohs.shape}, i: {i}")
            logp_scalar = (logp * ohs[i]).sum(dim=-1)  # [B]
            probs.append(logp_scalar)

        return torch.stack(probs, dim=-1).sum(dim=-1).exp()
    
    # ------- add this helper next to sample_from_model -----------------
    def sample_mixture(self, t, K, alpha):
        """Return a mixture of NN and uniform samples plus q-prob for IS."""

        Km = int(round(K * alpha))
        Ku = K - Km
        assert Km >= 0 and Ku >= 0

        # split times (both parts keep requires_grad from t)
        tm, tu = t[:Km], t[Km:]

        tm = tm.to(self.device)
        tu = tu.to(self.device)

        # NN part (same as before)
        if Km > 0:
            nns, oh_nns, ent_nns, ptheta_nns = self.sample_from_model(tm)
        else:
            nns = oh_nns = ent_nns = ptheta_nns = None

        # Uniform part
        if Ku > 0:
            us = []
            oh_us = []
            ent_us = []
            ptheta_us = []
            for species in self.crn.species:
                limit = self.crn.species_marginals_sizes[species]
                us.append(torch.randint(0, limit, (Ku,), device=self.device))
                oh_us.append(idx_to_onehot(us[-1], limit))
                ent_us.append(torch.zeros(Ku, device=self.device))
            us = torch.stack(us, dim=-1)                       # [Ku, species]
            # oh_us this remains a list of [Ku, C] tensors
            _, ent_us = self.entropy(oh_us, tu, return_components=True)  # [Ku, S]     
            ptheta_us = self.get_probability(oh_us, tu)
            
        else:
            us = oh_us = ent_us = ptheta_us = None

        # concat everything in original order: [NN | UNI]

        # print(f"nn_types = {[type(a) for a in [nns, oh_nns, ent_nns, ptheta_nns]]}")
        # print(f"uniform types = {[type(a) for a in [us, oh_us, ent_us, ptheta_us]]}")
        # print(f"nn shapes = {[a.shape for a in [nns, oh_nns, ent_nns, ptheta_nns]]}")
        # print(f"uniform shapes = {[a.shape for a in [us, oh_us, ent_us, ptheta_us]]}")

        def cat_not_none(*args):
            catlist = [a for a in args if a is not None]
            if len(catlist) == 0:
                return None
            if len(catlist) == 1:
                return catlist[0]
            # print(f"Concatenating {len(catlist)} tensors of shapes {[a.shape for a in catlist]}")
            return torch.cat([a for a in args if a is not None], dim=0)
    
        s = cat_not_none(nns, us)

        if oh_nns is None: 
            oh_s = oh_us
        elif oh_us is None:
            oh_s = oh_nns
        else:
            oh_s = [cat_not_none(n, u) for n, u in zip(oh_nns, oh_us)]
            
        ent_s = cat_not_none(ent_nns, ent_us)
        ptheta = cat_not_none(ptheta_nns, ptheta_us)

        # mixture q = α pθ + (1-α) U
        u_prob = 1.0 / float(self.crn.N_STATES)
        q_prob = alpha * ptheta.detach() + (1.0 - alpha) * u_prob

        
        return s, oh_s, ent_s, ptheta, q_prob

    def prob_mp(self, t_vec: torch.Tensor, species_vec: torch.Tensor):
        """
        Compute p_theta(x | t) from integer species values.

        Args:
            t_vec:      [B] time tensor (any dtype, on same device as model)
            species_vec:[B, S] integer tensor with one column per species, in the
                        same order as self.crn.species.

        Returns:
            p_theta for those states at times t_vec, via self.get_probability(ohs, ohs, t_vec).
        """
        # Expect species_vec to be on the same device as the model
        device = species_vec.device
        species = self.crn.species
        sizes = self.crn.species_marginals_sizes  # dict: name -> num_classes

        # Build one-hot per species in a single vectorized call (one loop over S only).
        oh_list = []
        for i, sp in enumerate(species):
            C = int(sizes[sp])
            idx = species_vec[:, i].clamp(0, C - 1).to(device=device, dtype=torch.long)
            oh_i = F.one_hot(idx, num_classes=C).to(dtype=torch.float32)  # [B, C]
            oh_list.append(oh_i)                             # [B, C]

        # Concatenate across species -> list(S, [B, C]);
        return self.get_probability(oh_list, t_vec)

    
    def sample_from_model(self, t):
        """Exact same sampling you used (via net.forward + vmap)."""
        vals, ohs, ents, ps = vmap(
            lambda t: self(t), (0,), randomness="different"
        )(t)

        # squeeze to match later usage
        vals = vals.squeeze(-1)  # [K, species]
        ps = ps.squeeze(-1)  # [K]
        # print(vals.shape, ps.shape)
        return vals, ohs, ents, ps

    # ---- helpers ----------------------------------------------------
    def _species_index(self, target):
        """target can be species name (str) or index (int)."""
        if isinstance(target, int):
            return int(target)
        return list(self.crn.species).index(str(target))

    # ---- 1) Marginal estimator: sample others, not the target -------
    @torch.no_grad()
    def sample_marginal(self, t, target, K=8192, return_all=False, chunk_size=None):
        """
        Estimate the marginal p_theta(x_target | t) via Monte Carlo:
        Draw K ancestral samples for species 0..target-1 from the model,
        stop at 'target', read the target head's softmax, and average.

        Args:
            t:           scalar (float or 0-dim tensor) or [K]-tensor of times
            target:      species name (str) or index (int)
            K:           total MC samples (ignored if t is a [K]-tensor)
            return_all:  if True, also returns stacked per-sample conditionals [K, C_target]
            chunk_size:  process in chunks to limit peak VRAM (e.g., 2048 or 65536)

        Returns:
            marginal: [C_target] tensor summing to 1
            (optionally) conds: [K, C_target] per-sample P(x_target | x_<target>, t)
        """
        # resolve target index + cardinality
        if isinstance(target, int):
            s_idx = int(target)
        else:
            s_idx = list(self.crn.species).index(str(target))
        C_tgt = int(self.crn.species_marginals_sizes[self.crn.species[s_idx]])

        # time vector
        if torch.is_tensor(t) and t.ndim == 1:
            t_vec = t.to(self.device)
            K = t_vec.numel()
        else:
            t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
            t_vec = torch.full((K,), t_scalar, device=self.device)

        # chunking plan
        if chunk_size is None or chunk_size >= K:
            starts = [0]; sizes = [K]
        else:
            bs = int(chunk_size)
            starts = list(range(0, K, bs))
            sizes = [min(bs, K - s) for s in starts]

        acc = torch.zeros(C_tgt, device=self.device, dtype=torch.float64)  # sum of conditionals
        cond_rows = [] if return_all else None
        S = len(self.crn.species)

        for s, bs in zip(starts, sizes):
            tt = t_vec[s:s+bs]  # [B]

            # time encoding
            x0   = remap(tt).to(self.device)
            pe_t = self.pe(x0)
            h    = self.time_emb(pe_t)
            c    = h.clone()
            cond = None

            # walk autoregressively up to the target head
            for i in range(S):
                # LSTM step
                if i == 0:
                    h, c = self.lstm(pe_t, (h, c))
                else:
                    h, c = self.lstm(pe_t + (cond if cond is not None else 0.0), (h, c))

                if i < s_idx:
                    # sample species i from the model (ancestral)
                    logits_i = self._species_logits(h, i)                      # [B, C_i]
                    idx_i = torch.distributions.Categorical(logits=logits_i).sample()
                    oh_i  = F.one_hot(idx_i, num_classes=logits_i.size(-1)).to(torch.float32)
                    cond  = self.species_projectors[i](oh_i)         # condition next step
                    continue

                # i == s_idx -> read target conditional and stop
                logits_t = self._species_logits(h, s_idx)            # [B, C_tgt]
                probs_t  = torch.softmax(logits_t, dim=-1)           # [B, C_tgt]
                acc += probs_t.sum(dim=0, dtype=torch.float64)
                if return_all:
                    cond_rows.append(probs_t.detach().cpu())
                break

        # average over K, and renormalize just in case of tiny drift
        marginal = (acc / float(K)).to(dtype=torch.float32)
        s = marginal.sum()
        if s > 0:
            marginal = marginal / s

        if return_all:
            conds = torch.cat(cond_rows, dim=0) if cond_rows else torch.empty(0, C_tgt)
            return marginal, conds
        return marginal


    # ---- 2) Conditional sampling: clamp one species -----------------
    # @torch.no_grad()
    # def sample_with_clamp(self, t, K, clamp, return_logp=False):
    #     """
    #     Sample full states from p_theta(x | t) with one species clamped:
    #     x_target = value, while all others are sampled ancestrally.

    #     Args:
    #         t:      scalar or [K]-tensor
    #         K:      number of samples (ignored if t is [K])
    #         clamp:  dict like {'P': 7} or {'M': 0} or {'S3': 12} etc.
    #         return_logp: also return joint log-prob of the sampled/clamped state

    #     Returns:
    #         vals:   [K, S] integer samples
    #         ohs:    list length S of [K, C_i] one-hots
    #         (optional) logp: [K] joint log prob under the model
    #     """
    #     device = self.device
    #     (name, value), = clamp.items()
    #     s_idx  = self._species_index(name)
    #     C_tgt  = int(self.crn.species_marginals_sizes[self.crn.species[s_idx]])
    #     value  = int(value)

    #     # time vector
    #     if torch.is_tensor(t) and t.ndim == 1:
    #         t_vec = t.to(device)
    #         K = t_vec.numel()
    #     else:
    #         t_scalar = float(t) if not torch.is_tensor(t) else float(t.item())
    #         t_vec = torch.full((K,), t_scalar, device=device)

    #     # time encoding
    #     x0   = remap(t_vec).to(device)
    #     pe_t = self.pe(x0)
    #     h    = self.time_emb(pe_t); c = h.clone()

    #     vals = []
    #     ohs  = []
    #     logps = []

    #     cond = None
    #     for i, sp in enumerate(self.crn.species):
    #         if i == 0:
    #             h, c = self.lstm(pe_t, (h, c))
    #         else:
    #             h, c = self.lstm(pe_t + cond, (h, c))

    #         logits = self.heads[i](h)
    #         logp   = torch.log_softmax(logits, dim=-1)

    #         if i == s_idx:
    #             idx_i = torch.full((K,), value, device=device, dtype=torch.long)
    #             oh_i  = F.one_hot(idx_i, num_classes=C_tgt).float()
    #             lp_i  = logp.gather(1, idx_i.unsqueeze(1)).squeeze(1)
    #         else:
    #             # ancestral sample
    #             dist  = torch.distributions.RelaxedOneHotCategorical(temperature=self.tau, logits=logits)
    #             oh_s  = dist.rsample()
    #             idx_i = oh_s.argmax(dim=-1)
    #             oh_i  = F.one_hot(idx_i, num_classes=logits.size(-1)).float()
    #             lp_i  = logp.gather(1, idx_i.unsqueeze(1)).squeeze(1)

    #         cond = self.species_projectors[i](oh_i)
    #         vals.append(idx_i)
    #         ohs.append(oh_i)
    #         logps.append(lp_i)

    #     vals  = torch.stack(vals, dim=1)   # [K, S]
    #     if return_logp:
    #         logp = torch.stack(logps, dim=-1).sum(dim=-1)
    #         return vals, ohs, logp
    #     return vals, ohs