import math
import torch
import torch.nn.functional as F
from functools import reduce
from operator import mul


class RepressilatorCRN:
    """
    6-species repressilator:
      S1=mRNA1, S2=Prot1, S3=mRNA2, S4=Prot2, S5=mRNA3, S6=Prot3

    Reactions (columns 0..11), stoichiometry ν (S x R):
      r1 : ∅  -> S1               rate: phi1/(1+S6^2) + phi2
      r2 : S1 -> ∅                rate: k2*S1
      r3 : S1 -> S1 + S2          rate: k3*S1
      r4 : S2 -> ∅                rate: k4*S2

      r5 : ∅  -> S3               rate: phi1/(1+S2^2) + phi2
      r6 : S3 -> ∅                rate: k6*S3
      r7 : S3 -> S3 + S4          rate: k7*S3
      r8 : S4 -> ∅                rate: k8*S4

      r9  : ∅  -> S5              rate: phi1/(1+S4^2) + phi2
      r10 : S5 -> ∅               rate: k10*S5
      r11 : S5 -> S5 + S6         rate: k11*S5
      r12 : S6 -> ∅               rate: k12*S6
    """

    def __init__(
        self,
        *,
        phi1=50.0,
        phi2=0.0,
        k2=1.0, k3=5.0, k4=1.0,
        k6=1.0, k7=5.0, k8=1.0,
        k10=1.0, k11=5.0, k12=1.0,
        max_count=[20, 200, 20, 200, 20, 200],  # [S1, S2, S3, S4, S5, S6]
        device="cpu",
        init_state=None                # dict or None; defaults to provided one below
    ):
        self.device = torch.device(device)

        # parameters
        self.phi1 = float(phi1)
        self.phi2 = float(phi2)
        self.k2   = float(k2);   self.k3  = float(k3);   self.k4  = float(k4)
        self.k6   = float(k6);   self.k7  = float(k7);   self.k8  = float(k8)
        self.k10  = float(k10);  self.k11 = float(k11);  self.k12 = float(k12)

        # species & sizes
        self.species = ["S1", "S2", "S3", "S4", "S5", "S6"]
        self.species_marginals_sizes = {s: C for s, C in zip(self.species, max_count)}
        self._sizes_vec = torch.tensor(max_count, device=self.device)    # [S]
        self.N_STATES = int(reduce(mul, (C for C in max_count), 1))

        # initial state (enforced at t=0)
        if init_state is None:
            init_state = {'S1': 1, 'S2': 50, 'S3': 0, 'S4': 0, 'S5': 0, 'S6': 0}
        self.init_state = {k: int(v) for k, v in init_state.items()}

        # stoichiometry ν (S x R)
        nu = torch.zeros((6, 12), dtype=torch.long)
        # r1..r4 (module 1)
        nu[0, 0] = +1         # r1: +S1
        nu[0, 1] = -1         # r2: -S1
        nu[1, 2] = +1         # r3: +S2
        nu[1, 3] = -1         # r4: -S2
        # r5..r8 (module 2)
        nu[2, 4] = +1         # r5: +S3
        nu[2, 5] = -1         # r6: -S3
        nu[3, 6] = +1         # r7: +S4
        nu[3, 7] = -1         # r8: -S4
        # r9..r12 (module 3)
        nu[4, 8]  = +1        # r9: +S5
        nu[4, 9]  = -1        # r10: -S5
        nu[5,10]  = +1        # r11: +S6
        nu[5,11]  = -1        # r12: -S6
        self.nu = nu.to(self.device)  # [6,12]

    # ---------- helpers ----------
    @torch.no_grad()
    def _bounds(self):
        # max index per species (inclusive)
        return (self._sizes_vec - 1)

    # ---------- propensities a_j(x) ----------
    def _propensities(self, x_int: torch.Tensor) -> torch.Tensor:
        """
        x_int: [K,6] integer counts
        returns a(x): [K,12] float propensities
        """
        x = x_int.to(self.device).to(dtype=torch.float32)
        S1, S2, S3, S4, S5, S6 = [x[:, i] for i in range(6)]
        # broadcast-safe shapes
        a1  = self.phi1 / (1.0 + S6*S6) + self.phi2
        a2  = self.k2  * S1
        a3  = self.k3  * S1
        a4  = self.k4  * S2

        a5  = self.phi1 / (1.0 + S2*S2) + self.phi2
        a6  = self.k6  * S3
        a7  = self.k7  * S3
        a8  = self.k8  * S4

        a9  = self.phi1 / (1.0 + S4*S4) + self.phi2
        a10 = self.k10 * S5
        a11 = self.k11 * S5
        a12 = self.k12 * S6

        return torch.stack([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12], dim=1)  # [K,12]

    # ---------- generator applied to p_theta via the NN ----------
    def ap_from_net(self, net, t, vals: torch.Tensor) -> torch.Tensor:
        """
        Compute (A p_theta)(x,t) for a batch of states x = vals using the generic CME form:
          (Ap)(x) = sum_j a_j(x - ν_j) p(x - ν_j, t) - [sum_j a_j(x)] p(x, t)
        Inputs:
            net  : model exposing prob_mp(t_vec, species_vec=[K,6])
            t    : scalar or [K] tensor
            vals : [K,6] integer tensor of counts in species order self.species
        Returns:
            tensor [K] with (A p_theta)(vals, t)
        """
        device = self.device
        vals = vals.to(device).long()
        K = vals.shape[0]

        # time vector
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=device)
        t = t.to(device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(K)

        # current prob and outflow
        a_curr = self._propensities(vals)                 # [K,12]
        p_curr = net.prob_mp(t, vals).clamp_min(0.0)      # [K]
        outflow = (a_curr.sum(dim=1)) * p_curr            # [K]

        # inflow: for each reaction j, source y = x - ν_j
        inflow = torch.zeros(K, device=device, dtype=torch.float32)
        bounds = self._bounds()                            # [6]
        for j in range(self.nu.shape[1]):
            nu_j = self.nu[:, j]                          # [6]
            y = vals - nu_j.view(1, -1)                   # [K,6]

            # mask states that fall outside the box
            valid = ((y >= 0) & (y <= bounds)).all(dim=1).to(torch.float32)  # [K]
            if valid.sum() == 0:
                continue

            a_src = self._propensities(y)[:, j]           # [K]
            p_src = net.prob_mp(t, y).clamp_min(0.0)      # [K]
            inflow = inflow + (a_src * p_src * valid)

        return inflow - outflow

    # ---------- initial-condition loss (NLL at the provided state) ----------
    def ic_loss(self, net, B: int = 1024):
        """
        Encourage p_theta(x0 | t=0) ≈ 1 where
          x0 = {'S1':1, 'S2':50, 'S3':0, 'S4':0, 'S5':0, 'S6':0} (by default).
        Uses -log p averaged over B replicas.
        """
        # build [B,6] integer batch at x0
        x0_vec = torch.tensor(
            [self.init_state[s] for s in self.species],
            device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)  # [B,6]

        t0 = torch.zeros(B, device=self.device)
        p0 = net.prob_mp(t0, x0_vec).clamp_min(1e-12)
        return -(p0.log()).mean()
    

    # put this inside RepressilatorCRN
    def propensity(self, x_int: torch.Tensor, t=None) -> torch.Tensor:
        """Public alias used by the training/critic code. Time-independent here."""
        return self._propensities(x_int)

    @property
    def nu_cols(self):
        """List of ν_j (each [S]) so you can iterate reaction-wise if you prefer."""
        return [self.nu[:, j] for j in range(self.nu.shape[1])]

    @property
    def nu_RS(self):                  # shape [R, S]
        return self.nu.t().contiguous()

    def stoich_by_reaction(self, j):  # returns [S]
        return self.nu[:, j]
