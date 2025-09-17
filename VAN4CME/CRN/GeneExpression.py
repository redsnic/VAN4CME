import torch
import torch.nn.functional as F
import scipy.sparse as sp

def _onehot(idx, n):
    return F.one_hot(idx.clamp(0, n - 1).long(), n).float()

class GeneExpressionCRN:
    
    def __init__(self, k_r=2, k_p=2, g_r=2, g_p=2, P_MAX=10, M_MAX=10, DEVICE="cpu"):
        self.k_r, self.k_p, self.g_r, self.g_p = k_r, k_p, g_r, g_p
        self._A_cache = None
        self._dims_cache = None  # (M_MAX, P_MAX)
        self.P_MAX = P_MAX
        self.M_MAX = M_MAX
        self.N_STATES = (M_MAX + 1) * (P_MAX + 1)
        self.DEVICE = DEVICE
        self.species = ["M", "P"]  # Example species, can be extended
        
        # Marginal sizes for species
        # self.margnal_sizes = {"M": M_MAX, "P": P_MAX}
        self.species_marginals_sizes = {"M": M_MAX + 1, "P": P_MAX + 1}

    @torch.no_grad()
    def _sizes(self):
        return self.M_MAX, self.P_MAX, self.N_STATES, self.DEVICE

    def ap_from_net(self, net, t, vals, *, eps=0.0):
        """
        Compute (A p_theta)(m,p,t) using NN probabilities from net.prob_mp.

        Args:
            net  : model with prob_mp(t_vec, species_vec)
            t    : scalar or [K] tensor
            vals : [K, 2] integer tensor [m, p]
        """
        M_MAX, P_MAX, _, device = self._sizes()

        vals = vals.to(device)
        m, p = vals[:, 0].long(), vals[:, 1].long()

        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=device)
        t = t.to(device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand_as(m)  # broadcast scalar

        # neighbors & masks
        m_prev = (m - 1).clamp(0, M_MAX); valid_m_prev = (m > 0).float()
        m_next = (m + 1).clamp(0, M_MAX); valid_m_next = (m < M_MAX).float()
        p_prev = (p - 1).clamp(0, P_MAX); valid_p_prev = (p > 0).float()
        p_next = (p + 1).clamp(0, P_MAX); valid_p_next = (p < P_MAX).float()

        # prob_mp calls — species_vec is [K, 2]
        p_m_prev = net.prob_mp(t, torch.stack([m_prev, p], dim=1)) * valid_m_prev
        p_m_next = net.prob_mp(t, torch.stack([m_next, p], dim=1)) * valid_m_next
        p_p_prev = net.prob_mp(t, torch.stack([m, p_prev], dim=1)) * valid_p_prev
        p_p_next = net.prob_mp(t, torch.stack([m, p_next], dim=1)) * valid_p_next
        p_cur    = net.prob_mp(t, torch.stack([m, p], dim=1))

        # rates
        k_r = torch.as_tensor(self.k_r, device=device)
        k_p = torch.as_tensor(self.k_p, device=device)
        g_r = torch.as_tensor(self.g_r, device=device)
        g_p = torch.as_tensor(self.g_p, device=device)

        inflow = (
            k_r * p_m_prev
            + g_r * (m + 1).float() * p_m_next
            + k_p * m.float()       * p_p_prev
            + g_p * (p + 1).float() * p_p_next
        )
        outflow_rate = (
            k_r
            + g_r * m.float()
            + k_p * m.float()
            + g_p * p.float()
        )
        outflow = outflow_rate * p_cur

        return inflow - outflow

    
    def ic_loss(self, net, B=1024): # NLL
        t0 = torch.zeros(B, device=net.device)
        # all-zero state for each species
        ohs = []
        for sp in net.crn.species:
            C = net.crn.species_marginals_sizes[sp]
            ohs.append(F.one_hot(torch.zeros(B, dtype=torch.long, device=t0.device), C).float())
        p00 = net.get_probability(ohs, t0).clamp_min(1e-12)
        return -p00.log().mean()
    
    @staticmethod
    def _index_of(m, p, P_MAX):
        # row-major: i = m*(P_MAX+1) + p
        return m * (P_MAX + 1) + p

    def build_generator(self, M_MAX, P_MAX):
        """Build and cache sparse generator A for current rates and box size."""
        dims = (M_MAX, P_MAX)
        if self._A_cache is not None and self._dims_cache == dims:
            return self._A_cache

        rows, cols, data = [], [], []
        k_r, k_p, g_r, g_p = self.k_r, self.k_p, self.g_r, self.g_p

        for m in range(M_MAX + 1):
            for p in range(P_MAX + 1):
                i = self._index_of(m, p, P_MAX)

                # ---- inflow to (m,p) ----
                # (m-1,p) -> (m,p) at k_r
                if m > 0:
                    j = self._index_of(m - 1, p, P_MAX)
                    rows.append(i); cols.append(j); data.append(k_r)

                # (m,p-1) -> (m,p) at k_p * m
                if p > 0:
                    j = self._index_of(m, p - 1, P_MAX)
                    rows.append(i); cols.append(j); data.append(k_p * m)

                # (m+1,p) -> (m,p) at g_r * (m+1)
                if m < M_MAX:
                    j = self._index_of(m + 1, p, P_MAX)
                    rows.append(i); cols.append(j); data.append(g_r * (m + 1))

                # (m,p+1) -> (m,p) at g_p * (p+1)
                if p < P_MAX:
                    j = self._index_of(m, p + 1, P_MAX)
                    rows.append(i); cols.append(j); data.append(g_p * (p + 1))

                # ---- diagonal (outflow that stays inside box) ----
                out = 0.0
                if m < M_MAX: out += k_r          # (m,p)->(m+1,p)
                if m > 0:     out += g_r * m      # (m,p)->(m-1,p)
                if p < P_MAX: out += k_p * m      # (m,p)->(m,p+1)
                if p > 0:     out += g_p * p      # (m,p)->(m,p-1)

                rows.append(i); cols.append(i); data.append(-out)

        N = (M_MAX + 1) * (P_MAX + 1)
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=float)
        self._A_cache = A
        self._dims_cache = dims
        return A

    def apply_A(self, y, M_MAX, P_MAX):
        """Compute A @ y using cached/built generator."""
        A = self.build_generator(M_MAX, P_MAX)
        return A.dot(y)


import torch
import torch.nn.functional as F
import scipy.sparse as sp

class GeneExpressionCRNWithFeedback(GeneExpressionCRN):
    """
    Same two-species CRN (M, P), but transcription has negative feedback:
        m -> m+1 at rate  k_r / (1 + feedback * m)

    This modifies:
      - inflow to (m,p) from (m-1,p): use source count (m-1) in the rate
      - outflow from (m,p) to (m+1,p): use current count m in the rate
    """
    def __init__(self, k_r=2, k_p=2, g_r=2, g_p=2, feedback=1.0,
                 P_MAX=10, M_MAX=10, DEVICE="cpu"):
        super().__init__(k_r=k_r, k_p=k_p, g_r=g_r, g_p=g_p,
                         P_MAX=P_MAX, M_MAX=M_MAX, DEVICE=DEVICE)
        self.feedback = float(feedback)

    def ap_from_net(self, net, t, vals, *, eps=0.0):
        """
        Compute (A p_theta)(m,p,t) using NN probabilities with feedback:
          transcription rate = k_r / (1 + feedback * m)
        """
        M_MAX, P_MAX, _, device = self._sizes()

        vals = vals.to(device)
        m, p = vals[:, 0].long(), vals[:, 1].long()

        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=device)
        t = t.to(device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand_as(m)

        # neighbors & masks
        m_prev = (m - 1).clamp(0, M_MAX); valid_m_prev = (m > 0).float()
        m_next = (m + 1).clamp(0, M_MAX); valid_m_next = (m < M_MAX).float()
        p_prev = (p - 1).clamp(0, P_MAX); valid_p_prev = (p > 0).float()
        p_next = (p + 1).clamp(0, P_MAX); valid_p_next = (p < P_MAX).float()

        # probabilities via your model API
        p_m_prev = net.prob_mp(t, torch.stack([m_prev, p], dim=1)) * valid_m_prev
        p_m_next = net.prob_mp(t, torch.stack([m_next, p], dim=1)) * valid_m_next
        p_p_prev = net.prob_mp(t, torch.stack([m, p_prev], dim=1)) * valid_p_prev
        p_p_next = net.prob_mp(t, torch.stack([m, p_next], dim=1)) * valid_p_next
        p_cur    = net.prob_mp(t, torch.stack([m, p],      dim=1))

        # rates with feedback for transcription
        k_r = torch.as_tensor(self.k_r, device=device, dtype=torch.float32)
        k_p = torch.as_tensor(self.k_p, device=device, dtype=torch.float32)
        g_r = torch.as_tensor(self.g_r, device=device, dtype=torch.float32)
        g_p = torch.as_tensor(self.g_p, device=device, dtype=torch.float32)
        fb  = torch.as_tensor(self.feedback, device=device, dtype=torch.float32)

        # transcription inflow uses source (m-1); outflow uses current m
        tr_in  = k_r / (1.0 + fb * (m - 1).clamp_min(0).float())
        tr_out = k_r / (1.0 + fb * m.float())

        inflow = (
            tr_in * p_m_prev
            + g_r * (m + 1).float() * p_m_next
            + k_p * m.float()       * p_p_prev
            + g_p * (p + 1).float() * p_p_next
        )

        outflow_rate = ( # strict boundary condition
            tr_out
            + g_r * m.float() * valid_m_prev
            + k_p * m.float() * valid_p_next
            + g_p * p.float() * valid_p_prev
        )
        outflow = outflow_rate * p_cur

        return inflow - outflow

    # Optional: also provide a generator matrix for FSP that matches the feedback.
    def build_generator(self, M_MAX, P_MAX):
        dims = (M_MAX, P_MAX)
        if self._A_cache is not None and self._dims_cache == dims:
            return self._A_cache

        rows, cols, data = [], [], []
        k_r, k_p, g_r, g_p, fb = self.k_r, self.k_p, self.g_r, self.g_p, self.feedback

        for m in range(M_MAX + 1):
            for p in range(P_MAX + 1):
                i = self._index_of(m, p, P_MAX)

                # inflow:
                # (m-1,p)->(m,p) with rate k_r/(1+fb*(m-1))
                if m > 0:
                    j = self._index_of(m - 1, p, P_MAX)
                    rows.append(i); cols.append(j); data.append(k_r / (1.0 + fb * (m - 1)))

                # (m,p-1)->(m,p) with rate k_p * m
                if p > 0:
                    j = self._index_of(m, p - 1, P_MAX)
                    rows.append(i); cols.append(j); data.append(k_p * m)

                # (m+1,p)->(m,p) with rate g_r * (m+1)
                if m < M_MAX:
                    j = self._index_of(m + 1, p, P_MAX)
                    rows.append(i); cols.append(j); data.append(g_r * (m + 1))

                # (m,p+1)->(m,p) with rate g_p * (p+1)
                if p < P_MAX:
                    j = self._index_of(m, p + 1, P_MAX)
                    rows.append(i); cols.append(j); data.append(g_p * (p + 1))

                # diagonal: outflow that stays inside the box
                out = 0.0
                # (m,p)->(m+1,p) with feedback rate k_r/(1+fb*m) if m < M_MAX
                if m < M_MAX: out += k_r / (1.0 + fb * m)
                if m > 0:     out += g_r * m
                if p < P_MAX: out += k_p * m
                if p > 0:     out += g_p * p

                rows.append(i); cols.append(i); data.append(-out)

        N = (M_MAX + 1) * (P_MAX + 1)
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=float)
        self._A_cache = A
        self._dims_cache = dims
        return A

    # `apply_A` and `ic_loss` are inherited from the base class.
    # If you want the NLL delta IC like the base `GeneExpressionCRN.ic_loss`,
    # just don't override it. If you prefer MSE, you could override here.
