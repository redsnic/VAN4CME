# Mathematical Notes on CME-Consistent Autoregressive Model and Training Losses

## Notation
- Lattice state: $(m,p) \in \{0,\dots,M\} \times \{0,\dots,P\}$
- Time: $t \in [0, t_{\max}]$, remapped to $[-1,1]$ via $\tilde t = 2t/t_{\max} - 1$
- One-hot vectors: $e_m, e_p$; relaxed one-hots: $\hat e_m, \hat e_p$
- Joint model: $P_\theta(m,p|t) = P_\theta(m|t) P_\theta(p|m,t)$
- CME generator $\mathcal A$ so that $\partial_t P = (\mathcal A P)$

## Model
Logits: $\ell_m(t)$ and $\ell_p(m,t)$
\[
P_\theta(m|t) = \mathrm{softmax}(\ell_m(t))_m, \quad P_\theta(p|m,t) = \mathrm{softmax}(\ell_p(m,t))_p
\]
Straight-through Gumbel–Softmax sampling; soft $m$ encoding for conditioning $p$; evaluation always uses hard one-hots.

## CME Operator
With rates $(k_r,k_p,g_r,g_p)$:
\[
(\mathcal A P)(m,p) = k_r P(m-1,p) + g_r(m+1)P(m+1,p) + k_p m P(m,p-1) + g_p(p+1)P(m,p+1) \\
\quad - [g_r m + k_r + g_p p + k_p m] P(m,p)
\]
Boundaries are clamped.

## Residuals
### RL Residual
\[
\mathcal L_{RL} = E\left[ \left( \frac{\partial_t P_\theta - A^{det}}{\max(P_\theta,\varepsilon)} \right)^2 \right]
\]
- $A^{det}$ computed with `no_grad`
- Denominator detached

### Importance Residual
\[
\mathcal L_{IMP} = E_q\left[ \frac{\mathrm{Huber}_\delta(R_\theta)}{\max(P_\theta,\varepsilon)^2} \right]
\]
Full gradients; denominator detached; $q = (1-\alpha)U + \alpha P_\theta$

## REINFORCE Correction
Gradient:
\[
\nabla E[f] = E[\nabla f] + E[(f-b) \nabla \log P_\theta]
\]
- Baseline $b$: EMA of $f$
- $(f-b)$ detached
- Small scale factor in code

## IC Loss
\[
\mathcal L_{IC} = -\log P_\theta(m=0,p=0|t=0)
\]

## Entropy Loss
\[
\mathcal L_{ENT} = E[H(P_\theta(m|t)) + H(P_\theta(p|m,t))]
\]

## Total Loss
\[
\mathcal L = \lambda_{RL}(\mathcal L_{RL} + \mathcal L_{PG}) + \lambda_{IC} \mathcal L_{IC} + \lambda_{ENT} \mathcal L_{ENT} + \lambda_{IMP} \mathcal L_{IMP}
\]

## Detach Policy
- RL: detach $A^{det}$ and denominator
- PG: detach $(f-b)$
- IMP: detach denominator
- IC/ENT: full gradient

## Optional Temporal Consistency
\[
\mathcal L_{TC} = \|P(t+\Delta t) - [P(t) + \Delta t A^{det}]\|^2
\]

## Mass Diagnostic
Check $\sum_{m,p} P_\theta(m,p|t) \approx 1$ at key times.