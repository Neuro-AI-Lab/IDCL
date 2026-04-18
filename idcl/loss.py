import torch
import torch.nn as nn
import torch.nn.functional as F


class IDCL(nn.Module):
    """
    Inter-Dialog Contrastive Learning (Lee, Kim, Choi, ICASSP 2026)

    - Eq. (4): N_i^m = topK over cosine similarity S(G_i^m, G_j^m), j != i
    - Eq. (5): L = -sum_i log( sum_{k in N_i^m} exp(S(G_i^{m_bar}, G_k^{m_bar})/tau)
                               / sum_{l=1..N_B}    exp(S(G_i^{m_bar}, G_l^{m_bar})/tau) )

    Args:
        K (int): Number of top-K neighbors used as positives. Default: 15
        temperature (float): Softmax temperature. Default: 0.1

    Example::
        >>> loss_fn = IDCL(K=15, temperature=0.05)
        >>> loss = loss_fn(audio_feat, text_feat)  # [B, L, D]
    """
    def __init__(self, K: int = 15, temperature: float = 0.1):
        super().__init__()
        self.K = K
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor   : modality m      (used to pick KNN)         [B, L, D]
            modality : modality m_bar  (where InfoNCE is applied)  [B, L, D]
        Returns:
            Scalar loss tensor.
        """
        batch_size = anchor.size(0)
        if batch_size < 2:
            return anchor.new_zeros(())

        # Dialog-level mean pooling + L2 normalization
        anchor_n = F.normalize(anchor.mean(dim=1),   p=2, dim=1)  # [B, D]
        mod_n    = F.normalize(modality.mean(dim=1), p=2, dim=1)  # [B, D]

        # Top-K neighbors via cosine similarity (Eq. 4)
        k = min(self.K, batch_size - 1)
        with torch.no_grad():
            sim_anchor = anchor_n @ anchor_n.T
            sim_anchor.fill_diagonal_(-float('inf'))
            knn_idx = sim_anchor.topk(k, dim=-1).indices           # [B, k]

        # Positive mask (Eq. 4)
        pos_mask = torch.zeros(batch_size, batch_size, device=anchor.device)
        pos_mask.scatter_(1, knn_idx, 1.0)

        # InfoNCE in the other modality (Eq. 5)
        sim_mod = (mod_n @ mod_n.T) / self.temperature
        return self._info_nce(sim_mod, pos_mask)

    def _info_nce(self, sim: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        B = sim.size(0)
        eye_mask = 1 - torch.eye(B, device=sim.device)
        sim_max  = sim.max(dim=1, keepdim=True)[0]
        exp_sim  = torch.exp(sim - sim_max) * eye_mask
        pos_sum  = (exp_sim * pos_mask).sum(dim=1, keepdim=True) + 1e-8
        all_sum  = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        return -torch.mean(torch.log(pos_sum) - torch.log(all_sum))
