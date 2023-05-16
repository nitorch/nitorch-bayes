import math
import torch
from .base import Mixture


class GaussianMixture(Mixture):
    # Multivariate Gaussian Mixture Model (GMM).
    def __init__(self, num_class=2, mu=None, Cov=None):
        """
        mu (torch.tensor): GMM means (C, K).
        Cov (torch.tensor): GMM covariances (C, C, K).
        """
        super().__init__(num_class=num_class)
        self.mu = mu
        self.Cov = Cov

    def get_means_variances(self):
        """
        Return means and variances.
        Returns:
            (torch.tensor): Means (C, K).
            (torch.tensor): Covariances (C, C, K).
        """
        return self.mu, self.Cov

    def _log_likelihood(self, X, k=0, c=None):
        """ Log-probability density function (pdf) of the standard normal
            distribution, evaluated at the values in X.
        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
            c (int, optional): Index of channel. Defaults to None.
        Returns:
            log_pdf (torch.tensor): (N, 1).
        """
        C = X.shape[1]
        device = X.device
        dtype = X.dtype
        pi = torch.tensor(math.pi, dtype=dtype, device=device)
        if c is not None:
            Cov = self.Cov[c, c, k].reshape(1, 1).cpu()
            mu = self.mu[c, k].reshape(1).cpu()
        else:
            Cov = self.Cov[:, :, k]
            mu = self.mu[:, k]
        if C == 1:
            chol_Cov = torch.sqrt(Cov)
            log_det_Cov = torch.log(chol_Cov[0, 0])
        else:
            chol_Cov = torch.cholesky(Cov)
            log_det_Cov = torch.sum(torch.log(torch.diag(chol_Cov)))
            chol_Cov = chol_Cov.inverse()
        chol_Cov = chol_Cov.type(dtype)
        mu = mu.type(dtype)
        if C == 1:
            diff = (X - mu) / chol_Cov
        else:
            diff = torch.tensordot(X - mu, chol_Cov, dims=([1], [0]))
        log_pdf = - (C / 2) * torch.log(
            2 * pi) - log_det_Cov - 0.5 * torch.sum(diff ** 2, dim=1)
        return log_pdf

    def _init_par(self, X, W=None):
        """ Initialise GMM specific parameters: mu, Cov
        """
        dtype = torch.float64
        K = self.K
        C = X.shape[1]
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]

        # Init mixing prop
        self._init_mp(dtype)

        if self.mu is None:
            # means
            self.mu = torch.zeros((C, K), dtype=dtype, device=self.dev)
        if self.Cov is None:
            # covariance
            self.Cov = torch.zeros((C, C, K), dtype=dtype, device=self.dev)
            for c in range(C):
                # rng = torch.linspace(start=mn[c], end=mx[c], steps=K, dtype=dtype, device=self.dev)
                # num_neg = sum(rng < 0)
                # num_pos = sum(rng > 0)
                # rng = torch.arange(-num_neg, num_pos, dtype=dtype, device=self.dev)
                # self.mu[c, :] = torch.reshape((rng * (mx[c] - mn[c]))/(K + 1), (1, K))
                self.mu[c, :] = torch.reshape(
                    torch.linspace(mn[c], mx[c], K, dtype=dtype,
                                   device=self.dev), (1, K))
                self.Cov[c, c, :] = \
                    torch.reshape(torch.ones(K, dtype=dtype, device=self.dev)
                                  * ((mx[c] - mn[c]) / (K)) ** 2, (1, 1, K))

    def _update(self, ss0, ss1, ss2):
        """ Update GMM means and variances
        Args:
            ss0 (torch.tensor): 0th moment (K).
            ss1 (torch.tensor): 1st moment (C, K).
            ss2 (torch.tensor): 2nd moment (C, C, K).
        """
        C = ss1.shape[0]
        K = ss1.shape[1]

        # Update means and covariances
        for k in range(K):
            # Update mean
            self.mu[:, k] = 1 / ss0[k] * ss1[:, k]

            # Update covariance
            self.Cov[:, :, k] = ss2[:, :, k] / ss0[k] \
                                - torch.ger(self.mu[:, k], self.mu[:, k])


GMM = GaussianMixture