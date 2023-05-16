import torch
import math
from nitorch_core.extra import make_vector
from .base import Mixture


class ChiMixture(Mixture):
    # Chi Mixture Model (CMM).
    def __init__(self, num_class=2, dof=None, sig=None,
                 update_dof=True, update_sig=True):
        """
        Parameters
        ----------
        dof : (K,) tensor, optional
            Degrees of freedom of each Chi distribution.
        sig : (K,) tensor, optional
            Standard deviation of each Chi distribution.
        """
        super().__init__(num_class=num_class)
        self.dof = dof
        self.sig = sig
        self.update_dof = update_dof
        self.update_sig = update_sig

    def get_means_variances(self):
        """ Return means and variances.

        Returns
        -------
        means : (1, K) tensor
        variances : (1, 1, K) tensor
        """
        K = self.K
        dtype = torch.float64

        # Compute means and variances
        mean = torch.zeros((1, K), dtype=dtype, device=self.dev)
        var = torch.zeros((1, 1, K), dtype=dtype, device=self.dev)
        for k in range(K):
            dof_k = self.dof[k]

            tmp = (torch.lgamma(0.5*(dof_k+1)) - torch.lgamma(0.5*dof_k)).exp()
            if torch.isfinite(tmp):
                mean[:, k] = self.sig[k] * math.sqrt(2) * tmp
            else:
                mean[:, k] = 0
            var[:, :, k] = self.sig[k].square() * dof_k - mean[:, k].square()
        return mean, var

    def _log_likelihood(self, X, k=0, c=None):
        """
        Log-probability density function (pdf) of the Chi
        distribution, evaluated at the values in X.

        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.

        Returns:
            log_pdf (torch.tensor): (N, 1).

        See also:
            https://en.wikipedia.org/wiki/Chi_distribution

        """

        N = X.shape[0]
        device = X.device
        dtype = X.dtype
        tiny = 1e-32
        log = lambda x: x.clamp_min(tiny).log_()

        # Get parameters
        dof = self.dof[k]
        dof = dof.type(dtype)
        dof = dof.to(device)
        sig2 = self.sig[k].square()
        sig2 = sig2.type(dtype)
        sig2 = sig2.to(device)

        # Use Chi distribution
        log_pdf = (1 - dof/2)*math.log(2) - 0.5*dof*log(sig2) \
                  - torch.lgamma(dof/2) + (dof - 1.)*log(X) \
                  - 0.5*X.square()/sig2.clamp_min(tiny)
        return log_pdf.flatten()

    def _init_par(self, X, W=None):
        """  Initialise CMM specific parameters: dof, sig

        """
        K = self.K
        dtype = torch.float64

        # Init mixing prop
        self._init_mp(dtype)

        if self.sig is None:
            if W is None:
                self.sig = torch.mean(X)*5
                self.sig = torch.sum((self.sig - X).square())/torch.numel(X)
                self.sig = torch.sqrt(self.sig/(K+1)*(torch.arange(1, K+1, dtype=dtype, device=self.dev)))
            else:
                self.sig = 5*(W*X).sum()/W.sum()
                self.sig = (W*(self.sig - X).square()).sum()/W.sum()
                self.sig = torch.sqrt(self.sig/(K+1)*(torch.arange(1, K+1, dtype=dtype, device=self.dev)))
        else:
            self.sig = make_vector(self.sig, K, dtype=dtype, device=self.dev)

        if self.dof is None:
            self.dof = torch.full([K], 3, dtype=dtype, device=self.dev)
        else:
            self.dof = make_vector(self.dof, K, dtype=dtype, device=self.dev)
        return

    def _suffstats(self, X, Z):
        """ Compute sufficient statistics.

        Args:
            X (torch.tensor): Observed data (N, 1).
            Z (torch.tensor): Responsibilities (N, K).

        Returns:
            ss0 (torch.tensor): sum(Z)          (K)
            ss1 (torch.tensor): sum(Z * log(X)) (K)
            ss2 (torch.tensor): sum(Z * X**2)   (K)

        """
        K = Z.shape[1]
        device = self.dev
        tiny = 1e-32
        X = X.flatten()
        logX = X.clamp_min(tiny).log_()

        # Suffstats
        ss0 = torch.zeros(K, dtype=torch.float64, device=device)
        ss1 = torch.zeros(K, dtype=torch.float64, device=device)
        ss2 = torch.zeros(K, dtype=torch.float64, device=device)

        # Compute sufficient statistics
        for k in range(K):
            Zk = Z[:, k]
            ss0[k] = Zk.sum(dtype=torch.float64)
            ss1[k] = (logX * Zk).sum(dim=0, dtype=torch.float64)
            ss2[k] = ((X*X) * Zk).sum(dim=0, dtype=torch.float64)
        return ss0, ss1, ss2

    def _update(self, ss0, ss1, ss2):
        """ Update CMM parameters.

        Args:
            ss0 (torch.tensor): sum(Z)          (K)
            ss1 (torch.tensor): sum(Z * log(X)) (K)
            ss2 (torch.tensor): sum(Z * X**2)   (K)

        """
        K = len(ss0)
        tiny = 1e-32

        def log(x):
            return (x+tiny).log_()

        # Update parameters (using means and variances)
        for k in range(K):

            max_iter = 10000 if self.update_dof else 1
            for i in range(max_iter):
                # print(self.sig[k].item(), self.dof[k].item())
                if self.update_sig:
                    # Closed form update of sig
                    self.sig[k] = (ss2[k]/(self.dof[k]*ss0[k])).sqrt()

                if self.update_dof:
                    # Gauss-Newton update of dof
                    gkl = torch.digamma(self.dof[k]/2) + log(2*self.sig[k].square())
                    gkl = 0.5 * ss0[k] * gkl - ss1[k]                        # gradient w.r.t. dof
                    hkl = 0.25 * ss0[k] * torch.polygamma(1, self.dof[k]/2)  # Hessian w.r.t. dof
                    self.dof[k].sub_(gkl/hkl)                                # G-N update
                    if self.dof[k] < 2:
                        self.dof[k] = 2
                        break
                    if gkl*gkl < 1e-9:
                        break


CMM = ChiMixture