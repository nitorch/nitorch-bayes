import torch
import math
from nitorch_fastmath import besseli
from .base import Mixture


class RiceMixture(Mixture):
    # Univariate Rician Mixture Model (RMM).
    def __init__(self, num_class=2, nu=None, sig=None):
        """
        nu (torch.tensor): "mean" parameter of each Rician (K).
        sig (torch.tensor): "standard deviation" parameter of each Rician (K).
        """
        super().__init__(num_class=num_class)
        self.nu = nu
        self.sig = sig

    def get_means_variances(self):
        """ Return means and variances.
        Returns:
            (torch.tensor): Means (1, K).
            (torch.tensor): Variances (1, 1, K).
        """
        K = self.K
        dtype = torch.float64
        pi = torch.tensor(math.pi, dtype=dtype, device=self.dev)

        # Laguerre polymonial for n=1/2
        Laguerre = lambda x: torch.exp(x/2) * \
            ((1 - x) * besseli(0, -x/2) - x * besseli(1, -x/2))

        # Compute means and variances
        mean = torch.zeros((1, K), dtype=dtype, device=self.dev)
        var = torch.zeros((1, 1, K), dtype=dtype, device=self.dev)
        for k in range(K):
            nu_k = self.nu[k]
            sig_k = self.sig[k]

            x = -nu_k**2/(2*sig_k**2)
            x = x.flatten()
            if x > -20:
                mean[:, k] = torch.sqrt(pi * sig_k**2/2)*Laguerre(x)
                var[:, :, k] = 2*sig_k**2 + nu_k**2 - (pi*sig_k**2/2)*Laguerre(x)**2
            else:
                mean[:, k] = nu_k
                var[:, :, k] = sig_k

        return mean, var

    def _log_likelihood(self, X, k=0, c=None):
        """
        Log-probability density function (pdf) of the Rician
        distribution, evaluated at the values in X.
        Args:
            X (torch.tensor): Observed data (N, C).
            k (int, optional): Index of mixture component. Defaults to 0.
        Returns:
            log_pdf (torch.tensor): (N, 1).
        See also:
            https://en.wikipedia.org/wiki/Rice_distribution#Characterization
        """
        backend = dict(dtype=X.dtype, device=X.device)
        tiny = 1e-32

        # Get Rice parameters
        nu = self.nu[k].to(**backend)
        sig2 = self.sig[k].to(**backend).square()

        log_pdf = (X + tiny).log() - sig2.log() - (X.square() + nu.square()) / (2 * sig2)
        log_pdf = log_pdf + besseli(0, X * (nu / sig2), 'log')
        return log_pdf.flatten()

    def _init_par(self, X, W=None):
        """  Initialise RMM specific parameters: nu, sig
        """
        K = self.K
        mn = torch.min(X, dim=0)[0]
        mx = torch.max(X, dim=0)[0]
        dtype = torch.float64

        # Init mixing prop
        self._init_mp(dtype)

        # RMM specific
        if self.nu is None:
            self.nu = (torch.arange(K, dtype=dtype, device=self.dev)*mx)/(K + 1)
        if self.sig is None:
            self.sig = torch.ones(K, dtype=dtype, device=self.dev)*((mx - mn)/(K*10))

    def _update(self, ss0, ss1, ss2):
        """ Update RMM parameters.
        Args:
            ss0 (torch.tensor): 0th moment (K).
            ss1 (torch.tensor): 1st moment (C, K).
            ss2 (torch.tensor): 2nd moment (C, C, K).
        See also
            Koay, C.G. and Basser, P. J., Analytically exact correction scheme
            for signal extraction from noisy magnitude MR signals,
            Journal of Magnetic Resonance, Volume 179, Issue = 2, p. 317–322, (2006)
        """
        K = ss1.shape[1]
        dtype = torch.float64

        # Compute means and variances
        mu1 = torch.zeros(K, dtype=dtype, device=self.dev)
        mu2 = torch.zeros(K, dtype=dtype, device=self.dev)
        for k in range(K):
            # Update mean
            mu1[k] = 1/ss0[k] * ss1[:, k]

            # Update covariance
            mu2[k] = (ss2[:, :, k] - ss1[:, k]*ss1[:, k]/ss0[k] + self.lam*1e-3)/(ss0[k] + 1e-3)

        # Update parameters (using means and variances)
        for k in range(K):
            r = mu1[k] / mu2[k].sqrt()
            theta = math.sqrt(math.pi/(4 - math.pi))
            theta = torch.as_tensor(theta, dtype=dtype, device=self.dev).flatten()
            if r > theta:
                theta2 = theta * theta
                for i in range(256):
                    xi = besseli(0, theta2/4) * (2 + theta2) + \
                         besseli(1, theta2/4) * theta2
                    xi = xi.square() * (math.pi/8*math.exp(-theta2/2))
                    xi = (2 + theta2) - xi
                    g = (xi*(1 + r**2) - 2).sqrt()
                    if torch.abs(theta - g) < 1e-6:
                        break
                    theta = g
                if not torch.isfinite(xi):
                    xi.fill_(1)
                self.sig[k] = mu2[k].sqrt() / xi.sqrt()
                self.nu[k] = (mu1[k].square() + mu2[k]*(xi - 2)/xi).sqrt()
            else:
                self.nu[k] = 0
                self.sig[k] = 0.5*math.sqrt(2)*(mu1[k].square() + mu2[k]).sqrt()


RMM = RiceMixture