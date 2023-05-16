import torch
from nitorch_core import extra


def vpca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False, rca=None):
    """Variational Principal Component Analysis

    !!! note
        - We manually orthogonalize the subspace within the optimization
          loop so that the output subspace is orthogonal (`E[z.T @ z]` and
          `E[u @ u.T]` are diagonal).
        - The output basis is not unitary.
        - Variational residual component analysis (RCA) can be performed
          instead of PCA by providing a function that applies the residual
          precision matrix. See reference [2].

    Parameters
    ----------
    x : (..., n, m) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `M` their dimension.
        If a sequence of callable, a memory efficient implementation is
        used, where tensors are loaded by calling the corresponding callable
        when needed.
    nb_components : int, default=min(n, m)-1
        Number of principal components (k) to return.
    mean : float or (..., m) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, use the mean of all observations (maximum-likelihood).
        If 0, nothing is done.
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-5
        Tolerance on log model evidence for early stopping.
    returns : {'latent', 'basis', 'var'}, default='latent+basis+var'
        Which variables to return.
    verbose : {0, 1, 2}, default=0
    rca : callable, optional
        A function (..., m) -> (..., m) that applies a residual precision
        matrix for residual component analysis.

    Returns
    -------
    latent : (..., n, k), if 'latent' in `returns`
        Latent coordinates
    basis : (..., k, m), if 'basis' in `returns`
        Orthogonal basis, scaled by sqrt(eigenvalue - residual variance)
    var : (...), if 'var' in `returns`
        Residual variance

    References
    ----------
    ..[1] "Variational principal components."
          Bishop, Christopher M.
          ICANN (1999)
    ..[2] "Residual Component Analysis: Generalising PCA for more
          flexible inference in linear-Gaussian models."
          Kalaitzis, Alfredo A. and Lawrence, Neil D.
          ICML (2012)

    """
    if isinstance(x, (list, tuple)) and callable(x[0]):
        raise NotImplementedError
        # TODO
        # return _vpca_callable(x, nb_components, mean, max_iter, tol,
        #                       returns, verbose, rca)
    else:
        return _vpca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose, rca)


def _vpca_tensor(x, k, mu, max_iter, tol, returns, verbose=False, rca=None):
    """Implementation that assumes that all the data is in memory."""

    # --- preproc ---

    x = torch.as_tensor(x)
    n, m = x.shape[-2:]
    backend = extra.backend(x)
    k = k or (min(x.shape[-1], x.shape[-2]) - 1)
    eps = 1e-6
    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    # subtract mean
    nomu = isinstance(mu, (float, int)) and mu == 0
    if mu is None:
        mu = x.mean(-2)
    mu = torch.as_tensor(mu, **backend)
    mu = mu.unsqueeze(-2)
    if not nomu:
        x = x - mu

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_diag(x):
        """Quick create diagonal matrix"""
        return torch.diagonal(x, dim1=-2, dim2=-1)

    def trace(x, **kwargs):
        """Batched trace"""
        return get_diag(x).sum(-1, **kwargs)

    def make_sym(x):
        """Make a matrux symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def reg(x, s):
        """Regularize matrix by adding number on the diagonal"""
        return reg_(x.clone(), s)

    def reg_(x, s):
        """Regularize matrix by adding number on the diagonal (inplace)"""
        get_diag(x).add_(s[..., None])
        return x

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        x = x.double()
        scl = get_diag(x).abs().max(-1)[0].mul_(1e-5)
        get_diag(x).add(scl[..., None])
        return x.inverse().to(dtype=dtype)

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    def logev(r, zz, uu, s, a, uuzz):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        zz : (*batch, K, K) - latent product (E[z @ z.T])
        uu : (*batch, K, K) - basis product (E[u @ u.T])
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z and u inside the function
        # (with the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = trace(zz).sum()                         # this should be n*k if optimal
        u = trace(uu.matmul(a)).sum()               # this should be m*k if optimal
        # uncertainty
        unc = ((trace(uu.matmul(zz)) - uuzz)/s).sum()   # uncertainty in likelihood
        az = uu/s[..., None, None]
        get_diag(az).add_(1)
        unc += az.logdet().sum() * n                # -E[log q(z)] in KL
        au = zz/s[..., None, None]
        au += a
        unc += au.logdet().sum() * m                # -E[log q(u)] in KL
        # log sigma
        s = s.log().sum() * (n * m)
        # log prior
        a = -a.logdet().sum() * m
        tot = (r + z + u + s + a + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{u.item() / (n*m):6f} | {s.item() / (n*m):6f} | '
        #       f'{a.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    # --- initialization ---

    # init residual var with 10% of full var
    if has_rca:
        s = (x * rca(x)).mean([-1, -2]).mul_(0.1)
    else:
        s = x.square().mean([-1, -2]).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*x.shape[:-1], k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))
    zz += torch.eye(k, **backend) * n

    # init basis
    im = inv(reg(zz, s))
    u = im.matmul(t(z)).matmul(x)
    uu = make_sym(u.matmul(t(rca(u))))
    uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
    su = im * s[..., None, None]
    uu += m * su
    a = inv(uu/m)

    # init log-evidence
    if has_rca:
        r = (x - z.matmul(u))
        r = (r * rca(r)).sum([-1, -2])
    else:
        r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, zz, uu, s, a, uuzz)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = x.matmul(t(rca(u))).matmul(im)              # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        sz = im * s[..., None, None]                    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        im = inv(zz + a*s[..., None, None])
        u = im.matmul(t(z)).matmul(x)
        uu = make_sym(u.matmul(t(rca(u))))
        uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
        su = im * s[..., None, None]
        uu += m * su

        # update sigma
        if has_rca:
            r = (x - z.matmul(u))
            r = (r * rca(r)).sum([-1, -2])
        else:
            r = (x - z.matmul(u)).square_().sum([-1, -2])
        s = r + trace(uu.matmul(zz)) - uuzz
        s /= (n*m)

        # orthogonalize (jointly)
        # For rescaling, writing out the terms that depend on it and
        # assuming E[zz], E[uu], Sz, Su diagonal (which they are after
        # joint diagonalization) and that A is immediately ML-updated shows
        # that the optimal scaling makes E[zz] an identity matrix.
        q, iq = joint_ortho(zz, uu)
        zz = t(q).matmul(zz).matmul(q)
        scl = get_diag(zz).div(n).sqrt_()
        zz = torch.eye(k, **backend).mul_(n)
        q /= scl[..., None, :]
        iq *= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update A
        a = inv(uu / m)

        # update log-evidence
        l = logev(r, zz, uu, s, a, uuzz)
        gain = (l1-l)/(l0 - l)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end=end)
        if abs(gain) < tol:
            break
        l1 = l
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'var':
            out.append(s)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)


def _vmpca_tensor(x, k, l, mu=None, max_iter=100, tol=1e-5,
                  returns='latent+basis', verbose=False, rca=None):
    """ Variational mixture of PCA
    Implementation that assumes that all the data is in memory.
    """

    # --- preproc ---

    x = torch.as_tensor(x)
    n, m = x.shape[-2:]
    backend = extra.backend(x)
    k = k or (min(x.shape[-1], x.shape[-2]) - 1)
    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    # subtract mean
    nomu = isinstance(mu, (float, int)) and mu == 0
    if mu is None:
        mu = x.mean(-2)
    mu = torch.as_tensor(mu, **backend)
    mu = mu.unsqueeze(-2)
    if not nomu:
        x = x - mu

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_diag(x):
        """Quick create diagonal matrix"""
        return torch.diagonal(x, dim1=-2, dim2=-1)

    def trace(x, **kwargs):
        """Batched trace"""
        return get_diag(x).sum(-1, **kwargs)

    def make_sym(x):
        """Make a matrux symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def reg(x, s):
        """Regularize matrix by adding number on the diagonal"""
        return reg_(x.clone(), s)

    def reg_(x, s):
        """Regularize matrix by adding number on the diagonal (inplace)"""
        get_diag(x).add_(s[..., None])
        return x

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        x = x.double()
        scl = get_diag(x).abs().max(-1)[0].mul_(1e-5)
        get_diag(x).add(scl[..., None])
        return x.inverse().to(dtype=dtype)

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    def logev(r, zz, uu, s, a, uuzz):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        zz : (*batch, K, K) - latent product (E[z @ z.T])
        uu : (*batch, K, K) - basis product (E[u @ u.T])
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z and u inside the function
        # (with the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = trace(zz).sum()                         # this should be n*k if optimal
        u = trace(uu.matmul(a)).sum()               # this should be m*k if optimal
        # uncertainty
        unc = ((trace(uu.matmul(zz)) - uuzz)/s).sum()   # uncertainty in likelihood
        az = uu/s[..., None, None]
        get_diag(az).add_(1)
        unc += az.logdet().sum() * n                # -E[log q(z)] in KL
        au = zz/s[..., None, None]
        au += a
        unc += au.logdet().sum() * m                # -E[log q(u)] in KL
        # log sigma
        s = s.log().sum() * (n * m)
        # log prior
        a = -a.logdet().sum() * m
        tot = (r + z + u + s + a + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{u.item() / (n*m):6f} | {s.item() / (n*m):6f} | '
        #       f'{a.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    # --- initialization ---

    # init residual var with 10% of full var
    if has_rca:
        s = (x * rca(x)).mean([-1, -2]).mul_(0.1)
    else:
        s = x.square().mean([-1, -2]).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*x.shape[:-1], k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))
    zz += torch.eye(k, **backend) * n

    # init basis
    im = inv(reg(zz, s))
    u = im.matmul(t(z)).matmul(x)
    uu = make_sym(u.matmul(t(rca(u))))
    uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
    su = im * s[..., None, None]
    uu += m * su
    a = inv(uu/m)

    # init log-evidence
    if has_rca:
        r = (x - z.matmul(u))
        r = (r * rca(r)).sum([-1, -2])
    else:
        r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, zz, uu, s, a, uuzz)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = x.matmul(t(rca(u))).matmul(im)              # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        sz = im * s[..., None, None]                    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        im = inv(zz + a*s[..., None, None])
        u = im.matmul(t(z)).matmul(x)
        uu = make_sym(u.matmul(t(rca(u))))
        uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
        su = im * s[..., None, None]
        uu += m * su

        # update sigma
        if has_rca:
            r = (x - z.matmul(u))
            r = (r * rca(r)).sum([-1, -2])
        else:
            r = (x - z.matmul(u)).square_().sum([-1, -2])
        s = r + trace(uu.matmul(zz)) - uuzz
        s /= (n*m)

        # orthogonalize (jointly)
        # For rescaling, writing out the terms that depend on it and
        # assuming E[zz], E[uu], Sz, Su diagonal (which they are after
        # joint diagonalization) and that A is immediately ML-updated shows
        # that the optimal scaling makes E[zz] an identity matrix.
        q, iq = joint_ortho(zz, uu)
        zz = t(q).matmul(zz).matmul(q)
        scl = get_diag(zz).div(n).sqrt_()
        zz = torch.eye(k, **backend).mul_(n)
        q /= scl[..., None, :]
        iq *= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update A
        a = inv(uu / m)

        # update log-evidence
        l = logev(r, zz, uu, s, a, uuzz)
        gain = (l1-l)/(l0 - l)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end=end)
        if abs(gain) < tol:
            break
        l1 = l
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'var':
            out.append(s)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)
