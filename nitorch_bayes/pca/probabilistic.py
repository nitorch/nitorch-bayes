import torch
from nitorch_core import extra


def ppca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False, rca=None):
    """Probabilistic Principal Component Analysis

    !!! note
        - We manually orthogonalize the subspace within the optimization
          loop so that the output subspace is orthogonal (`z.T @ z` and
          `u @ u.T` are diagonal).
        - The output basis is not unitary. Each basis is scaled by
          the square root of the corresponding eigenvalue of the sample
          covariance minus the residual variance:
                basis = unitary_basis * sqrt(lambda - sigma ** 2)
          See reference [1].
        - Probabilistic residual component analysis (RCA) can be performed
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
    ..[1] "Probabilistic principal component analysis."
          Tipping, Michael E. and Bishop, Christopher M.
          J. R. Stat. Soc., Ser. B (1999)
    ..[2] "Residual Component Analysis: Generalising PCA for more
          flexible inference in linear-Gaussian models."
          Kalaitzis, Alfredo A. and Lawrence, Neil D.
          ICML (2012)

    """

    if isinstance(x, (list, tuple)) and callable(x[0]):
        return _ppca_callable(x, nb_components, mean, max_iter, tol,
                              returns, verbose, rca)
    else:
        return _ppca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose, rca)


def _ppca_tensor(x, k, mu, max_iter, tol, returns, verbose=False, rca=None):
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
    if not nomu:
        x = x - mu.unsqueeze(-2)

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
        return x.double().inverse().to(dtype=dtype)

    def rinv(z, s, side='l'):
        """Regularized pseudo-inverse
        z : (..., N, K) matrix to invert
        s : (...) weight
        side : {'l', 'r'}
        returns (..., K, N) -> (z.T @ z + s * I).inv() @ z.T, if 'l'
                (..., N, K) -> (z @ z.T + s * I).inv() @ z,   if 'r'
        """
        if side[0] == 'l':
            zz = make_sym(t(z).matmul(z))
            zz = inv(reg_(zz, s))
            z = zz.matmul(t(z))
        else:
            zz = make_sym(z.matmul(t(z)))
            zz = inv(reg_(zz, s))
            z = zz.matmul(z)
        return z

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

    def rescale(uu, s):
        """Rescale after orthonormalization to optimize log-evidence.
        uu : (*batch, K, K) - basis product (u @ u.T) !!must be diagonal!!
        s  : (*batch) - Residual variance
        """
        # The objective function that I optimize here is the one computed
        # in `logev`, so it takes into account an "immediate" update
        # of the posterior covariance. This means that the scaling is
        # only applied to the basis u (and to the mean of z), and the latent
        # covariance is immediately updated according to:
        #                   Sz = inv(inv_scale(uu) + s*I)
        # In the shape (& appearance) papers, we kept the posterior
        # covariance fixed (under VB) and scaled it along with the mean:
        #                   z = scale(z)
        #                   Sz = scale(Sz)
        a = s[..., None] / get_diag(uu)
        scl = (1 + (1 + 4 * a * n).sqrt()) / (2*n)
        return scl.reciprocal_().sqrt_()

    def logev(r, z, uu, s):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        z  : (*batch, N, K) - latent variables
        uu : (*batch, K, K) - basis product (u @ u.T)
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z inside the function (with
        # the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = z.square().sum([-1, -2])
        z = z.sum()
        # uncertainty
        unc = reg(uu, s).logdet() - s.log() * k
        unc = unc.sum() * n
        # log sigma
        s = s.log().sum() * (n * m)
        tot = (r + z + s + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{s.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
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

    # init basis
    iz = rinv(z, s, 'l')
    u = iz.matmul(x)
    uu = make_sym(u.matmul(t(rca(u))))

    # init log-evidence
    if has_rca:
        r = (x - z.matmul(u))
        r = (r * rca(r)).sum([-1, -2])
    else:
        r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, z, uu, s)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = x.matmul(t(rca(u))).matmul(im)              # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        tiny = eps * get_diag(zz).abs().max(-1).values
        sz = im * s[..., None, None].clamp_min(tiny)    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        u = inv(zz).matmul(t(z)).matmul(x)
        uu = make_sym(u.matmul(t(rca(u))))

        # update sigma
        sz = s * inv(reg(uu, s))
        if has_rca:
            r = (x - z.matmul(u))
            r = (r * rca(r)).sum([-1, -2])
        else:
            r = (x - z.matmul(u)).square_().sum([-1, -2])
        s = r / (n*m) + trace(sz.matmul(uu)) / m     # residuals + uncertainty

        # orthogonalize
        zz = make_sym(t(z).matmul(z))
        q, iq = joint_ortho(zz, uu)
        scl = rescale(iq.matmul(uu).matmul(t(iq)), s)
        q *= scl[..., None, :]
        iq /= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update log-evidence
        l = logev(r, z, uu, s)
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
    return out[0] if len(out) == 0 else tuple(out)


def _ppca_callable(x, k, mu, max_iter, tol, returns, verbose=False, rca=None):
    """Inline implementation that loads data only when needed."""

    # --- preproc ---

    x = list(x)
    n = len(x)

    if callable(mu):
        mu = mu()
    nomu = isinstance(mu, (int, float)) and mu == 0
    if mu is not None:
        mu = torch.as_tensor(mu)

    # infer output shape/dtype/device
    shape = [mu.shape] if mu is not None else []
    dtype = [mu.dtype] if mu is not None else []
    device = [mu.device] if mu is not None else []
    if mu is None:
        mu = 0
    for x1 in x:
        x1 = torch.as_tensor(x1())
        mu += x1
        shape.append(tuple(x1.shape))
        dtype.append(x1.dtype)
        device.append(x1.device)
    mu /= n
    shape = list(torch.broadcast_shapes(*shape))
    m = shape.pop(-1)
    dtype = extra.max_dtype(dtype)
    device = extra.max_device(device)
    backend = dict(dtype=dtype, device=device)
    mu = mu.to(**backend)

    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    k = k or (min(n, m) - 1)
    eps = 1e-6

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
        return x.double().inverse().to(dtype=dtype)

    def rinv(z, s, side='l'):
        """Regularized pseudo-inverse
        z : (..., N, K) matrix to invert
        s : (...) weight
        side : {'l', 'r'}
        returns (..., K, N) -> (z.T @ z + s * I).inv() @ z.T, if 'l'
                (..., N, K) -> (z @ z.T + s * I).inv() @ z,   if 'r'
        """
        if side[0] == 'l':
            zz = make_sym(t(z).matmul(z))
            zz = inv(reg_(zz, s))
            z = zz.matmul(t(z))
        else:
            zz = make_sym(z.matmul(t(z)))
            zz = inv(reg_(zz, s))
            z = zz.matmul(z)
        return z

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

    def rescale(uu, s):
        """Rescale after orthonormalization to optimize log-evidence.
        uu : (*batch, K, K) - basis product (u @ u.T) !!must be diagonal!!
        s  : (*batch) - Residual variance
        """
        # The objective function that I optimize here is the one computed
        # in `logev`, so it takes into account an "immediate" update
        # of the posterior covariance. This means that the scaling is
        # only applied to the basis u (and to the mean of z), and the latent
        # covariance is immediately updated according to:
        #                   Sz = inv(inv_scale(uu) + s*I)
        # In the shape (& appearance) papers, we kept the posterior
        # covariance fixed (under VB) and scaled it along with the mean:
        #                   z = scale(z)
        #                   Sz = scale(Sz)
        a = s[..., None] / get_diag(uu)
        scl = (1 + (1 + 4 * a * n).sqrt()) / (2*n)
        return scl.reciprocal_().sqrt_()

    def logev(r, z, uu, s):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        z  : (*batch, N, K) - latent variables
        uu : (*batch, K, K) - basis product (u @ u.T)
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z inside the function (with
        # the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = z.square().sum([-1, -2])
        z = z.sum()
        # uncertainty
        unc = reg(uu, s).logdet() - s.log() * k
        unc = unc.sum() * n
        # log sigma
        s = s.log().sum() * (n * m)
        tot = (r + z + s + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{s.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    def matmul(x, y, out=None):
        """Matmul where one of the inputs is a list of callable"""
        if isinstance(x, list):
            if out is None:
                out = y.new_empty(n, y.shape[-1])
            for i, x1 in enumerate(x):
                x1 = x1()
                if not nomu:
                    x1 -= mu
                out[..., i, :] = x1[..., None, :].matmul(y)[..., 0, :]
        elif isinstance(y, list):
            if out is None:
                out = x.new_empty(x.shape[-2], m)
            out.zero_()
            for i, y1 in enumerate(y):
                y1 = y1()
                if not nomu:
                    y1 -= mu
                out += x[..., i, None] * y1[..., None]
        return out

    def get_sqres(x, z, u, out=None):
        """Compute sum of squared residuals"""
        if out is None:
            out = z.new_empty(shape)
        out.zero_()
        for i, x1 in enumerate(x):
            recon = (z[..., n, :, None] * u).sum(-2)
            x1 = x1()
            if not nomu:
                x1 -= mu
            x1 -= recon
            if has_rca:
                out += (x1 * rca(x1)).sum(-1)
            else:

                out += x1.square_().sum(-1)
        return out

    def var(x):
        out = torch.zeros(shape, **backend)
        for x1 in x:
            x1 = x1()
            if not nomu:
                x1 -= mu
            if has_rca:
                out += (x1 * rca(x1)).sum(-1)
            else:
                out += x1.square_().sum(-1)
        out /= (n*m)
        return out

    # --- initialization ---

    # init residual var with 10% of full var
    s = var(x).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*shape, n, k], **backend)
    z, _, _ = torch.svd(z, some=True)

    # init basis
    iz = rinv(z, s, 'l')
    u = matmul(iz, x)
    uu = make_sym(u.matmul(t(rca(u))))

    # init log-evidence
    r = get_sqres(x, z, u)
    l0 = l1 = logev(r, z, uu, s)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = matmul(x, t(rca(u)), out=z).matmul(im)      # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        tiny = eps * get_diag(zz).abs().max(-1).values
        sz = im * s[..., None, None].clamp_min(tiny)    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        u = matmul(inv(zz).matmul(t(z)), x, out=u)
        uu = make_sym(u.matmul(t(rca(u))))

        # update sigma
        sz = s * inv(reg(uu, s))
        r = get_sqres(x, z, u, out=r)
        s = r / (n*m) + trace(sz.matmul(uu)) / m     # residuals + uncertainty

        # orthogonalize
        zz = make_sym(t(z).matmul(z))
        q, iq = joint_ortho(zz, uu)
        scl = rescale(iq.matmul(uu).matmul(t(iq)), s)
        q *= scl[..., None, :]
        iq /= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update log-evidence
        l = logev(r, z, uu, s)
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
    return out[0] if len(out) == 0 else tuple(out)
