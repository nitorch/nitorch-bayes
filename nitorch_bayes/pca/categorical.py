import torch
from nitorch_core import extra
from nitorch_fastmath import logit, kron2


def _softmax_lse(x, dim=-1):
    """Implicit softmax that also returns the LSE"""
    x = x.clone()
    lse, _ = torch.max(x, dim=dim, keepdim=True)
    lse.clamp_min_(0)  # don't forget the class full of zeros

    x = x.sub_(lse).exp_()
    sumval = x.sum(dim=dim, keepdim=True)
    sumval += lse.neg().exp_()  # don't forget the class full of zeros
    x /= sumval

    sumval = sumval.log_()
    lse += sumval
    lse = lse.sum(dtype=torch.float64)
    return x, lse


def cpca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False):
    """(Probabilistic) Categorical Principal Component Analysis

    Notes
    -----
    .. We find the basis U that maximizes E_z[ Cat(x | SoftMax(U@z + mu)) ],
       where z stems form a standard Gaussian distribution.
    .. We manually orthogonalize the subspace within the optimization
       loop so that the output subspace is orthogonal (`z.T @ z` and
       `U @ U.T` are diagonal).

    Parameters
    ----------
    x : (..., n, m, i) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `MxI` their dimension,
        where `M` is the number of classes minus one, and `I` is the number
        of voxels.
    nb_components : int, default=min(n, m*i)-1
        Number of principal components (k) to return.
    mean : float or (..., m, i) tensor_like, optional
        Mean tensor. If not provided, it is estimated along with the bases.
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-5
        Tolerance on log model evidence for early stopping.
    returns : {'latent', 'basis', 'mean'}, default='latent+basis+mean'
        Which variables to return.
    verbose : {0, 1, 2}, default=0

    Returns
    -------
    latent : (..., n, k) tensor, if 'latent' in `returns`
        Latent coordinates
    basis : (..., k, m, i) tensor, if 'basis' in `returns`
        Orthogonal basis.
    mean : (..., m, i) tensor, if 'mean' in `returns`
        Mean

    References
    ----------
    ..[1] "Variational bounds for mixed-data factor analysis."
          Khan, Bouchard, Murphy, Marlin
          NeurIPS (2010)
    ..[2] "Factorisation-based Image Labelling."
          Yan, Balbastre, Brudfors, Ashburner.
          Preprint (2021)

    """

    if isinstance(x, (list, tuple)) and callable(x[0]):
        raise NotImplementedError
        # return _cpca_callable(x, nb_components, mean, max_iter, tol,
        #                       returns, verbose)
    else:
        return _cpca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose)


def _cpca_tensor(x, k, mu=None, max_iter=20, tol=1e-5,
                 returns='latent+basis+mean', verbose=False):

    # --- preproc ---
    x = torch.as_tensor(x)
    batch = x.shape[:-3]
    n, m, i = x.shape[-3:]
    backend = extra.backend(x)
    k = k or (min(n, m*i) - 1)

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_sym(x):
        """Make a matrix symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        return x.double().inverse().to(dtype=dtype)

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

    # --- initialization ---
    sumxn = x.sum(-3)

    # init mean
    optim_mu = mu is None
    if mu is None:
        mu = x.mean(-3)
        mu = logit(mu, -2, implicit=True)
    mu = torch.as_tensor(mu, **backend)

    # init latent with random orthogonal tensor
    z = torch.randn([*batch, n, k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))

    # init basis
    u = torch.zeros([*batch, k, m, i], **backend)

    # init approximate Hessian
    a = 0.5*(torch.eye(m, **backend) - 1/(m+1))
    b = torch.eye(m, **backend) + 1/(m+1)
    b = kron2(torch.eye(k, **backend), b)

    # init log-fit
    eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
    rho, nll = _softmax_lse(eta, -2)
    nll = nll.sum() - eta.flatten().dot(x.flatten())

    # init log-evidence
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {nll.item():6f}', end=end)

    for n_iter in range(max_iter):
        nll0 = nll

        # update mean (FIL, eq [30] simplified)
        if optim_mu and n_iter > 0:
            mu -= torch.einsum('...lm,mi->li', inv(a)/n, rho.sum(-3) - sumxn)

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update basis (FIL, eq [33, 34])
        h = kron2(zz, a) + b
        u = torch.einsum('...nk,kmi->nmi', z, u)
        u = torch.einsum('...ml,nli->nmi', a, u)
        u += x
        u -= rho
        u = torch.einsum('...nk,nmi->kmi', z, u)
        u = torch.einsum('...kl,li->ki', inv(h), u.reshape([*batch, k*m, i]))
        u = u.reshape([*batch, k, m, i])

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update latent (FIL, eq [25, 26])
        uu = torch.einsum('...kmi,ml,jli->kj', u, a, u)
        get_diag(uu).add_(1)
        uu = inv(uu)
        z = torch.einsum('...nk,kmi->nmi', z, u)
        z = torch.einsum('...ml,nli->nmi', a, z)
        z += x
        z -= rho
        z = torch.einsum('...kmi,nmi->nk', u, z)
        z = torch.einsum('...kj,nj->nk', uu, z)
        zz = make_sym(t(z).matmul(z))
        zz += uu

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # orthogonalize
        q, iq = joint_ortho(zz, uu)
        z = z.matmul(q)
        zz = t(q).matmul(zz).matmul(q)
        u = torch.einsum('...kl,lmi->kmi', iq, u)

        # update log-evidence
        nll = nll.sum() - eta.flatten().dot(x.flatten())
        gain = (nll0 - nll) / x.numel()
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {nll.item():6f} | '
                  f'{gain.item():.3e} ({"-" if nll < nll0 else "+"})', end=end)
        if abs(gain) < tol:
            break
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)


def _mcpca_tensor(x, k, q, mu=None, max_iter=20, tol=1e-5,
                  returns='latent+basis+mean+mixing', verbose=False):
    # x : (..., n, m, i)
    # n : number of independent observations
    # m : number of classes minus one
    # i : number of voxels per patch
    # k : number of latent dimensions
    # q : number of mixture components

    # --- preproc ---
    x = torch.as_tensor(x)
    batch = x.shape[:-3]
    n, m, i = x.shape[-3:]
    backend = extra.backend(x)
    k = k or (min(n, m*i) - 1)
    q = q or 1

    # --- helpers ---

    def flatdot(x, y):
        return x.flatten().dot(y.flatten())

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_sym(x):
        """Make a matrix symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        return x.double().inverse().to(dtype=dtype)

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

    # --- initialization ---
    sumxn = x.sum(-3)

    # init responsibilities
    pi = torch.rand([*batch, q], **backend)
    pi /= pi.sum(-1, keepdim=True)
    r = torch.distributions.Categorical(probs=pi).sample(n)
    r = r.transpose(-1, -2)  # (*batch, n, q)
    pi = r.sum(-2)
    pi /= pi.sum(-1, keepdim=True)
    klr = n * flatdot(pi, pi.log()) - flatdot(r, r.log())

    # init mean
    optim_mu = mu is None
    if mu is None:
        mu = torch.einsum('...nmi,nq->qmi', x, r)
        mu /= pi[..., None, None]
        mu = logit(mu, -2, implicit=True)
    mu = torch.as_tensor(mu, **backend)

    # init latent with random orthogonal tensor
    z = torch.randn([*batch, n, k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))

    # init basis
    u = torch.zeros([*batch, q, k, m, i], **backend)

    # init approximate Hessian
    a = 0.5*(torch.eye(m, **backend) - 1/(m+1))
    b = torch.eye(m, **backend) + 1/(m+1)
    b = kron2(torch.eye(k, **backend), b)

    # init log-fit
    eta = torch.einsum('...nk,qkmi->nqmi', z, u)
    eta = torch.einsum('...nqmi,qmi->nqmi', eta, mu)
    rho, nll = _softmax_lse(eta, -2)
    nll = flatdot(r, nll.sum(-1))
    nll -= torch.einsum('...nq,nqmi,nmi->', r, eta, x).sum()
    nll += klr

    # init log-evidence
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {nll.item():6f}', end=end)

    for n_iter in range(max_iter):
        nll0 = nll

        # update mean (FIL, eq [30])
        if optim_mu and n_iter > 0:
            mu = torch.einsum('...ml,qli->qmi', a, mu)
            mu.add_(sumxn.unsqueeze(-3), alpha=1/n)
            mu -= torch.einsum('...nqmi,nq,q->qmi', rho, r, 1/(n*pi))
            mu = torch.einsum('...lm,mi->li', inv(a), mu)

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update basis (FIL, eq [33, 34])
        h = kron2(zz, a) + b
        u = torch.einsum('...nk,kmi->nmi', z, u)
        u = torch.einsum('...ml,nli->nmi', a, u)
        u += x
        u -= rho
        u = torch.einsum('...nk,nmi->kmi', z, u)
        u = torch.einsum('...kl,li->ki', inv(h), u.reshape([*batch, k*m, i]))
        u = u.reshape([*batch, k, m, i])

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update latent (FIL, eq [25, 26])
        uu = torch.einsum('...kmi,ml,jli->kj', u, a, u)
        get_diag(uu).add_(1)
        uu = inv(uu)
        z = torch.einsum('...nk,kmi->nmi', z, u)
        z = torch.einsum('...ml,nli->nmi', a, z)
        z += x
        z -= rho
        z = torch.einsum('...kmi,nmi->nk', u, z)
        z = torch.einsum('...kj,nj->nk', uu, z)
        zz = make_sym(t(z).matmul(z))
        zz += uu

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # orthogonalize
        q, iq = joint_ortho(zz, uu)
        z = z.matmul(q)
        zz = t(q).matmul(zz).matmul(q)
        u = torch.einsum('...kl,lmi->kmi', iq, u)

        # update log-evidence
        nll = nll.sum() - eta.flatten().dot(x.flatten())
        gain = (nll0 - nll) / x.numel()
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {nll.item():6f} | '
                  f'{gain.item():.3e} ({"-" if nll < nll0 else "+"})', end=end)
        if abs(gain) < tol:
            break
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)
