"""Various flavors of component/factor analysis"""
import torch
from nitorch_core import extra



def pca(x, nb_components=None, mean=None,
        returns='latent+basis+scale', norm='latent+basis'):
    """Principal Component Analysis

    Factorize a NxM matrix X into the product ZSU where Z is NxK and
    unitary, U is KxM and unitary, and S is KxK and diagonal.

    By convention, N encodes independent replicates (individuals or samples)
    and M encodes correlated features, although in practice the problem
    is symmetric. Following probabilistic conventions, we say that
    each sample (X[n]) is encoded by "latent" coordinates (Z[n]) in an
    orthogonal "basis" (U).

    This function merely applies a singular value decomposition (SVD)
    under the hood.

    Parameters
    ----------
    x : (..., n, m) tensor_like or sequence[callable]
        Observed variables.
        `n` is the number of independent variables and `m` their dimension.
        If a sequence of callable, a memory efficient (but slower)
        implementation is used, where tensors are loaded by calling the
        corresponding callable when needed.
    nb_components : int, default=`min(n, m)`
        Number of principal components (k) to return.
    mean : float or (..., m) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, subtract the mean of all observations.
        If 0, nothing is done.
    returns : combination of {'latent', 'basis', 'scale'}, default='latent+basis+scale'
        Which variables to return.
    norm : {'latent', 'basis', 'latent+basis', None}, default='latent+basis'
        Which variable to normalize. If normalized, the corresponding
        matrix is unitary (Z @ Z.T == I).
        If 'latent+basis', a tensor of scales is returned.

    Returns
    -------
    latent : (..., n, k), if 'latent' in `returns`
    basis : (..., k, m), if 'basis' in `returns`
    scale : (..., k), if 'scale' in `returns`

    """
    if isinstance(x, (list, tuple)) and callable(x[0]):
        return _pca_callable(x, nb_components, mean, returns, norm)
    else:
        return _pca_tensor(x, nb_components, mean, returns, norm)


def _pca_tensor(x, k, mu, returns, norm):
    """Classic implementation: subtract mean and call SVD."""

    x = torch.as_tensor(x)
    if mu is None:
        mu = torch.mean(x, dim=-2)
    nomu = isinstance(mu, (int, float)) and mu == 0
    mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    if not nomu:
        x = x - mu[..., None, :]

    z, s, u = torch.svd(x, some=True)

    if k:
        if k > min(x.shape[-1], x.shape[-2]):
            raise ValueError('Number of components cannot be larger '
                             'than min(N,M)')
        z = z[..., k]
        u = u[..., k]
        s = s[..., k]

    if 'latent' not in norm:
        z.mul_(s[..., None, :])
    if 'basis' not in norm:
        u.mul_(s[..., None, :])
    u = u.transpose(-1, -2)

    out = []
    returns = returns or ''
    for var in returns.split('+'):
        if var == 'latent':
            out.append(z)
        elif var == 'basis':
            out.append(u)
        elif var == 'scale':
            out.append(s)
    return out[0] if len(out) == 1 else tuple(out)


def _pca_callable(x, k, mu, returns, norm):
    """Implementation that loads tensors one at a time.
    1) Compute the NxN covariance matrix
    2) Use SVD to compute the NxK latent vectors
    3) Compute the KxM basis by projection (= matmul by pseudoinversed latent)
    """

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

    if k and k > min(n, m):
        raise ValueError('Number of components cannot be larger '
                         'than min(N,M)')
    k = k or min(n, m)

    # build NxN covariance matrix
    cov = torch.empty([*shape, n, n])
    for n1 in range(n):
        x1 = torch.as_tensor(x[n1](), **backend)
        if not nomu:
            x1 = x1 - mu
        cov[..., n1, n1] = x1.square().sum(-1)
        for n2 in range(n1+1, n):
            x2 = torch.as_tensor(x[n2](), **backend)
            if not nomu:
                x2 = x2 - mu
            x2 = x2.mul(x1).sum(-1)
            cov[..., n1, n2] = x2
            cov[..., n2, n1] = x2

    # compute svd
    z, s, _ = torch.svd(cov, some=True)  # [..., n, k]
    s = s.sqrt_()
    z = z[..., :k]
    s = s[..., :k]

    if 'basis' in returns:
        # build basis by projection
        iz = torch.pinverse(z * s[..., None, :])
        u = iz.new_zeros([*shape, k, m])
        for n1 in range(n):
            x1 = torch.as_tensor(x[n1](), **backend)
            if not nomu:
                x1 -= mu
            u += iz[..., :, n1, None] * x1[..., None, :]

        if 'basis' not in norm:
            u *= s[..., None]

    if 'latent' not in norm:
        z *= s[..., None, :]

    out = []
    returns = returns or ''
    for var in returns.split('+'):
        if var == 'latent':
            out.append(z)
        elif var == 'basis':
            out.append(u)
        elif var == 'scale':
            out.append(s)
    return out[0] if len(out) == 1 else tuple(out)
