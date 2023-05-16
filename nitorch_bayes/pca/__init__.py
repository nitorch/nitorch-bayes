# TODO: the convention is not the same between pca and ppca in terms
#   of which factor is scaled and which is unitary.
#   Currently, pca returns a unitary basis whereas ppca returns
#   unitary latent coordinates.
#   We could have an option to specify which side should be normalized?
#   Or return the (diagonal) covariance matrix as well?

from .classic import pca
from .probabilistic import ppca
from .variational import vpca
from .categorical import cpca
