from .base import ICLBatch, ICLTask
from .linear import LinearTask
from .polynomial import PolynomialTask
from .chebyshev import ChebyshevTask

TASK_REGISTRY: dict[str, type[ICLTask]] = {
    "linear": LinearTask,
    "polynomial": PolynomialTask,
    "chebyshev": ChebyshevTask,
}
