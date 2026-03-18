from .base import ICLBatch, ICLTask
from .linear import LinearTask
from .chebyshev import ChebyshevTask
from .sparse_linear import SparseLinearTask
from .quadratic import QuadraticTask

TASK_REGISTRY: dict[str, type[ICLTask]] = {
    "linear": LinearTask,
    "chebyshev": ChebyshevTask,
    "sparse_linear": SparseLinearTask,
    "quadratic": QuadraticTask,
}
