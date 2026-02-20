from .base import ICLBatch, ICLTask
from .linear import LinearTask

TASK_REGISTRY: dict[str, type[ICLTask]] = {
    "linear": LinearTask,
}
