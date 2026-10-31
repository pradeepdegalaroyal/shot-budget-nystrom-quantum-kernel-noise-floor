"""Shot-budgeted Nystrom approximation of fidelity quantum kernels."""

from .feature_map import exact_kernel, zz_states
from .nystrom import nystrom_features, noise_floor_threshold
from .shots import depolarising_fidelity, sample_kernel_block

__all__ = [
    "exact_kernel",
    "zz_states",
    "nystrom_features",
    "noise_floor_threshold",
    "depolarising_fidelity",
    "sample_kernel_block",
]
__version__ = "0.1.0"
