from dataclasses import dataclass

@dataclass
class RouterConfig:
    # Window
    fragment_size: int = 128
    k: int = 64
    alpha_pos_prior: float = 0.10

    # Globals per head (shared across all queries in a layer)
    globals_per_head: int = 4

    # Teleports per head (shared)
    teleports_per_head: int = 2

    # Dense fallback
    dense_fallback_under: int = 512

    # Salience mixing (per-head shared selection)
    sal_a_keynorm: float = 1.0
    sal_b_boundary: float = 0.25

    # Misc / Debug
    log_once_pairs: bool = True
    debug_collect: bool = False

router_cfg = RouterConfig()
