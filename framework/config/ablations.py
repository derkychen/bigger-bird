from dataclasses import dataclass

@dataclass
class AblationsConfig:
    locals: bool = True
    globals: bool = True
    teleports: bool = True

ablations_cfg = AblationsConfig()

