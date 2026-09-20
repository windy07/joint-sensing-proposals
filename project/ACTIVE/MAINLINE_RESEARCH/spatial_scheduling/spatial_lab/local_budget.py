"""Stage4 world budget is enforced inside World.step, not this legacy hook."""
from .core import ContractError
def charge_world_cycle(config):
    raise ContractError("LEGACY_BUDGET_HOOK_RETIRED_USE_STAGE4_ENTRY")
