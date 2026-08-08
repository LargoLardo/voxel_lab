"""Semantic cellular-state layout."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StateLayout:
    """Channel indices shared by 2D and 3D states."""

    materials: int = 3
    hidden: int = 8
    energy: bool = False

    @property
    def occupancy(self) -> int:
        return 0

    @property
    def material_slice(self) -> slice:
        return slice(1, 1 + self.materials)

    @property
    def energy_index(self) -> int | None:
        return 1 + self.materials if self.energy else None

    @property
    def hidden_slice(self) -> slice:
        start = 1 + self.materials + int(self.energy)
        return slice(start, start + self.hidden)

    @property
    def channels(self) -> int:
        return 1 + self.materials + int(self.energy) + self.hidden

