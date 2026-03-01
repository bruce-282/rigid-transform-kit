"""
rigid_transform_kit.core.frame
===============================
Coordinate frame identifiers.
"""

from enum import Enum, auto


class Frame(Enum):
    """Coordinate frame identifiers.

    Extend freely — the transform chain validation only checks enum equality.
    """
    BASE = auto()
    CAMERA = auto()
    FLANGE = auto()
    TCP = auto()
    OBJECT = auto()
    WORLD = auto()
    MARKER = auto()

    def __repr__(self) -> str:
        return self.name
