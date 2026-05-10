"""Shared labels and sensor scaling constants."""

from __future__ import annotations

ACCEL_SENSITIVITY_16G = 2048.0
GYRO_SENSITIVITY = 16.4
QUATERNION_SENSITIVITY = 1_073_741_824.0
SAMPLE_RATE_HZ = 200.0
GRAVITY_M_S2 = 9.8

MOVEMENT_LABELS = {
    1: "WMFT 1: Forearm to table",
    2: "WMFT 2: Forearm to box",
    3: "WMFT 3: Extend elbow",
    4: "WMFT 4: Extend elbow with weight",
    5: "WMFT 5: Hand to table",
    6: "WMFT 6: Hand to box",
    7: "WMFT 7: Weight to box",
    8: "WMFT 8: Reach and retrieve",
    9: "WMFT 9: Lift can",
    10: "WMFT 10: Lift pencil",
    11: "WMFT 11: Lift paper clip",
    12: "WMFT 12: Stack checkers",
    13: "WMFT 13: Flip cards",
    14: "WMFT 14: Grip strength",
    15: "WMFT 15: Turn key in lock",
    16: "WMFT 16: Fold towel",
    17: "WMFT 17: Lift basket",
    18: "Undefined",
}
