"""Static per-park data: run-scoring factor, coordinates (for weather), and
dome/retractable-roof status. Park factors are not available from the free
MLB Stats API, so this is a hardcoded table sourced from publicly published
multi-year park factor figures (100 = league-neutral; e.g. 112 means that
park inflates run scoring by ~12% relative to average).

Refresh yearly -- park factors drift a little season to season, and teams
occasionally play in temporary venues (renovations, relocations).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Park:
    run_factor: int      # 100 = neutral; >100 favors offense, <100 favors pitching
    lat: float
    lon: float
    is_dome: bool         # fixed or reliably-closed roof -> skip weather adjustment


PARKS = {
    108: Park(100, 33.8003, -117.8827, False),   # Angel Stadium (LAA)
    109: Park(103, 33.4455, -112.0667, True),    # Chase Field (ARI) - roof usually closed in summer heat
    110: Park(97, 39.2839, -76.6217, False),     # Camden Yards (BAL)
    111: Park(104, 42.3467, -71.0972, False),    # Fenway Park (BOS)
    112: Park(102, 41.9484, -87.6553, False),    # Wrigley Field (CHC) - wind-dependent, see weather adj
    113: Park(101, 39.0975, -84.5061, False),    # Great American Ball Park (CIN)
    114: Park(96, 41.4962, -81.6852, False),     # Progressive Field (CLE)
    115: Park(112, 39.7559, -104.9942, False),   # Coors Field (COL) - altitude, biggest hitter's park
    116: Park(97, 42.3390, -83.0485, False),     # Comerica Park (DET)
    117: Park(99, 29.7573, -95.3555, True),      # Daikin Park (HOU)
    118: Park(98, 39.0517, -94.4803, False),     # Kauffman Stadium (KC)
    119: Park(98, 34.0739, -118.2400, False),    # Dodger Stadium (LAD)
    120: Park(101, 38.8730, -77.0074, False),    # Nationals Park (WSH)
    121: Park(96, 40.7571, -73.8458, False),     # Citi Field (NYM)
    133: Park(99, 38.5802, -121.5140, False),    # Sutter Health Park (ATH, temporary home)
    134: Park(97, 40.4469, -80.0057, False),     # PNC Park (PIT)
    135: Park(96, 32.7073, -117.1566, False),    # Petco Park (SD)
    136: Park(95, 47.5914, -122.3325, True),     # T-Mobile Park (SEA)
    137: Park(90, 37.7786, -122.3893, False),    # Oracle Park (SF) - marine layer, pitcher's park
    138: Park(99, 38.6226, -90.1928, False),     # Busch Stadium (STL)
    139: Park(97, 27.9483, -82.4572, True),      # Tropicana Field / temporary home (TB)
    140: Park(100, 32.7473, -97.0842, True),     # Globe Life Field (TEX) - retractable, usually closed
    141: Park(102, 43.6414, -79.3894, True),     # Rogers Centre (TOR) - retractable
    142: Park(99, 44.9817, -93.2777, False),     # Target Field (MIN)
    143: Park(105, 39.9061, -75.1665, False),    # Citizens Bank Park (PHI)
    144: Park(100, 33.8908, -84.4678, False),    # Truist Park (ATL)
    145: Park(100, 41.8299, -87.6338, False),    # Rate Field (CWS)
    146: Park(97, 25.7781, -80.2196, True),      # loanDepot park (MIA) - fixed roof
    147: Park(101, 40.8296, -73.9262, False),    # Yankee Stadium (NYY)
    158: Park(101, 43.0280, -87.9712, True),     # American Family Field (MIL) - retractable, often closed
}

NEUTRAL_PARK = Park(100, 0.0, 0.0, True)


def get_park(team_id: int) -> Park:
    """Return the Park for a team's home venue, or a league-neutral
    placeholder if the team isn't in the table.
    """
    return PARKS.get(team_id, NEUTRAL_PARK)


# US timezone per team. A pure longitude cutoff doesn't work -- the real
# Eastern/Central line snakes well east of a straight meridian in the upper
# Midwest (Chicago and Milwaukee are both Central despite sitting east of
# -90 deg longitude) -- so this is an explicit per-team lookup instead of a
# formula. Only 30 fixed cities, so a lookup table is both simpler and
# actually correct, unlike a general-purpose longitude heuristic.
TIMEZONES = {
    108: "pacific", 109: "mountain", 110: "eastern", 111: "eastern", 112: "central",
    113: "eastern", 114: "eastern", 115: "mountain", 116: "eastern", 117: "central",
    118: "central", 119: "pacific", 120: "eastern", 121: "eastern", 133: "pacific",
    134: "eastern", 135: "pacific", 136: "pacific", 137: "pacific", 138: "central",
    139: "eastern", 140: "central", 141: "eastern", 142: "central", 143: "eastern",
    144: "eastern", 145: "central", 146: "eastern", 147: "eastern", 158: "central",
}


def timezone_bucket(team_id: int) -> str:
    return TIMEZONES.get(team_id, "eastern")
