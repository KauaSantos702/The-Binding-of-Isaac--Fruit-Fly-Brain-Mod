"""Declarative spec of the Drosophila sub-connectome we extract for the Isaac mod.

Everything here is a selection over *real* FlyWire v783 cell types.  No weights or
connections are invented anywhere in this project: the spec only says which
neurons to keep, and how many of each population to subsample.  Weights always
come from ``data/2025_Connectivity_783.parquet`` (signed synapse counts).

The circuit implements one complete, well-characterised sensorimotor loop:

    visual projection neurons  ->  central complex  ->  descending neurons
    (LC/LPLC looming,              (EPG/PEN ring         (DNa02 turning,
     HS/VS optic flow)              attractor compass,    DNp09 escape,
                                    PFL2/PFL3 steering)   DNp01 giant fibre)

and the loop is closed in-game: the fly's own motion regenerates optic flow and
angular-velocity input, which feeds back into the compass.
"""

# ---------------------------------------------------------------------------
# Roles.  These are *our* functional labels, used by the mod to know which
# neurons to inject current into and which to read spikes from.  They do not
# affect connectivity.
# ---------------------------------------------------------------------------
ROLE_LOOMING = "looming"        # driven by nearby/approaching threats
ROLE_SMALL_OBJ = "small_object"  # driven by small moving objects (enemies, pickups)
ROLE_FLOW = "optic_flow"        # driven by the fly's own translation/rotation
ROLE_BAR = "bar"                # driven by a visual landmark (the player)
ROLE_LANDMARK = "landmark"      # MeTu/TuBu/TuTu: the sky-compass / landmark pathway
ROLE_COMPASS = "compass"        # EPG ring attractor + friends
ROLE_CX = "central_complex"     # rest of the central complex
ROLE_STEER = "steering"         # PFL output stage
ROLE_DN = "descending"          # motor command output
ROLE_BRIDGE = "bridge"          # interneurons kept to preserve real pathways


# ---------------------------------------------------------------------------
# Visual projection neurons.  ``cap`` = max neurons kept per (type, side);
# outgoing weights of the kept ones are rescaled by n_total/n_kept so the total
# drive each downstream partner receives is preserved.
#
# Literature anchors for the roles:
#   LC4     angular velocity of a looming disc -> DNp09 escape      (Ache 2019)
#   LPLC2   radially symmetric looming -> giant fibre DNp01         (Klapoetke 2017)
#   LC6     looming -> escape                                       (Wu 2016)
#   LPLC1   looming                                                 (Klapoetke 2017)
#   LC16    looming / backward locomotion + turning                 (Tanaka 2020)
#   LC11    small object motion                                     (Keles 2017)
#   LC18    small object                                            (Wu 2016)
#   LC17    small object                                            (Wu 2016)
#   LC9/LC10 object position / tracking                             (Ribeiro 2018)
#   HS/VS/H2 wide-field optic flow -> optomotor course control      (Schnell 2010)
# ---------------------------------------------------------------------------
VISUAL = [
    # (cell_type,        role,             cap_per_side)
    ("LC4",              ROLE_LOOMING,     16),
    ("LPLC2",            ROLE_LOOMING,     16),
    ("LC6",              ROLE_LOOMING,     12),
    ("LPLC1",            ROLE_LOOMING,     12),
    ("LC16",             ROLE_LOOMING,     12),
    ("LC11",             ROLE_SMALL_OBJ,   12),
    ("LC18",             ROLE_SMALL_OBJ,   10),
    ("LC17",             ROLE_SMALL_OBJ,   10),
    ("LC9",              ROLE_BAR,         10),
    ("LC10a",            ROLE_BAR,          8),
    ("LC10d",            ROLE_BAR,          8),
    ("LC12",             ROLE_SMALL_OBJ,    8),
    ("LC13",             ROLE_SMALL_OBJ,    6),
    ("LC15",             ROLE_SMALL_OBJ,    6),
    ("LC21",             ROLE_SMALL_OBJ,    6),
    ("LC22",             ROLE_SMALL_OBJ,    6),
    ("LPLC4",            ROLE_LOOMING,      8),
    ("HSN",              ROLE_FLOW,         4),
    ("HSE",              ROLE_FLOW,         4),
    ("HSS",              ROLE_FLOW,         4),
    ("H2",               ROLE_FLOW,         4),
    ("VS1",              ROLE_FLOW,         2),
    ("VS2",              ROLE_FLOW,         2),
    ("VS3",              ROLE_FLOW,         2),
    ("VS4",              ROLE_FLOW,         2),
    ("VS5",              ROLE_FLOW,         2),
    ("VS6",              ROLE_FLOW,         2),
]

# ---------------------------------------------------------------------------
# The visual landmark / sky-compass pathway:  MeTu -> TuBu -> ER -> EPG.
# This is how a real fly anchors its heading representation to the outside
# world (Omoto 2017; Sun 2017; Hulse 2021; Garner 2024).  MeTu neurons are
# retinotopic in the medulla, so the game can drive them by azimuth, and the
# connectome then decides which compass wedge that corresponds to.
# ---------------------------------------------------------------------------
LANDMARK = [
    ("MeTu1",            ROLE_LANDMARK,    10),
    ("MeTu2a",           ROLE_LANDMARK,     6),
    ("MeTu2b",           ROLE_LANDMARK,     4),
    ("MeTu3a",           ROLE_LANDMARK,     5),
    ("MeTu3b",           ROLE_LANDMARK,     6),
    ("MeTu3c",           ROLE_LANDMARK,    10),
    ("MeTu4a",           ROLE_LANDMARK,     8),
    ("MeTu4b",           ROLE_LANDMARK,     4),
    ("MeTu4c",           ROLE_LANDMARK,     6),
    ("MeTu4d",           ROLE_LANDMARK,     4),
]
# TuBu and TuTu are small enough to keep whole.
LANDMARK_COMPLETE_PREFIX = [("TuBu", ROLE_LANDMARK), ("TuTu", ROLE_LANDMARK)]

# ---------------------------------------------------------------------------
# Central complex.  The heading system is kept complete -- it is the part that
# actually has to hold a persistent attractor state, so subsampling it would
# break the dynamics.
# ---------------------------------------------------------------------------
CX_COMPLETE = [
    # ring attractor / heading (Seelig & Jayaraman 2015; Turner-Evans 2017)
    ("EPG",              ROLE_COMPASS),
    ("EPGt",             ROLE_COMPASS),
    ("PEN_a(PEN1)",      ROLE_COMPASS),
    ("PEN_b(PEN2)",      ROLE_COMPASS),
    ("PEG",              ROLE_COMPASS),
    ("Delta7",           ROLE_COMPASS),   # global inhibition across PB glomeruli
    ("EL",               ROLE_COMPASS),
    # steering output stage (Hulse 2021; Westeinde 2024; Mussells Pires 2024)
    ("PFL1",             ROLE_STEER),
    ("PFL2",             ROLE_STEER),
    ("PFL3",             ROLE_STEER),
]

# Ring neurons carry the landmark signal into the compass and are the dominant
# inhibitory input to EPG, so they are kept whole rather than subsampled.
CX_COMPLETE_PREFIX = [("ER", ROLE_CX)]

# Capped central-complex populations (goal vector / speed machinery)
CX_CAPPED = [
    ("PFNa",             ROLE_CX, 10),
    ("PFNd",             ROLE_CX, 12),
    ("PFNm",             ROLE_CX,  8),
    ("PFNp",             ROLE_CX,  8),
    ("PFNv",             ROLE_CX, 10),
    ("PFR",              ROLE_CX, 10),
    ("PFR_a",            ROLE_CX,  4),
    ("PFGs",             ROLE_CX,  8),
    ("FR1",              ROLE_CX,  8),
    ("FR2",              ROLE_CX,  8),
    ("FC1A,FC1B,FC1F",   ROLE_CX,  8),
    ("FC1C,FC1E",        ROLE_CX,  8),
    ("FC1D",             ROLE_CX,  6),
    ("FC2A",             ROLE_CX,  6),
    ("FC2B",             ROLE_CX,  6),
    ("FC2C",             ROLE_CX,  6),
    ("FC3",              ROLE_CX,  8),
]

# hDelta / vDelta (fan-shaped body local columnar cells) and LNO (noduli speed
# input) are matched by prefix because there are many small subtypes.
CX_PREFIX_CAPPED = [
    ("hDelta",           ROLE_CX,  4),
    ("vDelta",           ROLE_CX,  3),
    ("LNO",              ROLE_CX,  4),
    ("ExR",              ROLE_CX,  4),
    ("SpsP",             ROLE_CX,  4),
    ("IbSpsP",           ROLE_CX,  4),
]

# ---------------------------------------------------------------------------
# Descending neurons.  We keep every DN that is strongly wired to the rest of
# the circuit (selected by connectivity in build_circuit.py), plus this list of
# named DNs with known behavioural roles, which are always kept.
# ---------------------------------------------------------------------------
DN_ALWAYS = {
    "DNp09": "escape / freeze; stops walking, triggered by LC4+LPLC2 looming",
    "DNp01": "giant fibre; short-latency escape takeoff",
    "DNa02": "ipsilateral turning; the main steering output of PFL3",
    "DNa01": "turning / course correction",
    "DNa03": "turning",
    "DNa04": "turning",
    "DNb01": "forward walking modulation",
    "DNb02": "forward walking modulation",
    "DNp10": "escape-related",
    "DNp11": "escape-related",
    "DNg13": "locomotor",
    "DNp06": "escape-related",
}

# Behavioural read-out groups the mod uses.  ``(name, cell_types, side)``
MOTOR_READOUT = [
    ("turn_left",   ["DNa02", "DNa01", "DNa03"], "left"),
    ("turn_right",  ["DNa02", "DNa01", "DNa03"], "right"),
    ("escape",      ["DNp09", "DNp01", "DNp10", "DNp11", "DNp06"], None),
    ("forward",     ["DNb01", "DNb02", "DNa04", "DNg13"], None),
]

# Max number of extra "bridge" interneurons (not in any list above) kept because
# they carry strong real pathways between the selected populations.
MAX_BRIDGE = 340
BRIDGE_MIN_SYN = 90    # min synapses from core AND to core to qualify
BRIDGE_ROUNDS = 2      # repeat, so bridges-of-bridges (premotor LAL/PS) get in
