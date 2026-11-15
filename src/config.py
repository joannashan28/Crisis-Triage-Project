"""
Shared settings for the whole project.

Anything that more than one script needs to agree on (folder locations,
CrisisBench file names, label strings, display colors) lives here, so a
change only has to be made in one place.
"""

import os


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

# The project root is the folder that contains src/, server/, demo/, etc.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ---------------------------------------------------------------------------
# CrisisBench file names
#
# These are the English, combined, official-split files. prepare_data.py
# searches for them anywhere inside the CrisisBench folder you give it, so
# it does not matter exactly where you extracted the download.
# ---------------------------------------------------------------------------

SPLITS = ["train", "dev", "test"]

INFORMATIVENESS_FILE_TEMPLATE = (
    "crisis_consolidated_informativeness_filtered_lang_en_{split}.tsv"
)
HUMANITARIAN_FILE_TEMPLATE = (
    "crisis_consolidated_humanitarian_filtered_lang_en_{split}.tsv"
)

# ---------------------------------------------------------------------------
# Disaster types
#
# Each tweet's "event" column holds an event name such as
# "2015_nepal_earthquake" or "2013_alberta_floods-ontopic". We work out the
# disaster type from words in that name. The rules are checked in order and
# the first match wins, so more specific words come first.
#
# prepare_data.py lists every event name that matched no rule (in
# data_report.json and on screen). If one of those clearly has a type, add
# its keyword here and rerun.
# ---------------------------------------------------------------------------

DISASTER_TYPE_RULES = [
    # (disaster type, words that indicate it)
    ("earthquake", ["earthquake", "_eq"]),
    # Typhoons and cyclones are the same kind of storm as hurricanes,
    # just named differently by region.
    ("hurricane", ["hurricane", "typhoon", "cyclone"]),
    ("tornado", ["tornado"]),
    # 2013_italy_sardinia was the November 2013 Sardinia floods.
    ("flood", ["flood", "sardinia"]),
    ("fire", ["wildfire", "bushfire", "fire"]),
    ("landslide", ["landslide", "mudslide"]),
    ("bombing", ["bombing"]),
    # The 2013 West, Texas event was a fertilizer plant explosion.
    ("explosion", ["explosion", "west_texas", "blast"]),
    ("disease", ["ebola", "mers", "respiratory", "outbreak", "disease"]),
    ("shooting", ["shooting"]),
    # 2014_malaysia_airline was a Malaysia Airlines plane disaster.
    ("crash", ["crash", "derailment", "derail", "airline"]),
    ("collapse", ["collapse"]),
    ("volcano", ["volcano", "eruption"]),
    # Environmental hazards that are not one of the types above.
    ("hazard", ["haze", "meteor", "hazard"]),
]

# CrisisBench's generic "disaster_events" label covers tweets collected
# without a specific event, so it cannot be given one type.
GENERIC_EVENT_NAMES = ["disaster_events"]
GENERIC_DISASTER_TYPE = "none/generic"

UNKNOWN_DISASTER_TYPE = "unknown"


def disaster_type_from_event(event):
    """Return the disaster type for an event name, using DISASTER_TYPE_RULES."""
    name = event.lower()
    name = name.strip()

    if name in GENERIC_EVENT_NAMES:
        return GENERIC_DISASTER_TYPE

    for rule in DISASTER_TYPE_RULES:
        disaster_type = rule[0]
        keywords = rule[1]
        for keyword in keywords:
            if keyword in name:
                return disaster_type

    return UNKNOWN_DISASTER_TYPE


# ---------------------------------------------------------------------------
# Label strings
#
# If prepare_data.py reports that one of these is missing from the data,
# update it here to match the spelling CrisisBench actually uses.
# ---------------------------------------------------------------------------

INFORMATIVE_LABEL = "informative"
NOT_INFORMATIVE_LABEL = "not_informative"
NOT_HUMANITARIAN_LABEL = "not_humanitarian"

# Used only for end-to-end scoring. A tweet is "not actionable" if Stage 1
# calls it not informative OR Stage 2 calls it not humanitarian ("Other").
# Merging these two makes sense for triage: in both cases the tweet does not
# land in any responder-facing category.
NOT_ACTIONABLE_LABEL = "not_actionable"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

DEFAULT_TRANSFORMER = "vinai/bertweet-base"

# Default probability cutoff for Stage 1. A tweet goes on to Stage 2 when
# P(informative) is at least this value. Lowering it lets more tweets
# through, which means fewer real requests for help get filtered out, at the
# cost of more noise in the categories.
DEFAULT_STAGE1_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# How labels are shown in the demo
#
# The order here is the order of the category tabs in the extension:
# the most urgent categories come first.
# ---------------------------------------------------------------------------

LABEL_DISPLAY = [
    {"label": "requests_or_needs", "display": "Requests or needs", "color": "#C62F3B"},
    {"label": "injured_or_dead_people", "display": "Injured or dead people", "color": "#8C2350"},
    {"label": "missing_and_found_people", "display": "Missing or found people", "color": "#D2601A"},
    {"label": "displaced_and_evacuations", "display": "Displaced or evacuations", "color": "#B98500"},
    {"label": "infrastructure_and_utilities_damage", "display": "Infrastructure damage", "color": "#6A4BB0"},
    {"label": "caution_and_advice", "display": "Caution and advice", "color": "#8A7A00"},
    {"label": "affected_individual", "display": "Affected individuals", "color": "#1F6FAE"},
    {"label": "response_efforts", "display": "Response efforts", "color": "#1E8C80"},
    {"label": "donation_and_volunteering", "display": "Donations and volunteering", "color": "#3B7D3F"},
    {"label": "sympathy_and_support", "display": "Sympathy and support", "color": "#A2568F"},
    {"label": NOT_HUMANITARIAN_LABEL, "display": "Other (informative, no category)", "color": "#6B7178"},
    {"label": NOT_INFORMATIVE_LABEL, "display": "Not informative", "color": "#8F959C"},
    {"label": INFORMATIVE_LABEL, "display": "Informative", "color": "#3F6E9A"},
    {"label": NOT_ACTIONABLE_LABEL, "display": "No action needed", "color": "#8F959C"},
]

FALLBACK_COLOR = "#6B7178"


def get_label_display(label):
    """Return the display entry for a label, building a readable one if the
    label is not in LABEL_DISPLAY (for example, a spelling we did not expect)."""
    for entry in LABEL_DISPLAY:
        if entry["label"] == label:
            return entry

    readable_name = label.replace("_", " ")
    readable_name = readable_name.capitalize()
    return {"label": label, "display": readable_name, "color": FALLBACK_COLOR}
