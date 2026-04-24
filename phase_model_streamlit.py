"""
Streamlit web app for the theoretical cadence model.

=== Run locally ===
    pip install streamlit
    streamlit run phase_model_streamlit.py

=== Deploy to a public URL (Streamlit Community Cloud, free) ===
1. Put this file and a requirements.txt listing its dependencies into a
   public GitHub repo. Minimum requirements.txt:
       streamlit
       numpy
       matplotlib
2. Sign in at https://share.streamlit.io with your GitHub account.
3. Click "New app", select the repo/branch, point the main file path at
   phase_model_streamlit.py, and deploy. You get a persistent URL to share.
4. Redeploy any time you push changes to the repo.

For a private deployment, Streamlit Community Cloud also supports private
GitHub repos if you grant it access. Alternatively, run locally on a server
and share via your network.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ============================================================
# Model
# ============================================================

PHASES = ["Preparation", "Initiation", "Stabilisation", "Maintenance"]

PHASE_COLOURS = {
    "Preparation":   "#D3D3D3",
    "Initiation":    "#ADD8E6",
    "Stabilisation": "#8FBF8F",
    "Maintenance":   "#B19CD9",
}

# Stated assumptions (not adjustable).
PREP_FRACTION = 0.05
PREP_DURATION = 1
PREP_GAP_WEEKS = (2, 8)
MAINT_RAMP_TRANSITIONS = 3

SEED = 42
N_PATIENTS = 5_000


def _sample_truncnorm(rng, mean, sd, n, lo=0.0, hi=1.0):
    """Normal(mean, sd) truncated to [lo, hi] via rejection sampling."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        batch = rng.normal(mean, sd, size=(n - filled) * 2 + 10)
        valid = batch[(batch >= lo) & (batch <= hi)]
        k = min(len(valid), n - filled)
        out[filled:filled + k] = valid[:k]
        filled += k
    return out


def simulate(n_transitions, complexity_mean, complexity_sd,
             stab_earliest, stab_median, stab_latest,
             maint_earliest, maint_latest,
             init_gap_min, init_gap_max,
             stab_gap_min, stab_gap_max,
             maint_gap_min, maint_gap_max):
    rng = np.random.default_rng(SEED)
    c = _sample_truncnorm(rng, complexity_mean, complexity_sd, N_PATIENTS)

    prep_mask = rng.random(N_PATIENTS) < PREP_FRACTION
    prep_offset = np.where(prep_mask, PREP_DURATION, 0).astype(float)

    t_stab = np.interp(
        c,
        [0.0, 0.20, 0.50, 0.80, 1.0],
        [stab_earliest, stab_earliest, stab_median, stab_latest, stab_latest],
    ) + prep_offset

    t_maint = np.interp(
        c,
        [0.0, 0.20, 0.80, 1.0],
        [maint_earliest, maint_earliest, maint_latest, maint_latest],
    ) + prep_offset
    t_maint = np.maximum(t_maint, t_stab + 1)

    t = np.arange(1, n_transitions + 1)[None, :]
    t_stab_b = t_stab[:, None]
    t_maint_b = t_maint[:, None]
    phase = np.full((N_PATIENTS, n_transitions), 3, dtype=np.int8)
    phase = np.where(t < t_maint_b, 2, phase)
    phase = np.where(t < t_stab_b,  1, phase)
    phase = np.where(t <= prep_offset[:, None], 0, phase)

    gap_mins_days = np.array(
        [PREP_GAP_WEEKS[0], init_gap_min, stab_gap_min, maint_gap_min], dtype=float) * 7.0
    gap_maxs_days = np.array(
        [PREP_GAP_WEEKS[1], init_gap_max, stab_gap_max, maint_gap_max], dtype=float) * 7.0
    lo = gap_mins_days[phase]
    hi = gap_maxs_days[phase]
    mode = lo.copy()

    stab_dur = np.maximum(t_maint_b - t_stab_b, 1.0)
    stab_progress = np.clip((t - t_stab_b) / stab_dur, 0.0, 1.0)
    mode = np.where(phase == 2, lo + stab_progress * (hi - lo), mode)

    maint_progress = np.clip((t - t_maint_b) / max(MAINT_RAMP_TRANSITIONS, 1), 0.0, 1.0)
    mode = np.where(phase == 3, lo + maint_progress * (hi - lo), mode)

    mode = np.clip(mode, lo, hi)
    gaps = rng.triangular(lo, mode, hi)

    return phase, c, gaps


def phase_proportions(phase_arr, n_transitions):
    props = np.zeros((4, n_transitions))
    for k in range(4):
        props[k] = (phase_arr == k).mean(axis=0)
    return props


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Theoretical Cadence Model",
    page_icon=None,
    layout="wide",
)

st.title("Theoretical caseload cadence model")
st.markdown(
    "This model is based on the appointment cadence principle described "
    "[here](https://coda.io/d/Clinician-Knowledge-Base_dmavOZ76Uoi/Appointment-Cadence_su-QMYS0#_luw2WBku). "
    "Explore how a clinician's caseload distributes across clinical phases as patients "
    "progress through successive appointments. Adjust the sliders on the left to see the "
    "effect on phase mix and expected appointment cadence."
)

with st.expander("About this model"):
    st.markdown(f"""
### What it models

A patient progresses through four clinical phases:

- **Preparation** - patients needing more investigations before initiation can start.
- **Initiation** - titration and early treatment establishment.
- **Stabilisation** - consolidation once an effective plan is in place.
- **Maintenance** - routine follow-up once stable.

Each **transition** on the chart x-axis is the gap between two consecutive appointments: transition 1 is the gap between appointment 1 and appointment 2, transition 2 between appointment 2 and 3, and so on. The phase shown at a given transition is the phase of that gap.

How quickly a patient moves through each phase depends on their **complexity**
(a 0 to 1 score combining factors like comorbidity, symptom severity, psychosocial
load, and treatment resistance). The caseload is modelled as a bell-shaped
distribution of complexity scores.

### Reading the three panels

- **Complexity distribution** (left): how patient complexity is spread across the caseload.
- **Caseload phase mix** (middle): at each appointment transition, the proportion of the caseload in each phase.
- **Expected cadence spread** (right): the distribution of weeks between appointments at each transition. The median line rises as patients move into later, longer-spaced phases.

### Key assumptions

**How appointments are distributed within each phase's cadence range (lower to upper bracket):**

- **Initiation**: appointments are predominantly booked at the shorter end of the Initiation cadence range throughout the phase.
- **Stabilisation**: appointments are predominantly booked at the shorter end when a patient first enters Stabilisation. The typical gap then slides gradually toward the longer end as they progress through the phase, reaching the longer end by the time they enter Maintenance.
- **Maintenance**: appointments are predominantly booked at the shorter end when a patient first enters Maintenance. The typical gap slides gradually toward the longer end over their first {MAINT_RAMP_TRANSITIONS} Maintenance appointments and then stays at the longer end.

**Preparation is a fixed assumption**, not adjustable via sliders: {int(PREP_FRACTION * 100)}% of patients spend {PREP_DURATION} transition in Preparation (with a {PREP_GAP_WEEKS[0]} to {PREP_GAP_WEEKS[1]} week gap) before moving into Initiation. This represents patients awaiting pre-initiation investigations.

**C20 / C50 / C80** refer to complexity values 0.20, 0.50, and 0.80 - roughly the simplest 5%, median, and most complex 5% of the caseload at default settings.
""")

with st.sidebar:
    st.header("Parameters")

    with st.expander("Simulation", expanded=True):
        n_transitions = st.slider(
            "Number of transitions", 5, 20, 10, 1,
            help="Number of future appointment transitions to model.",
        )

    with st.expander("Caseload complexity", expanded=True):
        complexity_mean = st.slider(
            "Complexity mean", 0.20, 0.80, 0.50, 0.01,
            help="Average complexity of the caseload (0 = all simple, 1 = all complex).",
        )
        complexity_sd = st.slider(
            "Complexity SD", 0.05, 0.30, 0.18, 0.01,
            help="Spread of complexity. Narrow = homogeneous caseload, wide = diverse mix of simple and complex patients.",
        )

    with st.expander("Stabilisation start (transition)", expanded=True):
        stab_earliest = st.slider(
            "Simple (C20)",  1, 10, 2, 1,
            help="Transition at which the simplest patients (complexity 0.20) move into Stabilisation.",
        )
        stab_median = st.slider(
            "Median (C50)",  1, 10, 4, 1,
            help="Transition at which the median patient (complexity 0.50) moves into Stabilisation.",
        )
        stab_latest = st.slider(
            "Complex (C80)", 1, 10, 7, 1,
            help="Transition at which the most complex patients (complexity 0.80) move into Stabilisation.",
        )

    with st.expander("Maintenance start (transition)", expanded=True):
        maint_earliest = st.slider(
            "Simple (C20) ",  1, 15, 4, 1, key="maint_e",
            help="Transition at which the simplest patients move into Maintenance.",
        )
        maint_latest = st.slider(
            "Complex (C80) ", 1, 15, 8, 1, key="maint_l",
            help="Transition at which the most complex patients move into Maintenance.",
        )

    with st.expander("Initiation cadence (weeks)", expanded=False):
        init_gap_min = st.slider(
            "Initiation gap min", 0, 8, 2, 1,
            help="Shortest number of weeks between appointments during Initiation.",
        )
        init_gap_max = st.slider(
            "Initiation gap max", 0, 8, 4, 1,
            help="Longest number of weeks between appointments during Initiation.",
        )

    with st.expander("Stabilisation cadence (weeks)", expanded=False):
        stab_gap_min = st.slider(
            "Stabilisation gap min", 0, 12, 4, 1,
            help="Shortest number of weeks between appointments during Stabilisation.",
        )
        stab_gap_max = st.slider(
            "Stabilisation gap max", 0, 12, 8, 1,
            help="Longest number of weeks between appointments during Stabilisation.",
        )

    with st.expander("Maintenance cadence (weeks)", expanded=False):
        maint_gap_min = st.slider(
            "Maintenance gap min", 0, 24, 8, 1,
            help="Shortest number of weeks between appointments during Maintenance.",
        )
        maint_gap_max = st.slider(
            "Maintenance gap max", 0, 24, 15, 1,
            help="Longest number of weeks between appointments during Maintenance.",
        )

# ---- Run simulation ----
phase_arr, c, gaps = simulate(
    n_transitions, complexity_mean, complexity_sd,
    stab_earliest, stab_median, stab_latest,
    maint_earliest, maint_latest,
    init_gap_min, init_gap_max,
    stab_gap_min, stab_gap_max,
    maint_gap_min, maint_gap_max,
)
props = phase_proportions(phase_arr, n_transitions)
t = np.arange(1, n_transitions + 1)

# ---- Build figure ----
fig, (ax0, ax1, ax2) = plt.subplots(
    1, 3, figsize=(18, 5.5),
    gridspec_kw={"width_ratios": [1, 2, 2]},
)

# Complexity histogram
ax0.hist(c, bins=40, color="#6c8ebf", edgecolor="white")
for x in (0.20, 0.50, 0.80):
    ax0.axvline(x, color="#777", ls=":", alpha=0.5)
ax0.set_xlim(0, 1)
ax0.set_xlabel("Complexity")
ax0.set_ylabel("Patient count")
ax0.set_title(f"Complexity distribution\nmean={complexity_mean:.2f}, SD={complexity_sd:.2f}", fontsize=11)

# Phase mix
ax1.stackplot(
    t, props,
    labels=PHASES,
    colors=[PHASE_COLOURS[p] for p in PHASES],
    alpha=0.95,
)
ax1.set_xlim(1, n_transitions)
ax1.set_ylim(0, 1)
ax1.set_xticks(t)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.set_yticklabels([f"{int(y * 100)}%" for y in np.arange(0, 1.01, 0.2)])
ax1.set_xlabel("Appointment transition")
ax1.set_ylabel("Proportion of caseload")
ax1.set_title("Caseload phase mix over transitions", fontsize=11)
ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Gap spread (weeks)
gaps_weeks = gaps / 7.0
p50 = np.median(gaps_weeks, axis=0)
p25 = np.percentile(gaps_weeks, 25, axis=0)
p75 = np.percentile(gaps_weeks, 75, axis=0)
p10 = np.percentile(gaps_weeks, 10, axis=0)
p90 = np.percentile(gaps_weeks, 90, axis=0)
ax2.fill_between(t, p10, p90, color="#888", alpha=0.2, label="10-90%")
ax2.fill_between(t, p25, p75, color="#888", alpha=0.4, label="IQR")
ax2.plot(t, p50, "o-", color="black", label="Median")
ax2.set_xlim(1, n_transitions)
ax2.set_xticks(t)
ax2.set_xlabel("Appointment transition")
ax2.set_ylabel("Gap to next appointment (weeks)")
ax2.set_title("Expected cadence spread per transition", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()

st.pyplot(fig)
st.caption(
    "Left: distribution of patient complexity across the caseload. "
    "Middle: proportion of caseload in each phase at each transition. "
    "Right: median gap (black line), interquartile range (darker band), and 10-to-90 percentile range (lighter band) "
    "of weeks between appointments at each transition."
)

# ---- Download button ----
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
st.download_button(
    label="Download chart as PNG",
    data=buf,
    file_name="theoretical_caseload_model.png",
    mime="image/png",
)
