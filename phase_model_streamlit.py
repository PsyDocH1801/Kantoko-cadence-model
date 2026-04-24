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
MAINT_RAMP_TRANSITIONS = 5

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
st.caption(
    f"Prep (stated assumption): {int(PREP_FRACTION * 100)}% of caseload, "
    f"{PREP_DURATION} transition, {PREP_GAP_WEEKS[0]}-{PREP_GAP_WEEKS[1]} week gap. "
    f"Gap shape: triangular; Init mode at min; Stab and Maint modes slide min to max "
    f"(Maint ramp over {MAINT_RAMP_TRANSITIONS} transitions)."
)

with st.sidebar:
    st.header("Parameters")

    with st.expander("Simulation", expanded=True):
        n_transitions = st.slider("Number of transitions", 5, 20, 10, 1)

    with st.expander("Caseload complexity", expanded=True):
        complexity_mean = st.slider("Complexity mean", 0.20, 0.80, 0.50, 0.01)
        complexity_sd = st.slider("Complexity SD", 0.05, 0.30, 0.18, 0.01)

    with st.expander("Stab start (transition)", expanded=True):
        stab_earliest = st.slider("Simple (C20)",  1, 10, 2, 1)
        stab_median   = st.slider("Median (C50)",  2, 15, 5, 1)
        stab_latest   = st.slider("Complex (C80)", 5, 20, 8, 1)

    with st.expander("Maint start (transition)", expanded=True):
        maint_earliest = st.slider("Simple (C20) ",  2, 15, 5, 1, key="maint_e")
        maint_latest   = st.slider("Complex (C80) ", 5, 30, 10, 1, key="maint_l")

    with st.expander("Init cadence (weeks)", expanded=False):
        init_gap_min = st.slider("Init gap min", 0, 8, 2, 1)
        init_gap_max = st.slider("Init gap max", 1, 16, 3, 1)

    with st.expander("Stab cadence (weeks)", expanded=False):
        stab_gap_min = st.slider("Stab gap min", 1, 12, 3, 1)
        stab_gap_max = st.slider("Stab gap max", 2, 24, 8, 1)

    with st.expander("Maint cadence (weeks)", expanded=False):
        maint_gap_min = st.slider("Maint gap min", 4, 26, 10, 1)
        maint_gap_max = st.slider("Maint gap max", 8, 52, 15, 1)

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

# ---- Anchor summary ----
def at(col): return col - 1
prep_t1   = (phase_arr[:, at(1)]  == 0).mean()
prep_t2   = (phase_arr[:, at(2)]  == 0).mean()
stab_by_t5  = (phase_arr[:, at(5)]  >= 2).mean()
init_at_t10 = (phase_arr[:, at(min(10, n_transitions))] == 1).mean() if n_transitions >= 10 else None

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Prep at T1",            f"{prep_t1:.1%}")
col_b.metric("Prep at T2",            f"{prep_t2:.1%}")
col_c.metric("Reached Stab by T5",    f"{stab_by_t5:.1%}")
if init_at_t10 is not None:
    col_d.metric("Still in Init at T10", f"{init_at_t10:.1%}")

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
