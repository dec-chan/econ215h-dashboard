import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def expit(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))

st.set_page_config(
    page_title="One Equation, Many Biases",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Model functions ──────────────────────────────────────────────────

def bayesian_post(prior, lr):
    prior = np.asarray(prior, dtype=float)
    lr = np.asarray(lr, dtype=float)
    odds = prior / (1.0 - prior)
    po = lr * odds
    return po / (1.0 + po)


def qb_post(prior, lr, b1, b2):
    return expit(
        b1 * np.log(np.clip(lr, 1e-12, None))
        + b2 * logit(np.clip(prior, 1e-6, 1 - 1e-6))
    )


def cr_post(prior, eps, rho):
    return eps * prior + (1.0 - eps) * rho


mu = np.linspace(0.005, 0.995, 500)


# ── Load data ────────────────────────────────────────────────────────

@st.cache_data
def load_agarwal():
    df = pd.read_csv(os.path.join(DATA_DIR, "agarwal", "dashboard_data.csv"))
    df["prior"] = df["prior"].clip(0.005, 0.995)
    df["posterior"] = df["posterior"].clip(0.005, 0.995)
    return df


@st.cache_data
def load_eoy():
    return pd.read_csv(os.path.join(DATA_DIR, "eoy", "dashboard_data.csv"))


@st.cache_data
def load_grether():
    df = pd.read_csv(os.path.join(DATA_DIR, "grether", "grether_aggregate_data.csv"))
    prior_map = {"1/3": 1 / 3, "1/2": 1 / 2, "2/3": 2 / 3}
    df["prior_num"] = df["prior"].map(prior_map)
    df = df[df["total"] > 0].copy()
    df["frac_A"] = df["num_chose_A"] / df["total"]
    df["lr"] = 2.0 ** (2 * df["num_N"] - 6)
    df["bayes_post"] = bayesian_post(df["prior_num"].values, df["lr"].values)
    return df


ag_data = load_agarwal()
eoy_data = load_eoy()
gr_data = load_grether()

rng = np.random.default_rng(42)
N_KO = 600
ko_prior = np.clip(rng.beta(0.7, 0.7, N_KO), 0.02, 0.98)
ko_ai_pos = rng.random(N_KO) > 0.5
ko_posterior = np.where(
    ko_ai_pos,
    cr_post(ko_prior, 0.55, 0.73),
    cr_post(ko_prior, 0.55, 0.32),
) + rng.normal(0, 0.05, N_KO)
ko_posterior = np.clip(ko_posterior, 0.01, 0.99)


# ── Colors ───────────────────────────────────────────────────────────

C = dict(
    bayes="#888888", custom="#e67e22", grether="#2ca02c", agarwal="#e74c3c",
    cr="#ff7f0e", pos="#e74c3c", neg="#3498db", highlight="#9b59b6",
    orange_grp="darkorange", green_grp="seagreen", cd="#9b59b6",
)


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("QB Parameters")
st.sidebar.caption(
    "One set of (β₁, β₂) applied to every dataset below. "
    "Watch how parameters that fit one context fail in another."
)

PRESETS = {
    "Custom": (1.0, 1.0),
    "Bayesian (1.0, 1.0)": (1.0, 1.0),
    "Grether 1980 (0.88, 0.56)": (0.88, 0.56),
    "Agarwal et al. (0.26, 0.87)": (0.26, 0.87),
    "Complex Disclosure (~1.4, 1.0)": (1.4, 1.0),
}


def _apply_preset():
    p = st.session_state.preset
    if p != "Custom":
        st.session_state.b1 = PRESETS[p][0]
        st.session_state.b2 = PRESETS[p][1]


st.sidebar.selectbox("Presets", list(PRESETS.keys()), key="preset",
                      on_change=_apply_preset)
b1 = st.sidebar.slider("β₁ (signal weight)", 0.0, 2.5, 1.0, 0.05, key="b1")
b2 = st.sidebar.slider("β₂ (prior weight)", 0.0, 2.5, 1.0, 0.05, key="b2")
lr_overview = st.sidebar.slider("Signal LR (overview plot)", 0.5, 10.0, 3.0,
                                0.1, key="lr_ov")

st.sidebar.divider()
if abs(b1 - b2) < 0.06:
    st.sidebar.success(f"Coherent: β₁ ≈ β₂ ≈ {b1:.2f}")
elif b1 > b2:
    st.sidebar.warning(f"Base-rate neglect: β₁ ({b1:.2f}) > β₂ ({b2:.2f})")
else:
    st.sidebar.warning(f"Signal neglect: β₁ ({b1:.2f}) < β₂ ({b2:.2f})")

st.sidebar.divider()
st.sidebar.markdown(
    "**Try this:** select the Grether preset, scroll to the Grether "
    "section and see a good fit, then switch to Agarwal's preset and "
    "watch it break."
)


# ═════════════════════════════════════════════════════════════════════
# TITLE & INTRODUCTION
# ═════════════════════════════════════════════════════════════════════

st.title("One Equation, Many Biases")
st.markdown(
    "*The Quasi-Bayesian framework as a common language "
    "for belief updating across Econ 215H*"
)

st.markdown(r"""
Every paper in this course studies a version of the same problem: a person holds
a prior belief $\mu$, receives a signal with likelihood ratio $LR$, and forms a
posterior $\mu_R$. Bayes' rule provides the normative benchmark — in log-odds
form it says the posterior log-odds equal the prior log-odds plus the
log-likelihood ratio of the signal:

$$\text{logit}(\mu_R) \;=\; \log(LR) \;+\; \text{logit}(\mu)$$

This additive structure means the signal and prior contribute independently and
with **equal weight**. Grether (1980) proposed a parametric generalization —
the **Quasi-Bayesian (QB) model** — that nests Bayes as a special case:

$$\boxed{\;\text{logit}(\mu_R) \;=\; \beta_1 \cdot \log(LR) \;+\; \beta_2 \cdot \text{logit}(\mu)\;}$$

Here $\beta_1$ captures how much weight the agent places on **new information**
(the signal), while $\beta_2$ captures the weight on the **prior**. When
$\beta_1 = \beta_2 = 1$ this is Bayes. Deviations reveal specific biases:
**base-rate neglect** when $\beta_2 < \beta_1$ (people overweight signals
relative to priors), **conservatism** when both are below 1, and
**overconfidence** in own signals when $\beta_1 > 1$. The
$(\beta_1, \beta_2)$ plane becomes a map of all possible updating biases.

The central finding of this dashboard — and a unifying thread across the
course — is that $\beta_1$ and $\beta_2$ are **not fixed person-level
parameters**. They shift systematically with the **source** of information
(own experience vs. AI), the agent's **knowledge of the data-generating
process** (known urn composition vs. opaque algorithm), and the **decision
context** (neutral task vs. ego-relevant judgment). Grether's subjects
overweight signals relative to base rates. Agarwal et al.'s radiologists do
the opposite when the signal comes from AI. Kovach et al.'s bouncers, facing
unknown signal quality, abandon the log-odds structure entirely. This dashboard
traces those shifts across seven papers, using real experimental data where
available, to show how one equation can organize a diverse set of behavioral
findings.

Use the **sidebar sliders** to set $\beta_1$ and $\beta_2$ — the same QB model
is applied to every dataset below. Watch how parameters that fit one context
fail in another: **that mismatch is the finding.**
""")

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION I: THE MAP
# ═════════════════════════════════════════════════════════════════════

st.header("I. The (β₁, β₂) Map")

col_map, col_pp = st.columns(2)

with col_map:
    fig_map = go.Figure()

    fig_map.add_trace(go.Scatter(
        x=[0, 2.5], y=[0, 2.5], mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        name="β₁ = β₂ (coherent)",
    ))

    fig_map.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                      fillcolor="rgba(0,100,200,0.04)", line=dict(width=0))
    fig_map.add_shape(type="rect", x0=1, y0=1, x1=2.5, y1=2.5,
                      fillcolor="rgba(200,50,0,0.04)", line=dict(width=0))

    fig_map.add_annotation(x=1.95, y=0.45, text="Base-rate neglect<br>(β₂ < β₁)",
                           showarrow=False, font=dict(size=10, color="gray"))
    fig_map.add_annotation(x=0.35, y=2.05, text="Signal neglect<br>(β₁ < β₂)",
                           showarrow=False, font=dict(size=10, color="gray"))
    fig_map.add_annotation(x=0.35, y=0.25, text="Conservative",
                           showarrow=False, font=dict(size=9, color="silver"))
    fig_map.add_annotation(x=1.85, y=1.85, text="Overconfident",
                           showarrow=False, font=dict(size=9, color="silver"))

    papers = [
        ("Bayes", 1.0, 1.0, "star", "black", 16),
        ("Grether 1980", 0.88, 0.56, "circle", C["grether"], 14),
        ("Agarwal et al.", 0.26, 0.87, "diamond", C["agarwal"], 14),
        ("Complex Discl.", 1.4, 1.0, "square", C["cd"], 13),
    ]
    for name, px, py, sym, color, sz in papers:
        fig_map.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text", text=[name],
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=sz, color=color, symbol=sym,
                        line=dict(width=1, color="black")),
            showlegend=False,
        ))

    fig_map.add_trace(go.Scatter(
        x=[b1], y=[b2], mode="markers+text",
        text=[f"You ({b1:.2f}, {b2:.2f})"],
        textposition="bottom center", textfont=dict(size=10),
        marker=dict(size=16, color=C["custom"], symbol="x",
                    line=dict(width=2, color="black")),
        showlegend=False,
    ))

    fig_map.update_layout(
        xaxis_title="β₁ (signal weight)",
        yaxis_title="β₂ (prior weight)",
        xaxis=dict(range=[0, 2.5], dtick=0.5),
        yaxis=dict(range=[0, 2.5], dtick=0.5),
        height=470, margin=dict(t=10),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_pp:
    fig_pp = go.Figure()
    fig_pp.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color="gray", width=1), name="No update",
    ))
    fig_pp.add_trace(go.Scatter(
        x=mu, y=bayesian_post(mu, lr_overview), mode="lines",
        line=dict(color=C["bayes"], dash="dash", width=2), name="Bayesian",
    ))
    fig_pp.add_trace(go.Scatter(
        x=mu, y=qb_post(mu, lr_overview, b1, b2), mode="lines",
        line=dict(color=C["custom"], width=3),
        name=f"Your QB ({b1:.2f}, {b2:.2f})",
    ))
    fig_pp.add_trace(go.Scatter(
        x=mu, y=qb_post(mu, lr_overview, 0.88, 0.56), mode="lines",
        line=dict(color=C["grether"], width=2, dash="dashdot"),
        name="Grether (0.88, 0.56)",
    ))
    fig_pp.add_trace(go.Scatter(
        x=mu, y=qb_post(mu, lr_overview, 0.26, 0.87), mode="lines",
        line=dict(color=C["agarwal"], width=2, dash="dashdot"),
        name="Agarwal (0.26, 0.87)",
    ))
    fig_pp.update_layout(
        xaxis_title="Prior μ", yaxis_title="Posterior μ_R",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        height=470, margin=dict(t=10),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    )
    st.plotly_chart(fig_pp, use_container_width=True)

st.markdown(r"""
**Left:** Where each paper lands in $(\beta_1, \beta_2)$ space. The diagonal
is the *coherence line* ($\beta_1 = \beta_2$) — along it, QB reduces to an
LLO recalibration of the Bayesian posterior (Epping et al.). Points below the
diagonal exhibit base-rate neglect; above it, signal neglect. **Right:** The
prior-posterior curve at $LR =$ """ + f"{lr_overview:.1f}" + r""" for each
paper's estimated parameters. Grether's curve (green) and Agarwal's curve
(red) have nearly **opposite shapes** — same equation, opposite biases. Your
current slider position is shown in orange.
""")

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION II: GRETHER (1980)
# ═════════════════════════════════════════════════════════════════════

st.header("II. Grether (1980) — The Baseline")
st.subheader("Known DGP, own signals, no ego stakes")

st.markdown(r"""
Grether ran the canonical belief-updating experiment. Subjects observed six ball
draws from one of two cages — Cage A (probability $\frac{2}{3}$ of drawing an
N-ball) or Cage B (probability $\frac{1}{3}$) — and guessed which cage
generated the sample. The prior probability of Cage A was announced before each
round as $\frac{1}{3}$, $\frac{1}{2}$, or $\frac{2}{3}$. With a fully known
DGP (urn composition and prior), Bayes' rule gives the correct answer for every
possible sample, making deviations easy to measure.

The plot below uses aggregate data from El-Gamal & Grether (1995): 4,520
decisions across 257 subjects. Each **circle** is one experimental cell
(prior × sample composition), sized by the number of observations. The x-axis is
the Bayesian posterior — the normatively correct probability. The y-axis is the
fraction of subjects who chose Cage A. **Dashed curves** show the QB prediction
at your current sidebar $(\beta_1, \beta_2)$: for each prior, as the sample
evidence varies from all non-N to all N, the model traces out a curve. If the
data points lie on the curve, the model fits.
""")

col_gr, col_gm = st.columns([2.8, 1])

with col_gm:
    gr_qb_custom = qb_post(gr_data["prior_num"].values, gr_data["lr"].values,
                            b1, b2)
    gr_qb_grether = qb_post(gr_data["prior_num"].values, gr_data["lr"].values,
                             0.88, 0.56)
    w = gr_data["total"].values.astype(float)
    mse_bayes = np.average(
        (gr_data["frac_A"].values - gr_data["bayes_post"].values) ** 2,
        weights=w)
    mse_grether = np.average(
        (gr_data["frac_A"].values - gr_qb_grether) ** 2, weights=w)
    mse_custom = np.average(
        (gr_data["frac_A"].values - gr_qb_custom) ** 2, weights=w)

    st.markdown("**Weighted MSE**")
    st.metric("Bayesian", f"{mse_bayes:.4f}")
    st.metric("Grether (0.88, 0.56)", f"{mse_grether:.4f}")
    st.metric(f"You ({b1:.2f}, {b2:.2f})", f"{mse_custom:.4f}")
    if mse_custom <= mse_grether * 1.01:
        st.success("Good fit!")
    elif mse_custom < mse_bayes:
        st.info("Better than Bayes")

with col_gr:
    fig_gr = go.Figure()
    fig_gr.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Perfect Bayesian",
    ))

    prior_colors = {1 / 3: C["neg"], 1 / 2: C["bayes"], 2 / 3: C["pos"]}
    prior_labels = {1 / 3: "Prior = 1/3", 1 / 2: "Prior = 1/2",
                    2 / 3: "Prior = 2/3"}

    lr_smooth = 2.0 ** np.linspace(-6, 6, 300)

    for pv in [1 / 3, 1 / 2, 2 / 3]:
        sub = gr_data[gr_data["prior_num"] == pv]
        color = prior_colors[pv]

        fig_gr.add_trace(go.Scatter(
            x=sub["bayes_post"], y=sub["frac_A"], mode="markers",
            marker=dict(size=np.clip(sub["total"] / 20, 6, 30), color=color,
                        opacity=0.85, line=dict(width=1, color="black")),
            name=f"{prior_labels[pv]} (n={sub['total'].sum():,})",
            text=[f"N={n}, n={t}" for n, t in
                  zip(sub["num_N"], sub["total"])],
            hoverinfo="text+x+y",
        ))

        bp_smooth = bayesian_post(pv, lr_smooth)
        qb_smooth = qb_post(pv, lr_smooth, b1, b2)
        fig_gr.add_trace(go.Scatter(
            x=bp_smooth, y=qb_smooth, mode="lines",
            line=dict(color=color, width=2.5, dash="dash"),
            showlegend=False,
        ))

    fig_gr.update_layout(
        xaxis_title="Bayesian P(Cage A)",
        yaxis_title="Fraction choosing A (circles) / QB prediction (curves)",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        height=480, margin=dict(t=10),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    )
    st.plotly_chart(fig_gr, use_container_width=True)

st.markdown(r"""
At Grether's estimated $(\beta_1, \beta_2) = (0.88, 0.56)$, the model captures
two features of the data: general **conservatism** (points pulled toward 0.5
relative to Bayes) and **base-rate neglect** ($\beta_2 = 0.56 < \beta_1 = 0.88$
— the prior has less effect than it should). The three prior-specific curves
separate because different priors enter the QB equation differently — this
separation is visible in the data as well: the blue circles (prior = 1/3) sit
below the red circles (prior = 2/3) at the same Bayesian posterior, exactly as
the model predicts.

This is the best-case scenario for belief updating: a fully known DGP, "own"
signals (direct observation), and no ego stakes. Even here, updating is
imperfect. The question is how it gets *worse* — and in which direction — as
these conditions change.
""")

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION III: AGARWAL ET AL. (2023)
# ═════════════════════════════════════════════════════════════════════

st.header("III. Agarwal et al. (2023) — The Mirror Image")
st.subheader("AI signal, partial DGP knowledge, professional ego stakes")

st.markdown(r"""
Agarwal et al. study 336 radiologists diagnosing chest X-rays, half the time
with AI assistance and half without. By comparing each radiologist's assessment
of the **same patient-pathology** pair under both conditions, we construct
within-subject prior-posterior pairs: the **prior** is the probability
assessment without AI, and the **posterior** is the assessment with AI.

Their logit model estimates $b = 0.26$ (weight on AI signal) and $d = 0.87$
(weight on own prior). In QB terms: $(\beta_1, \beta_2) = (0.26, 0.87)$ — deep
in the **signal neglect** region. When the signal comes from an AI algorithm
rather than own experience, radiologists massively underweight it while
preserving most of their prior weight. Same equation as Grether, **opposite
region** in parameter space.
""")

AG_TPR, AG_TNR = 0.85, 0.85
ag_lr_pos = AG_TPR / (1 - AG_TNR)
ag_lr_neg = (1 - AG_TPR) / AG_TNR

ag_pos = ag_data[ag_data["ai_positive"] == True]
ag_neg = ag_data[ag_data["ai_positive"] == False]

fig_ag = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"AI Positive (n = {len(ag_pos):,})",
        f"AI Negative (n = {len(ag_neg):,})",
    ),
)

for ci, (subset, lr_val) in enumerate(
    [(ag_pos, ag_lr_pos), (ag_neg, ag_lr_neg)], 1
):
    n_show = min(len(subset), 2000)
    idx = rng.choice(len(subset), n_show, replace=False)
    d_color = C["pos"] if ci == 1 else C["neg"]

    fig_ag.add_trace(go.Scatter(
        x=subset["prior"].values[idx], y=subset["posterior"].values[idx],
        mode="markers",
        marker=dict(size=3, color=d_color, opacity=0.15),
        showlegend=False,
    ), row=1, col=ci)

    fig_ag.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color="gray", width=1), showlegend=False,
    ), row=1, col=ci)

    fig_ag.add_trace(go.Scatter(
        x=mu, y=bayesian_post(mu, lr_val), mode="lines",
        line=dict(color=C["bayes"], dash="dash", width=1.5),
        showlegend=(ci == 1), name="Bayesian",
    ), row=1, col=ci)

    fig_ag.add_trace(go.Scatter(
        x=mu, y=qb_post(mu, lr_val, b1, b2), mode="lines",
        line=dict(color=C["custom"], width=3),
        showlegend=(ci == 1), name=f"Your QB ({b1:.2f}, {b2:.2f})",
    ), row=1, col=ci)

    fig_ag.add_trace(go.Scatter(
        x=mu, y=qb_post(mu, lr_val, 0.26, 0.87), mode="lines",
        line=dict(color=C["agarwal"], width=2, dash="dashdot"),
        showlegend=(ci == 1), name="Agarwal (0.26, 0.87)",
    ), row=1, col=ci)

    fig_ag.update_xaxes(title_text="Prior (without AI)", range=[0, 1],
                        row=1, col=ci)
    fig_ag.update_yaxes(title_text="Posterior (with AI)", range=[0, 1],
                        row=1, col=ci)

fig_ag.update_layout(height=480, legend=dict(orientation="h", y=-0.13))
st.plotly_chart(fig_ag, use_container_width=True)

st.markdown(r"""
The Bayesian curve (gray dashed) predicts dramatic updating — radiologists
should shift their assessments substantially when the AI flags a finding. In
reality, the scatter barely moves from the diagonal: most radiologists largely
ignore the AI signal. Agarwal's fit (red, $\beta_1 = 0.26$) captures this
**automation neglect**.

Try switching between the **Grether** and **Agarwal** presets in the sidebar.
Grether's base-rate-neglecting parameters predict huge posterior swings that
don't exist in this data. Agarwal's signal-neglecting parameters produce a
near-flat curve that misses the Grether data entirely. **No single
$(\beta_1, \beta_2)$ fits both datasets** — the weights are
context-dependent, and the context that changes is the **source** of the signal.
Own experience with urns produces $\beta_1 > \beta_2$; an AI recommendation
produces $\beta_1 \ll \beta_2$.
""")

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION IV: EOY (2023)
# ═════════════════════════════════════════════════════════════════════

st.header("IV. Enke, Oprea & Yang (2023) — When the Prior Itself Is Biased")
st.subheader("Group stereotypes contaminate the input to the updating process")

st.markdown(r"""
EOY study how people estimate a continuous value when they know the **group** an
observation belongs to. Their Representativeness-based Stereotype Deviation
(RSD) model predicts:

$$\hat{t} \;=\; \Delta_g \;+\; \omega \, \mu_g \;+\; (1 - \omega) \, t$$

where $t$ is the true value, $\mu_g$ is the group mean, $\omega$ is the weight
on the group prior, and $\Delta_g = \gamma^s \cdot \omega \cdot (\mu_g -
\mu_{-g})$ is a **representativeness bias** that inflates the prior's influence
for stereotypically representative groups. This is structurally parallel to QB:
$\omega$ plays the role of $\beta_2$ (prior weight), $1 - \omega$ plays the
role of $\beta_1$ (signal weight), and $\Delta_g$ adds something QB lacks — an
**intercept shift** from categorical thinking.

The direction matches Agarwal: people overweight the categorical prior (group
identity) relative to individual-level evidence. But the mechanism is
representativeness, not automation neglect — and the prior itself is
contaminated, not just underweighted.
""")

col_eoy, col_eoy_ctrl = st.columns([2.8, 1])

with col_eoy_ctrl:
    eoy_treat = st.selectbox(
        "Treatment",
        ["Baseline", "SignalFirst", "OneGroup", "NoGroup"],
        key="eoy_t",
    )
    st.divider()
    rsd_gamma = st.slider("γˢ (representativeness)", 0.0, 1.0, 0.45, 0.01,
                           key="rsd_g")
    rsd_omega = st.slider("ω (prior weight)", 0.0, 1.0, 0.50, 0.01,
                           key="rsd_w")
    st.caption(
        "In the NoGroup treatment, subjects don't observe group labels — "
        "set γˢ = 0 to match."
    )

with col_eoy:
    eoy_sub = eoy_data[eoy_data["treatmentname"] == eoy_treat].copy()
    eoy_hi = eoy_sub[eoy_sub["highgroup"] == 1.0]
    eoy_lo = eoy_sub[eoy_sub["highgroup"] == 0.0]

    mu_hi = eoy_hi["mean_theory"].median() if len(eoy_hi) > 0 else 60
    mu_lo = eoy_lo["mean_theory"].median() if len(eoy_lo) > 0 else 40

    fig_eoy = go.Figure()

    for grp, color, name in [
        (eoy_hi, C["orange_grp"], "Orange (high-mean group)"),
        (eoy_lo, C["green_grp"], "Green (low-mean group)"),
    ]:
        if len(grp) > 0:
            idx_e = rng.choice(len(grp), min(2000, len(grp)), replace=False)
            fig_eoy.add_trace(go.Scatter(
                x=grp["value"].values[idx_e],
                y=grp["guess"].values[idx_e],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.12),
                name=name,
            ))

    t_range = np.linspace(15, 85, 200)
    delta_hi = rsd_gamma * rsd_omega * (mu_hi - mu_lo)
    delta_lo = rsd_gamma * rsd_omega * (mu_lo - mu_hi)
    est_hi = delta_hi + rsd_omega * mu_hi + (1 - rsd_omega) * t_range
    est_lo = delta_lo + rsd_omega * mu_lo + (1 - rsd_omega) * t_range

    fig_eoy.add_trace(go.Scatter(
        x=t_range, y=est_hi, mode="lines",
        line=dict(color=C["orange_grp"], width=3),
        name=f"RSD Orange (Δ = {delta_hi:+.1f})",
    ))
    fig_eoy.add_trace(go.Scatter(
        x=t_range, y=est_lo, mode="lines",
        line=dict(color=C["green_grp"], width=3),
        name=f"RSD Green (Δ = {delta_lo:+.1f})",
    ))
    fig_eoy.add_trace(go.Scatter(
        x=[15, 85], y=[15, 85], mode="lines",
        line=dict(dash="dot", color="gray", width=1), name="Perfect",
    ))

    fig_eoy.update_layout(
        xaxis_title="True value",
        yaxis_title="Subject's guess",
        height=480, margin=dict(t=10),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    )
    st.plotly_chart(fig_eoy, use_container_width=True)

st.markdown(r"""
The two groups produce **parallel lines** — each pulled toward its respective
group mean, with a gap driven by the representativeness bias $\Delta_g$. In the
QB analogy, the "signal weight" ($1 - \omega$) and "prior weight" ($\omega$) may
be roughly stable, but the **prior itself is contaminated** by categorical
thinking: subjects act as if the group mean is more extreme than it actually is.

This is a failure mode the QB equation does not capture. Grether and Agarwal
both assume the prior $\mu$ entering the equation is accurate — the bias is in
the *weights*. EOY shows the bias can instead be in the *input*. Try the
**NoGroup** treatment: without group labels, the two lines merge, confirming
that $\Delta_g$ comes from categorical thinking, not from the signal.
""")

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION V: KOVACH ET AL. (2026)
# ═════════════════════════════════════════════════════════════════════

st.header("V. Kovach et al. (2026) — When the Framework Breaks Down")
st.subheader("Unknown DGP, opaque AI, linear heuristics")

st.markdown(r"""
Kovach et al. study bouncers estimating customer ages with AI assistance — but
subjects **don't know the AI's accuracy**. Without a known DGP, the likelihood
ratio $LR$ that anchors the QB model is undefined. Their **Contraction Rule**
(CR) proposes a fundamentally different functional form:

$$\mu_R \;=\; \varepsilon \cdot \mu \;+\; (1 - \varepsilon) \cdot \rho_R$$

This is **linear** in the prior, unlike QB's S-curve in log-odds. When the
likelihood ratio is unobservable, the log-odds machinery of QB becomes
inapplicable — people fall back to a simpler heuristic: contract toward a
signal-specific attractor $\rho_R$. Data below is **simulated** from reported
CR parameters (paper in draft, no public microdata).
""")

col_ko, col_ko_ctrl = st.columns([2.8, 1])

with col_ko_ctrl:
    ko_eps = st.slider("ε (slope)", 0.0, 1.0, 0.55, 0.01, key="ko_e")
    ko_rho_p = st.slider("ρ⁺ (pos attractor)", 0.0, 1.0, 0.73, 0.01,
                          key="ko_rp")
    ko_rho_n = st.slider("ρ⁻ (neg attractor)", 0.0, 1.0, 0.32, 0.01,
                          key="ko_rn")
    st.divider()
    st.caption("QB uses LR ≈ 2.77 from simulated AI accuracy.")

KO_LR_POS = 0.83 / 0.30
KO_LR_NEG = 0.17 / 0.70

fig_ko = make_subplots(
    rows=1, cols=2,
    subplot_titles=("AI: Over 21 (simulated)", "AI: Under 21 (simulated)"),
)

for ci, (mask, rho_v, lr_v) in enumerate([
    (ko_ai_pos, ko_rho_p, KO_LR_POS),
    (~ko_ai_pos, ko_rho_n, KO_LR_NEG),
], 1):
    dp, dq = ko_prior[mask], ko_posterior[mask]

    fig_ko.add_trace(go.Scatter(
        x=dp, y=dq, mode="markers",
        marker=dict(size=3, color=C["cr"], opacity=0.25),
        showlegend=False,
    ), row=1, col=ci)

    fig_ko.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color="gray", width=1), showlegend=False,
    ), row=1, col=ci)

    fig_ko.add_trace(go.Scatter(
        x=mu, y=cr_post(mu, ko_eps, rho_v), mode="lines",
        line=dict(color=C["cr"], width=3),
        showlegend=(ci == 1), name="Contraction Rule",
    ), row=1, col=ci)

    fig_ko.add_trace(go.Scatter(
        x=mu, y=qb_post(mu, lr_v, b1, b2), mode="lines",
        line=dict(color=C["custom"], width=2.5, dash="dash"),
        showlegend=(ci == 1), name=f"QB ({b1:.2f}, {b2:.2f})",
    ), row=1, col=ci)

    fig_ko.update_xaxes(title_text="Prior μ", range=[0, 1], row=1, col=ci)
    fig_ko.update_yaxes(title_text="Posterior μ_R", range=[0, 1], row=1,
                        col=ci)

fig_ko.update_layout(height=430, legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig_ko, use_container_width=True)

st.markdown(r"""
The linear CR pattern is clear — and the QB curve (dashed orange) **cannot
reproduce it** regardless of $(\beta_1, \beta_2)$. This is not a parameter
mismatch — it is a **functional form** mismatch. QB's S-curve in log-odds space
always produces a sigmoid in probability space; CR's linearity is a
qualitatively different shape. The implication is that the QB framework has a
**domain of applicability**: it works when agents know (or can estimate) the
likelihood ratio, and breaks down when they cannot.

Kovach's INFO vs. NOINFO treatment tests this boundary directly: even when told
the AI's accuracy (which should supply the LR), subjects still update roughly
linearly. This suggests heuristic updating may not just be about missing
information — it may be a **default mode** that parametric structure only
partially overrides.
""")

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION VI: BOUNDARIES OF THE FRAMEWORK
# ═════════════════════════════════════════════════════════════════════

st.header("VI. Three Qualifications")

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    st.subheader("LLO Recalibration")
    st.markdown(r"""
    **Epping et al. (2026)** show that reported beliefs are distorted:
    $\text{logit}(f(p)) = \alpha \cdot \text{logit}(p) + \beta$. If priors
    entering the QB equation are miscalibrated, apparent $\beta_2$ shifts
    could partly be **reporting artifacts**. Forcing coherence
    ($\beta_1 = \beta_2$) corresponds to LLO with $\beta = 0$ — internally
    consistent updating, even if miscalibrated. Agarwal's incoherence
    ($\beta_1 \neq \beta_2$) resists this interpretation: the asymmetry
    between AI and own-info weights is too large for calibration to explain.
    """)

with col_b2:
    st.subheader("Ego Utility")
    st.markdown(r"""
    **Koszegi (2006)** provides a **motivational** foundation for signal
    neglect. If ego utility rewards believing good things about oneself,
    $\beta_1$ shrinks for **ego-threatening** signals — bad news about own
    ability. This predicts *asymmetric* $\beta_1$: small for negative
    signals, normal for positive. It may explain why Agarwal's radiologists
    underweight AI (it implicitly challenges their diagnostic expertise)
    while Grether's subjects don't (urns carry no ego stakes).
    """)

with col_b3:
    st.subheader("Complex Disclosure")
    st.markdown(r"""
    **Jin, Luca & Martin (2022)** study what happens when signal structure
    is **complex**. Subjects may overweight their own reading of a complex
    disclosure ($\beta_1 > 1$), landing in the **overconfident** region —
    the only paper in the course where the primary bias runs this direction.
    This suggests **signal complexity** can flip the sign of bias: simple,
    external signals are underweighted (Agarwal), but complex, self-
    interpreted signals are overweighted.
    """)

st.divider()

# ═════════════════════════════════════════════════════════════════════
# SECTION VII: SYNTHESIS
# ═════════════════════════════════════════════════════════════════════

st.header("VII. Synthesis: Context-Dependent Updating")

st.markdown(r"""
Looking across the $(\beta_1, \beta_2)$ map, a pattern emerges. The same
cognitive architecture produces **opposite biases** in different environments:

| Context | $\beta_1$ | $\beta_2$ | Bias direction | Paper |
|---------|-----------|-----------|----------------|-------|
| Known DGP, own signal | 0.88 | 0.56 | Base-rate neglect | Grether 1980 |
| AI signal, partial DGP | 0.26 | 0.87 | Signal neglect | Agarwal et al. |
| Complex own-signal | ~1.4 | ~1.0 | Overconfidence | Jin, Luca & Martin |
| Unknown DGP | — | — | QB breaks down → linear | Kovach et al. |
| Group stereotypes | — | — | Prior contaminated by Δ | Enke, Oprea & Yang |

The weight people place on new information is not a stable cognitive parameter.
It depends on whether the information feels **familiar** (own experience vs. AI),
**interpretable** (known vs. unknown DGP), and **ego-relevant** (neutral urn
draws vs. professional competence). Grether's urn experiment — all signals are
"own" experience, the DGP is fully known, ego is uninvolved — is the best case
for rational updating, and even there, base-rate neglect persists. Move any of
those conditions, and the bias shifts: toward signal neglect (Agarwal), toward
overconfidence (Jin, Luca & Martin), toward linear heuristics (Kovach), or
toward contaminated priors (EOY).

For the design of **human-AI systems**, this has immediate practical
implications. Increasing AI accuracy will not help if $\beta_1 \approx 0.26$ —
the signal is discounted before it can influence the decision. Providing DGP
statistics may not restore parametric updating, as Kovach's INFO treatment
shows. Recalibration of reported beliefs (Epping et al.) fixes only the
**reporting** distortion, not the weight distortion. And Koszegi's ego utility
warns that some neglect is **motivated** — interventions that merely make
information more salient will fail if the agent doesn't *want* to update.

The QB equation does not solve these problems, but it provides a **common
diagnostic language**: estimate $(\beta_1, \beta_2)$ in each new environment,
locate it on the map, and design interventions accordingly. The single most
important lesson from this map is that the intervention that works in one
region (e.g., making signals more prominent to fight conservatism) can backfire
in another (e.g., making AI more salient when the problem is ego-driven
avoidance). The parameters must be re-estimated — or the functional form
reconsidered — for each decision context.
""")


# ── Footer ───────────────────────────────────────────────────────────

st.divider()
st.caption(
    "**Data:** Agarwal et al. (OSF/Collab-CXR, CC-BY 4.0 — 10,500 "
    "within-subject pairs from 336 radiologists); Enke, Oprea & Yang "
    "(Harvard Dataverse/QJE — 18,075 obs, 241 subjects); Grether "
    "(El-Gamal & Grether 1995 JASA — 4,520 aggregate decisions, 257 "
    "subjects). Kovach et al. simulated from reported parameters (paper "
    "in draft). | **Papers:** Grether (1980), Agarwal et al. (2023), "
    "Kovach et al. (2026), Enke, Oprea & Yang (2023), Jin, Luca & Martin "
    "(2022), Epping et al. (2026), Koszegi (2006)."
)
