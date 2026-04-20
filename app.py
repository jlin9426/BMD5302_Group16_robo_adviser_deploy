"""
BMD5302 — Robo Adviser Platform (Part 3)
Single-file Streamlit application.

Run with:   streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data import (
    FUNDS, FUND_NAMES, COV_MATRIX, MEAN_RETURNS, GMVP,
    RTI_WEIGHTS, A_MIN, A_MAX, DIMENSION_TITLES,
    compute_rti_and_A, optimize_portfolio, risk_label, risk_color, risk_description,
    EF_POINTS, QUESTIONS, TRADING_DAYS,
    annualize_return, annualize_std,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Robo Adviser | BMD5302",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CUSTOM STYLES (simpler, more robust rendering)
# =============================================================================
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* Metric cards (native-looking, no fancy CSS tricks) */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-sub { font-size: 0.82rem; color: #64748B; margin-top: 0.2rem; }

    .profile-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        color: white;
    }

    .info-box {
        background: #F0F9FF;
        border-left: 3px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #334155;
        font-size: 0.9rem;
        margin: 1rem 0;
    }

    .section-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background: #EFF6FF;
        color: #1E40AF;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }

    .question-text {
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.3rem;
        margin-top: 0.8rem;
    }
    .question-id { color: #3B82F6; font-weight: 700; margin-right: 0.4rem; }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }

    hr { border: none; border-top: 1px solid #E2E8F0; margin: 1.2rem 0; }

    .formula-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'Inter', 'SF Mono', Menlo, monospace;
        font-size: 0.92rem;
        line-height: 1.8;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    if "page"    not in st.session_state: st.session_state.page = "welcome"
    if "answers" not in st.session_state: st.session_state.answers = {}

init_state()

def goto(page: str):
    st.session_state.page = page
    st.rerun()


# =============================================================================
# PAGE 1 — WELCOME
# =============================================================================
def render_welcome():
    # Robust title — plain Streamlit widgets, no custom HTML gymnastics
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align:center; font-size:3rem; margin-bottom:0.3rem; color:#1E3A8A;'>Robo Adviser</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center; color:#64748B; font-size:1.1rem; line-height:1.6; margin-bottom:2rem;'>"
            "A personalized portfolio recommendation engine powered by Modern Portfolio Theory<br>"
            "and a 20-question risk assessment across four behavioral and financial dimensions."
            "</p>", unsafe_allow_html=True)

    # How it works — 3 steps in columns
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [
        ("1️⃣ Risk Questionnaire",
         "Answer 20 questions covering investment horizon, financial capacity, behavioral tolerance, and market knowledge."),
        ("2️⃣ Weighted RTI → A",
         "Dimension scores are normalized and combined into a Risk Tolerance Index, then mapped linearly to a risk aversion coefficient A."),
        ("3️⃣ Optimal Portfolio",
         "Across 10 FSMOne funds, we solve the mean-variance optimization under long-only constraint and return the utility-maximizing allocation."),
    ]
    for col, (title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("<br>", unsafe_allow_html=True)

    # Universe facts
    st.markdown("### Fund Universe & Methodology")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Fund Universe</div>'
                    '<div class="metric-value">10</div><div class="metric-sub">FSMOne funds</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">Sample Period</div>'
                    '<div class="metric-value">~1Y</div><div class="metric-sub">234 trading days</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">A Range</div>'
                    f'<div class="metric-value">[{A_MIN:.0f}, {A_MAX:.0f}]</div><div class="metric-sub">Risk aversion bounds</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Framework</div>'
                    '<div class="metric-value">MPT</div><div class="metric-sub">Markowitz 1952</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Start Questionnaire →", type="primary", width="stretch"):
            goto("questionnaire")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;color:#94A3B8;font-size:0.82rem;">'
        'BMD5302 Financial Modeling · AY2025/26 Sem 2 · Group Project'
        '</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — QUESTIONNAIRE
# =============================================================================
def render_questionnaire():
    st.markdown("## Risk Assessment Questionnaire")
    answered = len(st.session_state.answers)
    total = len(QUESTIONS)
    st.progress(answered / total, text=f"{answered} / {total} answered")

    st.markdown(
        '<div class="info-box">'
        'Each response is scored on a 1–5 scale, where higher scores indicate '
        'greater willingness or capacity to bear investment risk. Your scores are '
        'aggregated into four dimension sub-indices, combined into a weighted '
        'Risk Tolerance Index (RTI*), and mapped linearly to a risk aversion coefficient A.'
        '</div>', unsafe_allow_html=True)

    # Group by section
    sections = {}
    for q in QUESTIONS:
        sections.setdefault((q["section"], q["section_title"]), []).append(q)

    tab_labels = [f"{sec_id}. {sec_title}" for (sec_id, sec_title) in sections.keys()]
    tabs = st.tabs(tab_labels)

    for tab, ((sec_id, sec_title), qs) in zip(tabs, sections.items()):
        with tab:
            st.markdown(f'<div class="section-badge">Section {sec_id} · {sec_title}</div>',
                        unsafe_allow_html=True)
            for q in qs:
                st.markdown(
                    f'<div class="question-text"><span class="question-id">{q["id"]}.</span>{q["text"]}</div>',
                    unsafe_allow_html=True)
                opt_labels = [f"{label} ({score})" for (label, score) in q["options"]]
                current = st.session_state.answers.get(q["id"])
                index = None
                if current is not None:
                    for i, (_, score) in enumerate(q["options"]):
                        if score == current:
                            index = i; break
                choice = st.radio(
                    "Select one", opt_labels, key=f"radio_{q['id']}",
                    index=index, label_visibility="collapsed")
                if choice is not None:
                    chosen_score = q["options"][opt_labels.index(choice)][1]
                    st.session_state.answers[q["id"]] = chosen_score
                st.markdown("<hr>", unsafe_allow_html=True)

    # Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Back", width="stretch"):
            goto("welcome")
    with c3:
        complete = len(st.session_state.answers) == total
        if st.button("Submit & See Results →", type="primary",
                     width="stretch", disabled=not complete):
            goto("results")
        if not complete:
            st.caption(f"{total - answered} question(s) remaining.")


# =============================================================================
# PAGE 3 — RESULTS
# =============================================================================
def render_results():
    # 1. Compute RTI → A → optimal portfolio
    rti = compute_rti_and_A(st.session_state.answers)
    port = optimize_portfolio(rti["A"])

    # ---- Header + headline metrics ----
    st.markdown("## Your Personalized Portfolio Recommendation")

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Total Score</div>
                <div class="metric-value">{rti["total_score"]}<span style="font-size:1rem;color:#64748B;"> / 100</span></div>
                <div class="metric-sub">20 questions × 1–5 scale</div>
            </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Weighted RTI*</div>
                <div class="metric-value">{rti["RTI"]:.3f}</div>
                <div class="metric-sub">Normalized [0, 1]</div>
            </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Risk Aversion A</div>
                <div class="metric-value">{rti["A"]:.2f}</div>
                <div class="metric-sub">Range: [{A_MIN:.0f}, {A_MAX:.0f}]</div>
            </div>''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Risk Profile</div>
                <div style="margin-top:0.5rem;">
                    <span class="profile-badge" style="background:{rti["color"]};">{rti["label"]}</span>
                </div>
                <div class="metric-sub" style="margin-top:0.6rem;">Tier based on A bands</div>
            </div>''', unsafe_allow_html=True)

    # Profile description
    st.markdown(
        f'<div class="info-box" style="border-left-color:{rti["color"]};">'
        f'<strong>{rti["label"]} investor.</strong> {rti["description"]}'
        f'</div>', unsafe_allow_html=True)

    # ---- Risk Assessment Breakdown ----
    st.markdown("### Risk Assessment Breakdown")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        _render_dimension_chart(rti)
    with c2:
        st.markdown(f'''
            <div class="formula-box">
                <strong>Weighted RTI calculation</strong><br><br>
                H ({RTI_WEIGHTS['H']:.0%}) = {rti['H']:.3f}<br>
                F ({RTI_WEIGHTS['F']:.0%}) = {rti['F']:.3f}<br>
                B ({RTI_WEIGHTS['B']:.0%}) = {rti['B']:.3f}<br>
                K ({RTI_WEIGHTS['K']:.0%}) = {rti['K']:.3f}<br>
                <hr style="margin:0.5rem 0;">
                RTI* = {RTI_WEIGHTS['H']}·H + {RTI_WEIGHTS['F']}·F + {RTI_WEIGHTS['B']}·B + {RTI_WEIGHTS['K']}·K<br>
                <strong style="color:#1E3A8A;">RTI* = {rti['RTI']:.3f}</strong><br><br>
                A = A_max − RTI*·(A_max − A_min)<br>
                A = {A_MAX:.0f} − {rti['RTI']:.3f}·{A_MAX - A_MIN:.0f}<br>
                <strong style="color:#1E3A8A;">A = {rti['A']:.2f}</strong>
            </div>
        ''', unsafe_allow_html=True)

    # ---- Expected performance ----
    st.markdown("### Expected Performance")
    c1, c2, c3 = st.columns(3, gap="small")
    ann_r = annualize_return(port["return"]) * 100
    ann_s = annualize_std(port["std"]) * 100
    sharpe = ann_r / ann_s if ann_s > 0 else 0
    with c1:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Expected Return (Annualized)</div>
                <div class="metric-value">{ann_r:+.2f}%</div>
                <div class="metric-sub">Daily: {port["return"]*100:+.4f}%</div>
            </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Volatility (Annualized)</div>
                <div class="metric-value">{ann_s:.2f}%</div>
                <div class="metric-sub">Daily σ: {port["std"]*100:.4f}%</div>
            </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Utility Score</div>
                <div class="metric-value">{port["utility"]*10000:+.3f}<span style="font-size:0.9rem;color:#64748B;"> × 10⁻⁴</span></div>
                <div class="metric-sub">U = r − (A/2)·σ²</div>
            </div>''', unsafe_allow_html=True)

    # ---- Portfolio composition ----
    st.markdown("### Recommended Portfolio Composition")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        _render_weights_chart(port["weights"])
    with c2:
        _render_weights_table(port["weights"])

    # ---- Efficient Frontier ----
    st.markdown("### Efficient Frontier")
    st.caption("Your optimal portfolio plotted against the Efficient Frontier (long-only). "
               "Individual funds and the Global Minimum Variance Portfolio (GMVP) are also shown.")
    _render_ef_chart(port, rti)

    # ---- Methodology note ----
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("📌 Methodology note — why the portfolio concentrates in fixed income"):
        st.markdown("""
        Across the 10-fund universe over this sample period, only **US Income** and **EU Income**
        exhibited positive historical mean returns. Under the long-only constraint and for any
        A > 0, the mean-variance optimizer allocates exclusively to these two funds, shifting the
        mix between them as risk aversion changes:

        - **Higher A** (conservative) → heavier weight on the lower-volatility fund (US Income)
        - **Lower A** (aggressive) → shift toward the higher-return fund (EU Income)

        This result is mathematically correct but reflects the idiosyncrasies of the sample period.
        In production systems, this would be augmented with forward-looking capital market
        assumptions or Black-Litterman-style priors rather than relying solely on historical means.
        """)

    # ---- Retake ----
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Retake Questionnaire", width="stretch"):
            st.session_state.answers = {}
            goto("welcome")


# =============================================================================
# PLOT HELPERS
# =============================================================================
def _render_dimension_chart(rti: dict):
    """Horizontal bar chart of H/F/B/K sub-indices."""
    dims = [
        ("H · Horizon",    rti["H"], RTI_WEIGHTS["H"], "#3B82F6"),
        ("F · Financial",  rti["F"], RTI_WEIGHTS["F"], "#10B981"),
        ("B · Behavioral", rti["B"], RTI_WEIGHTS["B"], "#F59E0B"),
        ("K · Knowledge",  rti["K"], RTI_WEIGHTS["K"], "#A855F7"),
    ]
    labels  = [d[0] + f"  (w={d[2]:.0%})" for d in dims]
    values  = [d[1] for d in dims]
    colors  = [d[3] for d in dims]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}<br>Sub-index: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=30, t=10, b=30),
        xaxis=dict(title="Sub-index [0, 1]", range=[0, 1.1],
                   gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_weights_chart(weights):
    """Horizontal bar chart of portfolio weights (long-only, non-negative)."""
    df = pd.DataFrame({
        "Fund":       FUND_NAMES,
        "Weight":     [w * 100 for w in weights],
        "AssetClass": [f["asset_class"] for f in FUNDS],
    }).sort_values("Weight", ascending=True)

    colors = []
    for ac in df["AssetClass"]:
        if   ac == "Equity":       colors.append("#3B82F6")
        elif ac == "Fixed Income": colors.append("#10B981")
        elif ac == "Multi-Asset":  colors.append("#A855F7")
        else:                      colors.append("#94A3B8")

    fig = go.Figure(go.Bar(
        x=df["Weight"], y=df["Fund"], orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"{w:.1f}%" for w in df["Weight"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=420, margin=dict(l=0, r=40, t=10, b=30),
        xaxis=dict(title="Weight (%)", range=[0, max(df["Weight"].max() * 1.15, 5)],
                   gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_weights_table(weights):
    rows = []
    for fund, w in zip(FUNDS, weights):
        if w > 1e-6:
            rows.append({
                "Fund":   fund["short"],
                "Class":  fund["asset_class"],
                "Weight": f"{w*100:.2f}%",
                "_raw":   w,
            })
    if not rows:
        st.info("Portfolio is empty (all weights zero).")
        return
    rows.sort(key=lambda r: -r["_raw"])
    df = pd.DataFrame(rows).drop(columns=["_raw"])
    st.dataframe(df, width="stretch", hide_index=True, height=420)


def _render_ef_chart(port: dict, rti: dict):
    """Efficient frontier with funds, GMVP, and user's optimum."""
    fig = go.Figure()

    # Individual funds
    fig.add_trace(go.Scatter(
        x=[annualize_std(f["std"]) * 100 for f in FUNDS],
        y=[annualize_return(f["mean"]) * 100 for f in FUNDS],
        mode="markers+text",
        marker=dict(size=9, color="#94A3B8", line=dict(width=1, color="white")),
        text=FUND_NAMES, textposition="top center",
        textfont=dict(size=10, color="#475569"),
        name="Individual Funds",
        hovertemplate="<b>%{text}</b><br>Return: %{y:.2f}%<br>Vol: %{x:.2f}%<extra></extra>",
    ))

    # EF curve
    if EF_POINTS:
        ef_std = [annualize_std(p["std"]) * 100 for p in EF_POINTS]
        ef_ret = [annualize_return(p["return"]) * 100 for p in EF_POINTS]
        fig.add_trace(go.Scatter(
            x=ef_std, y=ef_ret, mode="lines",
            line=dict(color="#1E3A8A", width=2.5),
            name="Efficient Frontier (long-only)",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

    # GMVP
    fig.add_trace(go.Scatter(
        x=[annualize_std(GMVP["std"]) * 100],
        y=[annualize_return(GMVP["return"]) * 100],
        mode="markers+text",
        marker=dict(size=16, color="#10B981", symbol="diamond",
                    line=dict(width=2, color="white")),
        text=["GMVP"], textposition="bottom center",
        textfont=dict(size=11, color="#10B981", family="Inter"),
        name="Global Min Variance",
        hovertemplate="<b>GMVP</b><br>Return: %{y:.2f}%<br>Vol: %{x:.2f}%<extra></extra>",
    ))

    # User's optimum
    fig.add_trace(go.Scatter(
        x=[annualize_std(port["std"]) * 100],
        y=[annualize_return(port["return"]) * 100],
        mode="markers+text",
        marker=dict(size=22, color=rti["color"], symbol="star",
                    line=dict(width=2.5, color="white")),
        text=["Your Portfolio"], textposition="top center",
        textfont=dict(size=12, color=rti["color"], family="Inter"),
        name=f"Your Optimum (A={rti['A']:.1f})",
        hovertemplate=f"<b>Your Optimal Portfolio</b><br>"
                     f"Profile: {rti['label']}<br>"
                     f"A = {rti['A']:.2f}<br>"
                     f"Return: %{{y:.2f}}%<br>Vol: %{{x:.2f}}%<extra></extra>",
    ))

    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=60),
        xaxis=dict(title="Annualized Volatility (%)",
                   gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        yaxis=dict(title="Annualized Expected Return (%)",
                   gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# =============================================================================
# ROUTER
# =============================================================================
page = st.session_state.page
if   page == "welcome":       render_welcome()
elif page == "questionnaire": render_questionnaire()
elif page == "results":       render_results()
else:                         st.error(f"Unknown page: {page}")
