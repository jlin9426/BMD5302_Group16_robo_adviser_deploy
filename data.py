"""
Pre-computed static data and helper functions for the Robo Adviser.

Source: Part 1 analysis (Copy_of_BMD5302_Part_1_v5.xlsx).
Methodology: follows Part 2 writeup (weighted RTI → continuous A → mean-variance
optimization with long-only constraint).
"""
import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize

# =============================================================================
# 1. FUND UNIVERSE
# =============================================================================
FUNDS = [
    {"short": "Asia Equity",    "full": "AB FCP I - Asia Ex-Japan Equity Portfolio",         "asset_class": "Equity",      "mean": -0.001480, "std": 0.015945},
    {"short": "US Income",      "full": "AB FCP I - American Income Portfolio",              "asset_class": "Fixed Income","mean":  0.000016, "std": 0.002261},
    {"short": "EU Income",      "full": "AB FCP I - European Income Portfolio",              "asset_class": "Fixed Income","mean":  0.000033, "std": 0.005062},
    {"short": "Global HY",      "full": "AB FCP I - Global High Yield Portfolio",            "asset_class": "Fixed Income","mean": -0.000039, "std": 0.002583},
    {"short": "Int'l Tech",     "full": "AB SICAV I - International Technology Portfolio",   "asset_class": "Equity",      "mean": -0.001594, "std": 0.015178},
    {"short": "EM Multi-Asset", "full": "AB SICAV I - Emerging Markets Multi-Asset",         "asset_class": "Multi-Asset", "mean": -0.000868, "std": 0.008368},
    {"short": "Int'l Health",   "full": "AB SICAV I - International Health Care Portfolio",  "asset_class": "Equity",      "mean": -0.000129, "std": 0.009238},
    {"short": "Low Vol Eq",     "full": "AB SICAV I - Low Volatility Equity Portfolio",      "asset_class": "Equity",      "mean": -0.000038, "std": 0.006536},
    {"short": "Asian Credit",   "full": "ABRDN SICAV I - Asian Credit Opportunities Fund",   "asset_class": "Fixed Income","mean": -0.000277, "std": 0.002000},
    {"short": "China A-Share",  "full": "ABRDN SICAV I - China A Shares Equity Fund",        "asset_class": "Equity",      "mean": -0.000924, "std": 0.010959},
]

FUND_NAMES    = [f["short"] for f in FUNDS]
MEAN_RETURNS  = np.array([f["mean"] for f in FUNDS])
N_FUNDS       = len(FUNDS)

# =============================================================================
# 2. COVARIANCE MATRIX (daily, 10×10) — extracted from Part 1 EFF sheet
# =============================================================================
COV_MATRIX = np.array([
    [ 2.5424e-04,  8.0530e-07,  5.3068e-06, -1.8340e-06,  6.0974e-06,  1.6469e-05, -1.8069e-06, -7.2036e-07,  6.3898e-06,  6.8959e-05],
    [ 8.0530e-07,  5.1128e-06,  6.3326e-06,  2.9820e-06,  6.9891e-06,  6.2376e-06,  8.9937e-06,  5.7683e-06,  5.7310e-07, -2.2466e-07],
    [ 5.3068e-06,  6.3326e-06,  2.5627e-05,  3.9533e-06,  8.5996e-06,  1.3778e-05,  1.2242e-05,  8.7977e-06,  1.9068e-06, -3.4810e-07],
    [-1.8340e-06,  2.9820e-06,  3.9533e-06,  6.6715e-06,  1.8163e-05,  1.0072e-05,  4.5697e-06,  7.6898e-06,  3.8181e-07, -1.9014e-06],
    [ 6.0974e-06,  6.9891e-06,  8.5996e-06,  1.8163e-05,  2.3038e-04,  9.7984e-05,  2.8250e-05,  7.5772e-05,  2.6495e-06,  3.8410e-07],
    [ 1.6469e-05,  6.2376e-06,  1.3778e-05,  1.0072e-05,  9.7984e-05,  7.0023e-05,  2.0860e-05,  3.8388e-05,  3.5872e-06,  3.6887e-06],
    [-1.8069e-06,  8.9937e-06,  1.2242e-05,  4.5697e-06,  2.8250e-05,  2.0860e-05,  8.5344e-05,  3.4321e-05,  2.1527e-06, -4.7552e-06],
    [-7.2036e-07,  5.7683e-06,  8.7977e-06,  7.6898e-06,  7.5772e-05,  3.8388e-05,  3.4321e-05,  4.2716e-05,  1.7927e-06, -5.9232e-06],
    [ 6.3898e-06,  5.7310e-07,  1.9068e-06,  3.8181e-07,  2.6495e-06,  3.5872e-06,  2.1527e-06,  1.7927e-06,  4.0000e-06,  4.8691e-06],
    [ 6.8959e-05, -2.2466e-07, -3.4810e-07, -1.9014e-06,  3.8410e-07,  3.6887e-06, -4.7552e-06, -5.9232e-06,  4.8691e-06,  1.2111e-04],
])

# =============================================================================
# 3. GMVP (long-only, from Part 1)
# =============================================================================
GMVP = {
    "weights": [0.0000, 0.2805, 0.0000, 0.2020, 0.0000, 0.0000, 0.0000, 0.0000, 0.5147, 0.0028],
    "return":  -0.0001,
    "std":      0.0016,
}

# =============================================================================
# 4. RTI CONFIGURATION (per Part 2 methodology, Section 2.3)
# =============================================================================
# Weights for the 4 dimensions — greater weight on behavioral & financial
# capacity as specified in Part 2 Section 2.3.
RTI_WEIGHTS = {
    "H": 0.20,   # Investment Horizon
    "F": 0.30,   # Financial Capacity
    "B": 0.30,   # Behavioral Risk Tolerance
    "K": 0.20,   # Investment Knowledge
}
assert abs(sum(RTI_WEIGHTS.values()) - 1.0) < 1e-9

# A range — standard academic bounds
A_MIN = 1.0
A_MAX = 10.0

def compute_rti_and_A(answers: dict) -> dict:
    """
    Compute dimension sub-indices, weighted RTI* and risk aversion coefficient A.

    Args:
        answers: dict like {"Q1": 3, "Q2": 4, ..., "Q20": 2}
    Returns:
        dict with S_A/B/C/D, H/F/B/K, RTI, A, label, color
    """
    # Section totals (each question scored 1-5, 5 questions per section)
    S_A = sum(answers[f"Q{i}"] for i in range(1, 6))    # Horizon
    S_B = sum(answers[f"Q{i}"] for i in range(6, 11))   # Financial
    S_C = sum(answers[f"Q{i}"] for i in range(11, 16))  # Behavioral
    S_D = sum(answers[f"Q{i}"] for i in range(16, 21))  # Knowledge

    # Normalize each dimension to [0, 1] using min=5, max=25
    H = (S_A - 5) / 20
    F = (S_B - 5) / 20
    B = (S_C - 5) / 20
    K = (S_D - 5) / 20

    # Weighted RTI*
    RTI = (RTI_WEIGHTS["H"] * H +
           RTI_WEIGHTS["F"] * F +
           RTI_WEIGHTS["B"] * B +
           RTI_WEIGHTS["K"] * K)

    # Linear mapping to A: high RTI → low A
    A = A_MAX - RTI * (A_MAX - A_MIN)

    return {
        "S_A": S_A, "S_B": S_B, "S_C": S_C, "S_D": S_D,
        "total_score": S_A + S_B + S_C + S_D,
        "H": H, "F": F, "B": B, "K": K,
        "RTI": RTI, "A": A,
        "label": risk_label(A),
        "color": risk_color(A),
        "description": risk_description(A),
    }

def risk_label(A: float) -> str:
    """Map a continuous A value to a discrete profile label."""
    if   A >= 8.0: return "Very Conservative"
    elif A >= 6.0: return "Conservative"
    elif A >= 3.5: return "Moderate"
    elif A >= 2.0: return "Aggressive"
    else:          return "Very Aggressive"

def risk_color(A: float) -> str:
    if   A >= 8.0: return "#2E86AB"
    elif A >= 6.0: return "#6A994E"
    elif A >= 3.5: return "#F4A261"
    elif A >= 2.0: return "#E76F51"
    else:          return "#B23A48"

def risk_description(A: float) -> str:
    return {
        "Very Conservative": "Prioritizes capital preservation over growth; minimal tolerance for drawdowns.",
        "Conservative":      "Favors stable returns with limited downside; prefers income-generating assets.",
        "Moderate":          "Balances growth and stability; willing to accept moderate fluctuations.",
        "Aggressive":        "Seeks long-term growth; comfortable with significant short-term volatility.",
        "Very Aggressive":   "Maximizes capital appreciation; accepts high volatility for return potential.",
    }[risk_label(A)]

# =============================================================================
# 5. ON-THE-FLY PORTFOLIO OPTIMIZATION (long-only, continuous A)
# =============================================================================
def optimize_portfolio(A: float) -> dict:
    """
    Solve:  max_w  μ'w - (A/2) w'Σw
    s.t.    sum(w) = 1,   w_i ∈ [0, 1]  for all i

    Returns dict with keys: weights, return, std, variance, utility
    """
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bnds = [(0, 1)] * N_FUNDS
    w0 = np.ones(N_FUNDS) / N_FUNDS

    def neg_u(w):
        return -(MEAN_RETURNS @ w - 0.5 * A * w @ COV_MATRIX @ w)

    res = minimize(neg_u, w0, method="SLSQP",
                   constraints=cons, bounds=bnds,
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = np.where(res.x < 1e-8, 0, res.x)
    w = w / w.sum()

    r   = float(MEAN_RETURNS @ w)
    var = float(w @ COV_MATRIX @ w)
    sd  = float(np.sqrt(var))
    u   = r - 0.5 * A * var

    return {
        "weights":  w.tolist(),
        "return":   r,
        "std":      sd,
        "variance": var,
        "utility":  u,
    }

# =============================================================================
# 6. EFFICIENT FRONTIER CURVE (long-only, for visualization)
# =============================================================================
_EF_PATH = Path(__file__).parent / "ef_points_noshort.json"
if _EF_PATH.exists():
    with open(_EF_PATH) as _f:
        EF_POINTS = json.load(_f)
else:
    EF_POINTS = []

# =============================================================================
# 7. QUESTIONNAIRE (20 questions across 4 dimensions, 1-5 scoring)
# =============================================================================
QUESTIONS = [
    # ---- Section A · Investment Horizon and Objectives ----
    {"id": "Q1", "section": "A", "section_title": "Investment Horizon and Objectives",
     "text": "What is your intended holding period for this investment portfolio?",
     "options": [("Less than 1 year", 1), ("1–3 years", 2), ("3–5 years", 3), ("5–10 years", 4), ("More than 10 years", 5)]},
    {"id": "Q2", "section": "A", "section_title": "Investment Horizon and Objectives",
     "text": "What is your primary investment objective?",
     "options": [("Preserve capital", 1), ("Generate stable income", 2), ("Balance income and growth", 3), ("Achieve long-term growth", 4), ("Maximize capital appreciation", 5)]},
    {"id": "Q3", "section": "A", "section_title": "Investment Horizon and Objectives",
     "text": "How important is avoiding short-term losses to you?",
     "options": [("Extremely important", 1), ("Very important", 2), ("Moderately important", 3), ("Slightly important", 4), ("Not important", 5)]},
    {"id": "Q4", "section": "A", "section_title": "Investment Horizon and Objectives",
     "text": "How soon do you expect to need a substantial portion of this money?",
     "options": [("Within 12 months", 1), ("Within 1–2 years", 2), ("Within 3–5 years", 3), ("Within 5–10 years", 4), ("No foreseeable need", 5)]},
    {"id": "Q5", "section": "A", "section_title": "Investment Horizon and Objectives",
     "text": "Which statement best describes your attitude toward long-term investing?",
     "options": [("I prefer certainty even if returns are low", 1), ("I accept very little fluctuation for modest returns", 2), ("I can accept some fluctuation for balanced returns", 3), ("I accept sizable fluctuation for higher long-term returns", 4), ("I prioritize long-term return even if volatility is high", 5)]},

    # ---- Section B · Financial Capacity ----
    {"id": "Q6", "section": "B", "section_title": "Financial Capacity",
     "text": "How stable is your main source of income?",
     "options": [("Very unstable", 1), ("Somewhat unstable", 2), ("Moderately stable", 3), ("Stable", 4), ("Very stable", 5)]},
    {"id": "Q7", "section": "B", "section_title": "Financial Capacity",
     "text": "How much emergency savings do you currently have?",
     "options": [("Less than 3 months of expenses", 1), ("3–6 months", 2), ("6–12 months", 3), ("12–24 months", 4), ("More than 24 months", 5)]},
    {"id": "Q8", "section": "B", "section_title": "Financial Capacity",
     "text": "What proportion of your total investable wealth does this portfolio represent?",
     "options": [("More than 80%", 1), ("60%–80%", 2), ("40%–60%", 3), ("20%–40%", 4), ("Less than 20%", 5)]},
    {"id": "Q9", "section": "B", "section_title": "Financial Capacity",
     "text": "How would you describe your debt burden?",
     "options": [("Very high", 1), ("High", 2), ("Moderate", 3), ("Low", 4), ("Very low or none", 5)]},
    {"id": "Q10", "section": "B", "section_title": "Financial Capacity",
     "text": "If this portfolio lost 15% in value, how much would it affect your lifestyle or essential spending?",
     "options": [("Severely affect it", 1), ("Significantly affect it", 2), ("Moderately affect it", 3), ("Slightly affect it", 4), ("Not affect it at all", 5)]},

    # ---- Section C · Behavioral Risk Tolerance ----
    {"id": "Q11", "section": "C", "section_title": "Behavioral Risk Tolerance",
     "text": "If your portfolio fell by 10% in one month, what would you most likely do?",
     "options": [("Sell everything immediately", 1), ("Sell a large part", 2), ("Hold and wait", 3), ("Buy a little more", 4), ("Buy substantially more", 5)]},
    {"id": "Q12", "section": "C", "section_title": "Behavioral Risk Tolerance",
     "text": "If your portfolio fell by 20% over six months, how would you react?",
     "options": [("Exit the market completely", 1), ("Reduce exposure significantly", 2), ("Keep my position unchanged", 3), ("Increase allocation gradually", 4), ("Increase allocation aggressively", 5)]},
    {"id": "Q13", "section": "C", "section_title": "Behavioral Risk Tolerance",
     "text": "Which annual portfolio fluctuation would you tolerate for potentially higher long-term returns?",
     "options": [("Less than 5%", 1), ("5%–10%", 2), ("10%–15%", 3), ("15%–20%", 4), ("More than 20%", 5)]},
    {"id": "Q14", "section": "C", "section_title": "Behavioral Risk Tolerance",
     "text": "Which trade-off would you prefer?",
     "options": [("Very low risk and very low return", 1), ("Low risk and low return", 2), ("Moderate risk and moderate return", 3), ("High risk and high return", 4), ("Very high risk for maximum return potential", 5)]},
    {"id": "Q15", "section": "C", "section_title": "Behavioral Risk Tolerance",
     "text": "When markets become highly volatile, how confident are you in sticking to your investment plan?",
     "options": [("Not confident at all", 1), ("Slightly confident", 2), ("Moderately confident", 3), ("Confident", 4), ("Very confident", 5)]},

    # ---- Section D · Investment Knowledge and Experience ----
    {"id": "Q16", "section": "D", "section_title": "Investment Knowledge and Experience",
     "text": "How many years of investment experience do you have?",
     "options": [("None", 1), ("Less than 1 year", 2), ("1–3 years", 3), ("3–5 years", 4), ("More than 5 years", 5)]},
    {"id": "Q17", "section": "D", "section_title": "Investment Knowledge and Experience",
     "text": "How familiar are you with diversification?",
     "options": [("Not familiar at all", 1), ("Slightly familiar", 2), ("Moderately familiar", 3), ("Familiar", 4), ("Very familiar", 5)]},
    {"id": "Q18", "section": "D", "section_title": "Investment Knowledge and Experience",
     "text": "How familiar are you with the relationship between risk and expected return?",
     "options": [("Not familiar at all", 1), ("Slightly familiar", 2), ("Moderately familiar", 3), ("Familiar", 4), ("Very familiar", 5)]},
    {"id": "Q19", "section": "D", "section_title": "Investment Knowledge and Experience",
     "text": "How well do you understand the concept of volatility?",
     "options": [("Not at all", 1), ("Slightly", 2), ("Moderately", 3), ("Well", 4), ("Very well", 5)]},
    {"id": "Q20", "section": "D", "section_title": "Investment Knowledge and Experience",
     "text": "How comfortable are you making investment decisions after reviewing performance data and fund information?",
     "options": [("Very uncomfortable", 1), ("Uncomfortable", 2), ("Neutral", 3), ("Comfortable", 4), ("Very comfortable", 5)]},
]

DIMENSION_TITLES = {
    "A": "Investment Horizon and Objectives",
    "B": "Financial Capacity",
    "C": "Behavioral Risk Tolerance",
    "D": "Investment Knowledge and Experience",
}

# =============================================================================
# 8. ANNUALIZATION
# =============================================================================
TRADING_DAYS = 252

def annualize_return(daily: float) -> float:
    return daily * TRADING_DAYS

def annualize_std(daily_std: float) -> float:
    return daily_std * np.sqrt(TRADING_DAYS)
