# Robo Adviser — BMD5302 Group 16

A personalized portfolio recommendation platform built on Markowitz mean-variance optimization.

Part 3 of the NUS BMD5302 Financial Modeling group project (AY 2025/26 Semester 2).

## Live Demo

🚀 **[Try it live](https://your-app-name.streamlit.app)** *(link will be active after deployment)*

## What It Does

The platform takes an investor through a structured 20-question risk questionnaire covering four dimensions (Investment Horizon, Financial Capacity, Behavioral Tolerance, Knowledge & Experience), aggregates the responses into a weighted Risk Tolerance Index, maps the index to a continuous risk aversion coefficient A ∈ [1, 10], and then solves a long-only mean-variance optimization over a universe of 10 FSMOne funds to produce a personalized optimal portfolio.

Every step is deterministic — two users with identical answers always receive identical portfolios.

## Tech Stack

- **Streamlit** — web UI and reactive state
- **Plotly** — interactive charts
- **SciPy (SLSQP)** — on-the-fly portfolio optimization
- **NumPy / pandas** — numerical and data handling

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
.
├── app.py                      # Main Streamlit application (3 pages)
├── data.py                     # Fund data, covariance matrix, RTI & optimization logic
├── ef_points_noshort.json      # Pre-computed efficient frontier points
├── requirements.txt
└── .streamlit/config.toml      # Theme
```

## Group 16

Chen Yang · Cheng Yiming · Lai Tianzhou · Lin Jie · Zou Zhihua
