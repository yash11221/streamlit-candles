# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Candles + GANBOX Viewer", layout="wide")

st.title("Interactive Candlestick Viewer with GANBOX (Streamlit)")

st.markdown(
    """
Upload a CSV with OHLC (and a datetime column). The app auto-detects common column names.
- Required columns: time (gmt_time, date, datetime), open, high, low, close
- Upload large CSVs — app will downsample for plotting if needed.
"""
)

# ----- Upload -----
uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="CSV with Gmt time,Open,High,Low,Close or similar")

# Quick helper to try many name variants
def find_column(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # try fuzzy: contains
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

@st.cache_data(show_spinner=False)
def load_csv_bytes(uploaded_file):
    # read raw bytes to let pandas sniff separators
    data = uploaded_file.getvalue().decode("utf-8", errors="replace")
    return StringIO(data)

def parse_df(raw_io):
    # Try common separators
    try:
        df = pd.read_csv(raw_io)
    except Exception:
        raw_io.seek(0)
        df = pd.read_csv(raw_io, sep=';')
    return df

def preprocess_df(df):
    # standardize column names
    orig_cols = list(df.columns)
    cols_map = {c: c.strip() for c in orig_cols}
    df = df.rename(columns=cols_map)
    cols = list(df.columns)

    # detect time column
    time_col = find_column(cols, ["gmt_time", "time", "datetime", "date", "timestamp", "date_time"])
    if time_col is None:
        st.warning("Couldn't find a datetime column automatically. Please provide a column named 'gmt_time' or 'time' or similar.")
        raise ValueError("No datetime column found")

    # detect OHLC
    open_col = find_column(cols, ["open", "o", "bidopen"])
    high_col = find_column(cols, ["high", "h"])
    low_col = find_column(cols, ["low", "l"])
    close_col = find_column(cols, ["close", "c", "bidclose", "last"])

    if not all([open_col, high_col, low_col, close_col]):
        missing = [n for n,v in zip(["open","high","low","close"], [open_col,high_col,low_col,close_col]) if v is None]
        st.warning(f"Missing columns detected: {missing}. Found columns: {cols}.")
        raise ValueError("Missing OHLC columns")

    # parse datetimes safely
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce', infer_datetime_format=True)
    except Exception:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # rename to standard names
    df = df.rename(columns={
        time_col: "time",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close"
    })

    # ensure numeric
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    return df

# ----- App controls -----
if uploaded:
    try:
        raw_io = load_csv_bytes(uploaded)
        df = parse_df(raw_io)
        df = preprocess_df(df)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    st.success(f"Loaded {len(df):,} rows. Time range: {df['time'].min()} → {df['time'].max()}")

    # user options
    col1, col2, col3 = st.columns([1.2,1,1])
    with col1:
        show_ganbox = st.checkbox("Show GANBOX midline (0.5)", value=True)
        show_full_ganbox = st.checkbox("Show full GANBOX levels (0.0,0.25,0.5,0.75,1.0)", value=False)
    with col2:
        downsample = st.selectbox("Downsample for plotting (max points)", options=[500000,200000,100000,50000,20000,10000,5000,2000,1000], index=5)
    with col3:
        resample_agg = st.selectbox("Resample (optional)", options=["None","1T","5T","15T","30T","60T"], index=1, help="Aggregate candles (T = minutes). Useful for heavy data.")

    # allow date range selection
    min_time = df['time'].min()
    max_time = df['time'].max()
    start_end = st.slider("Date range", min_value=min_time, max_value=max_time, value=(min_time, max_time), format="YYYY-MM-DD HH:mm")
    start_dt, end_dt = start_end

    df = df[(df['time'] >= pd.to_datetime(start_dt)) & (df['time'] <= pd.to_datetime(end_dt))]

    # optional resampling (requires time as index)
    if resample_agg != "None":
        df = df.set_index("time").resample(resample_agg).agg({
            "open":"first",
            "high":"max",
            "low":"min",
            "close":"last"
        }).dropna().reset_index()

    # downsample if too many points (uniform sampling)
    n = len(df)
    if n > downsample:
        factor = int(np.ceil(n / downsample))
        df = df.iloc[::factor].reset_index(drop=True)
        st.info(f"Downsampled to {len(df):,} points (every {factor}th row).")

    # compute GANBOX midline and optional levels
    df['ganbox_midline'] = (df['high'] + df['low']) / 2.0
    if show_full_ganbox:
        df['ganbox_0'] = df['low']
        df['ganbox_25'] = df['low'] + 0.25 * (df['high'] - df['low'])
        df['ganbox_50'] = df['ganbox_midline']
        df['ganbox_75'] = df['low'] + 0.75 * (df['high'] - df['low'])
        df['ganbox_100'] = df['high']

    # Plotly candlestick
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    if show_ganbox:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['ganbox_midline'],
            mode='lines',
            line=dict(width=1.2, dash='dot'),
            name='GANBOX 0.5'
        ))

    if show_full_ganbox:
        fig.add_trace(go.Scatter(x=df['time'], y=df['ganbox_25'], mode='lines', name='GANBOX 0.25', line=dict(dash='dash', width=1)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['ganbox_75'], mode='lines', name='GANBOX 0.75', line=dict(dash='dash', width=1)))
        # top/bottom thin lines
        fig.add_trace(go.Scatter(x=df['time'], y=df['ganbox_0'], mode='lines', name='GANBOX 0.0', line=dict(width=0.6)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['ganbox_100'], mode='lines', name='GANBOX 1.0', line=dict(width=0.6)))

    fig.update_layout(
        xaxis_rangeslider_visible=True,
        height=700,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # show a small table preview and download trimmed dataset
    with st.expander("Data preview & download"):
        st.dataframe(df.head(200))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download processed CSV", data=csv, file_name="processed_candles.csv", mime="text/csv")

    st.markdown("---")
    st.caption("If the chart is slow for very large CSVs, try resampling (5T/15T) or lower downsample option. You can also filter the date range before plotting.")
else:
    st.info("Upload a CSV to start. If you want, you can drag a CSV into the uploader or select one from your device.")
    st.markdown("""
    **CSV format tips**
    - Column names: `Gmt time`, `Open`, `High`, `Low`, `Close` (common), or any variants such as `time`, `datetime`, `o`, `h`, `l`, `c`.
    - Time column should be parseable by pandas.
    - Decimal separator: use `.` for decimal. If `,` is used, convert before upload.
    """)
