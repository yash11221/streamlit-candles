# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Interactive Candles + GANBOX", layout="wide")
st.title("Interactive Candlestick Viewer — draw GANBOX (0.5 midline)")

# ---------- Helpers ----------
def find_column(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

@st.cache_data(show_spinner=False)
def load_csv_bytes(uploaded_file):
    data = uploaded_file.getvalue().decode("utf-8", errors="replace")
    return StringIO(data)

def parse_df(raw_io):
    raw_io.seek(0)
    try:
        df = pd.read_csv(raw_io)
    except Exception:
        raw_io.seek(0)
        df = pd.read_csv(raw_io, sep=';')
    return df

def preprocess_df(df):
    orig_cols = list(df.columns)
    df = df.rename(columns={c: c.strip() for c in orig_cols})
    cols = list(df.columns)

    time_col = find_column(cols, ["gmt_time","time","datetime","date","timestamp","date_time"])
    if time_col is None:
        raise ValueError("No datetime column found. Name it something like 'Gmt time' or 'time'.")

    open_col = find_column(cols, ["open","o","bidopen"])
    high_col = find_column(cols, ["high","h"])
    low_col = find_column(cols, ["low","l"])
    close_col = find_column(cols, ["close","c","bidclose","last"])

    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError(f"Missing OHLC columns. Found: {cols}")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)

    df = df.rename(columns={
        time_col: "time",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close"
    })

    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    return df

# ---------- Session state for boxes ----------
if "ganboxes" not in st.session_state:
    st.session_state.ganboxes = []   # each box: dict {id, x0_idx, x1_idx, y0, y1, created}

def add_ganbox(x0_idx, x1_idx, y0, y1):
    box_id = len(st.session_state.ganboxes) + 1
    st.session_state.ganboxes.append({
        "id": box_id,
        "x0_idx": int(x0_idx),
        "x1_idx": int(x1_idx),
        "y0": float(y0),
        "y1": float(y1),
        "created": datetime.utcnow().isoformat()
    })

def remove_ganbox(box_id):
    st.session_state.ganboxes = [b for b in st.session_state.ganboxes if b["id"] != box_id]

# ---------- Upload ----------
uploaded = st.file_uploader("Upload CSV (Gmt time,Open,High,Low,Close or similar)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to start. You can also drag the file onto the uploader.")
    st.stop()

# ---------- Load and preprocess ----------
try:
    raw_io = load_csv_bytes(uploaded)
    df = parse_df(raw_io)
    df = preprocess_df(df)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

st.success(f"Loaded {len(df):,} rows — {df['time'].min()} → {df['time'].max()}")

# ---------- Controls ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("Plot controls")
    resample_opt = st.selectbox("Resample (aggregate candles)", options=["None", "1T","5T","15T","30T","60T"], index=1)
    if resample_opt != "None":
        df_plot = df.set_index("time").resample(resample_opt).agg({
            "open":"first","high":"max","low":"min","close":"last"
        }).dropna().reset_index()
    else:
        df_plot = df.copy()

    # Downsample before plotting if huge
    max_points = st.slider("Max plotted candles (downsample if larger)", min_value=500, max_value=200000, value=20000, step=500)
    n = len(df_plot)
    if n > max_points:
        factor = int(np.ceil(n / max_points))
        df_plot = df_plot.iloc[::factor].reset_index(drop=True)
        st.info(f"Downsampled to {len(df_plot):,} candles (every {factor}th).")

with right:
    st.subheader("Draw GANBOX")
    st.markdown("Select start / end candle indices (or use time pickers below), then click **Add GANBOX**.")
    start_idx = st.number_input("Start candle index (0 = first)", min_value=0, max_value=max(0, len(df_plot)-1), value=0, step=1, format="%d")
    end_idx = st.number_input("End candle index", min_value=0, max_value=max(0, len(df_plot)-1), value=min(50, max(0, len(df_plot)-1)), step=1, format="%d")

    # quick pick using times
    start_time_picker = st.selectbox("Start time (choose for convenience)", options=[None] + list(df_plot['time'].astype(str).values)[:500], index=0)
    end_time_picker = st.selectbox("End time (choose for convenience)", options=[None] + list(df_plot['time'].astype(str).values)[:500], index=0)

    if start_time_picker:
        try:
            t = pd.to_datetime(start_time_picker)
            # find nearest index in the original plotting df
            start_idx = int(df_plot.index[df_plot['time'] >= t][0])
        except Exception:
            st.warning("Couldn't set start index from time picker.")
    if end_time_picker:
        try:
            t = pd.to_datetime(end_time_picker)
            end_idx = int(df_plot.index[df_plot['time'] <= t][-1])
        except Exception:
            st.warning("Couldn't set end index from time picker.")

    # ensure order
    if end_idx < start_idx:
        st.warning("End index is before start index — values will be swapped on Add.")
    box_action_col1, box_action_col2 = st.columns([1,1])
    with box_action_col1:
        add_click = st.button("Add GANBOX (using selected indices)")
    with box_action_col2:
        clear_all = st.button("Clear all GANBOXes")

# ---------- Add / clear actions ----------
if clear_all:
    st.session_state.ganboxes = []

if add_click:
    # clamp indices
    x0 = min(start_idx, end_idx)
    x1 = max(start_idx, end_idx)
    # compute y-range as min low and max high inside selection (use df_plot as plotted)
    seg = df_plot.iloc[x0:x1+1]
    if seg.empty:
        st.error("Selected range is empty — choose a valid index range.")
    else:
        y0 = float(seg['low'].min())
        y1 = float(seg['high'].max())
        add_ganbox(x0, x1, y0, y1)

# ---------- Build Plotly Figure ----------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_plot['time'],
    open=df_plot['open'],
    high=df_plot['high'],
    low=df_plot['low'],
    close=df_plot['close'],
    name='Candles',
    increasing_line_color='green',
    decreasing_line_color='red',
    showlegend=False
))

# add each saved ganbox as shapes
shapes = []
annotations = []
for b in st.session_state.ganboxes:
    # map index to time on df_plot
    try:
        x0_time = df_plot['time'].iloc[b['x0_idx']]
        x1_time = df_plot['time'].iloc[b['x1_idx']]
    except Exception:
        # if indices out of range (after resample/downsample), skip showing that box
        continue
    y0 = b['y0']
    y1 = b['y1']
    # rectangle
    shapes.append({
        'type': 'rect',
        'xref': 'x',
        'yref': 'y',
        'x0': x0_time,
        'x1': x1_time,
        'y0': y0,
        'y1': y1,
        'fillcolor': 'rgba(100,100,255,0.12)',
        'line': {'width': 1, 'color': 'rgba(100,100,255,0.6)'},
        'layer': 'below'
    })
    # midline at 0.5
    mid = (y0 + y1) / 2.0
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'y',
        'x0': x0_time,
        'x1': x1_time,
        'y0': mid,
        'y1': mid,
        'line': {'color': 'rgba(255,255,0,0.9)', 'width': 1.5, 'dash': 'dot'},
        'layer': 'above'
    })
    # annotation label with box id
    annotations.append({
        'xref':'x','yref':'y','x':x0_time,'y':y1,
        'text':f"GANBOX {b['id']}",
        'showarrow': False,
        'yanchor': 'bottom',
        'font': {'size':10, 'color':'white'},
        'bgcolor': 'rgba(0,0,0,0.4)'
    })

fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    xaxis_rangeslider_visible=True,
    template='plotly_dark',
    height=720,
    margin={'t':40, 'b':40, 'l':40, 'r':20}
)

# make y-axis zoomable more easily
fig.update_yaxes(fixedrange=False, autorange=True)

# ---------- Draw UI for existing boxes ----------
st.subheader("Canvas (zoom/pan with mouse — drag to zoom, scroll to zoom x-axis)")
col_a, col_b = st.columns([3,1])
with col_a:
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Your GANBOXes")
    if len(st.session_state.ganboxes) == 0:
        st.write("No GANBOXes yet — add one using the controls.")
    else:
        for b in st.session_state.ganboxes:
            try:
                x0t = str(df_plot['time'].iloc[b['x0_idx']])
                x1t = str(df_plot['time'].iloc[b['x1_idx']])
            except Exception:
                x0t = "out-of-range"
                x1t = "out-of-range"
            st.markdown(f"**ID {b['id']}** — `{x0t}` → `{x1t}`  |  y-range `{b['y0']:.5f}` → `{b['y1']:.5f}`")
            col1, col2 = st.columns([1,2])
            with col1:
                if st.button(f"Remove {b['id']}", key=f"rm_{b['id']}"):
                    remove_ganbox(b['id'])
                    st.experimental_rerun()

# ---------- Download processed view ----------
st.markdown("---")
st.caption("You can zoom & pan the plot. Use the range slider below the chart to quickly move time windows.")
with st.expander("Download plotted (possibly downsampled/resampled) CSV"):
    csv = df_plot.to_csv(index=False).encode('utf-8')
    st.download_button("Download plotted CSV", data=csv, file_name="plot_candles.csv", mime="text/csv")
