import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta, datetime
import traceback

# Optional dependencies
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import talib
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

# =========================================
# 🌈 Page Config & Theme
# =========================================
st.set_page_config(
    page_title="💹 CFX - Explainable Stock Prediction System",
    layout="centered",
    page_icon="💹",
)

st.markdown(""" 
    <style>
    @keyframes softGlow {
        0% { text-shadow: 0 0 6px #a78bfa, 0 0 12px #60a5fa; }
        50% { text-shadow: 0 0 14px #f472b6, 0 0 28px #818cf8; }
        100% { text-shadow: 0 0 6px #a78bfa, 0 0 12px #60a5fa; }
    }
    @keyframes backgroundWave {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #312e81, #701a75);
        background-size: 300% 300%;
        animation: backgroundWave 15s ease infinite;
        color: #f1f5f9;
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        color: #e0e7ff;
        text-align: center;
        font-weight: 800;
        font-size: 2.9em;
        letter-spacing: 1px;
        animation: softGlow 4s ease-in-out infinite alternate;
        text-shadow: 0 0 10px rgba(192, 132, 252, 0.6);
    }
    .tagline {
        text-align: center;
        color: #c084fc;
        font-size: 1.1em;
        margin-top: -10px;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(120deg,
            rgba(99, 102, 241, 0.3),
            rgba(168, 85, 247, 0.25),
            rgba(236, 72, 153, 0.25));
        color: #f9fafb;
        font-weight: 600;
        border: 1px solid rgba(199, 210, 254, 0.3);
        border-radius: 14px;
        padding: 12px 30px;
        backdrop-filter: blur(10px);
        box-shadow:
            inset 0 0 12px rgba(255,255,255,0.05),
            0 0 20px rgba(147, 197, 253, 0.12);
        transition: all 0.35s ease-in-out;
    }
    div.stButton > button:hover {
        background: linear-gradient(120deg,
            rgba(168, 85, 247, 0.4),
            rgba(236, 72, 153, 0.35),
            rgba(99, 102, 241, 0.4));
        border-color: rgba(199, 210, 254, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(236, 72, 153, 0.25);
        color: #ffffff;
    }
    div.stButton > button:active {transform: scale(0.98);}
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #cbd5e1;
        font-size: 13px;
        letter-spacing: 0.3px;
    }
    .footer b {color: #a78bfa;}
    </style>
""", unsafe_allow_html=True)

# =========================================
# 🧠 Load Models & Helper
# =========================================
@st.cache_resource
def load_models():
    direction_model = joblib.load("models/xgb_direction_model.pkl")
    price_model = joblib.load("models/xgb_price_model.pkl")
    helper = joblib.load("models/prediction_helper_data.joblib")
    return direction_model, price_model, helper

direction_model, price_model, helper = load_models()

X_test = helper.get("X_test", None)
test_dates = pd.to_datetime(helper.get("test_dates", pd.Series([]))) if helper.get("test_dates", None) is not None else pd.Series([])
test_actual_close = helper.get("test_actual_close", pd.Series([]))
pattern_names_test = helper.get("pattern_names_test", pd.Series([]))
mae_price = helper.get("mae_price", None)
lookback_days = helper.get("lookback_days", 30)
YAHOO_TICKER = "^NSEI"

# =========================================
# 📊 Header
# =========================================
st.markdown("<h1>💹 CFX</h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>Candlestick–FinBERT–XGBoost: Explainable Hybrid Stock Prediction System ⚡</p>", unsafe_allow_html=True)

st.info("Pick a date and (optionally) enter financial news headlines. The app auto-detects whether to use Historic or Real-time mode.", icon="🔁")

# =========================================
# 📅 User Inputs
# =========================================
default_date = (test_dates.max().date() if len(test_dates) > 0 else datetime.today().date())
date_input = st.date_input("📅 Select a date:", value=default_date)
headline = st.text_area("📰 Optional: Enter headline(s) for this date (affects sentiment).", value="", height=120)

# =========================================
# 📈 Helper Functions
# =========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df2 = df.copy()
    try:
        if HAS_TALIB:
            df2["rsi_14"] = talib.RSI(df2["Close"], timeperiod=14)
        else:
            df2["rsi_14"] = compute_rsi(df2["Close"], 14)
    except Exception:
        df2["rsi_14"] = compute_rsi(df2["Close"], 14)
    df2["std_10"] = df2["Close"].rolling(10).std()
    return df2

# =========================================
# 🔮 Prediction Button
# =========================================
if st.button("🔮 Predict Next Day"):
    try:
        date_ts = pd.Timestamp(date_input)

        # ---- FIXED: Compare date correctly ----
        in_historic = False
        if len(test_dates) > 0:
            test_dates_dates = pd.to_datetime(test_dates).dt.date
            if any(test_dates_dates == date_ts.date()):
                in_historic = True

        # =========================================
        # 🧾 HISTORIC MODE
        # =========================================
        if in_historic and X_test is not None:
            idx = test_dates[test_dates.dt.date == date_ts.date()].index[0]
            features = X_test.iloc[[idx]]
            direction_pred = direction_model.predict(features)[0]
            direction_proba = direction_model.predict_proba(features)[0]
            confidence = float(np.max(direction_proba) * 100)
            pct_change_pred = price_model.predict(features)[0]
            last_close = float(test_actual_close.iloc[idx])
            next_day_price = last_close * (1 + pct_change_pred)
            direction_label = "🟢 Uptrend" if direction_pred == 1 else "🔴 Downtrend / Sideways"
            if mae_price:
                low = next_day_price - mae_price
                high = next_day_price + mae_price
            else:
                low = next_day_price * 0.99
                high = next_day_price * 1.01
            pattern_today = pattern_names_test.iloc[idx] if len(pattern_names_test) > idx else "N/A"

            st.markdown("### 🔮 Prediction Summary (Historic)")
            st.markdown(f"**📅 Input Date:** {date_ts.date()}")
            st.markdown(f"**📊 Prediction for:** {(date_ts + timedelta(days=1)).date()}")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"**Market Direction:** {direction_label}")
            st.markdown(f"**Predicted Closing Price:** ₹{next_day_price:,.2f}")
            st.markdown(f"**Expected Range:** ₹{low:,.2f} – ₹{high:,.2f}")
            st.markdown(f"**Model Confidence:** {confidence:.2f}%")
            st.markdown(f"**Detected Candlestick Pattern(s):** {pattern_today}")

        # =========================================
        # ⚡ REAL-TIME MODE (with full pattern features)
        # =========================================
        else:
            if yf is None:
                st.error("❌ yfinance not available.")
                raise RuntimeError("yfinance missing")

            # 🗓 Fetch live NSEI data
            start_date = date_ts - pd.Timedelta(days=lookback_days)
            end_date = date_ts + pd.Timedelta(days=2)
            df = yf.download(
                YAHOO_TICKER,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                st.error("⚠️ No market data found (maybe weekend/holiday).")
                raise ValueError("Empty data")

            df = df.sort_index()
            df = add_indicators(df)
            history_df = df.copy()

            # 📰 FinBERT Sentiment
            sentiment_score = 0
            headlines = [h.strip() for h in headline.strip().split("\n") if h.strip()]
            if HAS_TRANSFORMERS and len(headlines) > 0:
                if "finbert_pipeline" not in st.session_state:
                    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                    model_f = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                    st.session_state.finbert_pipeline = pipeline("sentiment-analysis", model=model_f, tokenizer=tokenizer)
                finpipe = st.session_state.finbert_pipeline
                results = finpipe(headlines)
                pos = sum(1 for r in results if r["label"].lower() == "positive")
                neg = sum(1 for r in results if r["label"].lower() == "negative")
                sentiment_score = pos - neg

            # 🧮 Indicators
            rsi = float(history_df['rsi_14'].iloc[-1])
            rolling_std = float(history_df['Close'].rolling(window=5).std().iloc[-1])
            last_close_price = float(history_df['Close'].iloc[-1])

            # 🕯️ Patterns
            pattern_features = {}
            pattern_names = []
            if HAS_TALIB:
                for pattern_code in talib.get_function_groups()['Pattern Recognition']:
                    try:
                        func = getattr(talib, pattern_code)
                        result = func(history_df['Open'], history_df['High'], history_df['Low'], history_df['Close'])
                        is_pattern = 1 if result.iloc[-1] != 0 else 0
                        pattern_features[f'pattern_{pattern_code}'] = is_pattern
                        if is_pattern:
                            pattern_names.append(pattern_code)
                    except Exception:
                        pattern_features[f'pattern_{pattern_code}'] = 0

            pattern_name_str = ", ".join(pattern_names) if pattern_names else "No Specific Pattern"

            # 🧠 Assemble features
            feature_vector = [sentiment_score, rsi, rolling_std] + list(pattern_features.values())
            columns = ['sentiment_score', 'RSI', 'rolling_std_5'] + list(pattern_features.keys())
            model_input = pd.DataFrame([feature_vector], columns=columns)

            # 🔮 Predict
            direction_pred = direction_model.predict(model_input)[0]
            direction_proba = direction_model.predict_proba(model_input)[0]
            direction = "🟢 Uptrend" if direction_pred == 1 else "🔴 Downtrend / Sideways"
            confidence = float(direction_proba[direction_pred] * 100)
            pct_change_pred = price_model.predict(model_input)[0]
            price_pred = last_close_price * (1 + pct_change_pred)
            low, high = (price_pred - mae_price, price_pred + mae_price) if mae_price else (price_pred * 0.99, price_pred * 1.01)

            # 🧾 Display
            st.markdown("### 🔮 Real-time Prediction Summary")
            st.markdown(f"**📅 Input Date:** {date_ts.date()}")
            st.markdown(f"**📊 Prediction for:** {(date_ts + timedelta(days=1)).date()}")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"**Market Direction:** {direction}")
            st.markdown(f"**Predicted Closing Price:** ₹{price_pred:,.2f}")
            st.markdown(f"**Expected Range:** ₹{low:,.2f} – ₹{high:,.2f}")
            st.markdown(f"**Model Confidence:** {confidence:.2f}%")
            st.markdown(f"**Detected Pattern(s):** {pattern_name_str}")
            st.markdown(f"**Sentiment Score:** {sentiment_score}")
            st.markdown(f"**RSI:** {rsi:.2f}")

            # 🧠 Explanation
            explanation = f"The model predicts tomorrow's direction is likely to be {direction} with {confidence:.1f}% confidence.\n\n"
            if sentiment_score > 0:
                explanation += "🟩 News Sentiment was Positive.\n"
            elif sentiment_score < 0:
                explanation += "🟥 News Sentiment was Negative.\n"
            else:
                explanation += "⬜ No strong sentiment detected.\n"
            if pattern_names:
                explanation += f"📈 Candlestick Pattern(s) Detected: {pattern_name_str}\n"
            if rsi > 70:
                explanation += "⚠️ RSI indicates overbought momentum.\n"
            elif rsi < 30:
                explanation += "💹 RSI indicates oversold momentum.\n"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 🧠 Model Explanation")
            st.info(explanation)

    except Exception as e:
        st.error("⚠️ Error during prediction:")
        st.error(traceback.format_exc())

# =========================================
# 📏 Model Evaluation (Accuracy & MAE)
# =========================================
# =========================================
# 📏 Model Performance Metrics
# =========================================
try:
    st.sidebar.markdown("## 🧠 Model Performance")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Extract stored test data
    X_test = helper.get("X_test", None)
    y_true_direction = helper.get("y_test_direction", None)
    y_true_price = helper.get("y_test_price_change", None)
    test_actual_close = helper.get("test_actual_close", pd.Series([]))

    # ----- Direction Accuracy -----
    if X_test is not None and y_true_direction is not None:
        y_pred_direction = direction_model.predict(X_test)
        direction_accuracy = (y_pred_direction == y_true_direction).mean() * 100
        st.sidebar.success(f"**Direction Accuracy:** {direction_accuracy:.2f}%")
    else:
        st.sidebar.warning("Direction Accuracy: Not Available")

    # ----- Price Model MAE -----
    if X_test is not None and y_true_price is not None:
        y_pred_price = price_model.predict(X_test)
        mae_price_calc = np.mean(np.abs(y_true_price - y_pred_price))
        st.sidebar.info(f"**Price MAE:** ±{mae_price_calc:.2f}")
    elif "mae_price" in helper:
        mae_price_calc = helper["mae_price"]
        st.sidebar.info(f"**Price MAE (Saved):** ±{mae_price_calc:.2f}")
    else:
        mae_price_calc = None
        st.sidebar.warning("Price MAE: Not Available")

    # ----- Approx Price Accuracy -----
    if mae_price_calc is not None and len(test_actual_close) > 0:
        avg_close = np.mean(test_actual_close)
        approx_price_accuracy = 100 - ((mae_price_calc / avg_close) * 100)
        st.sidebar.success(f"**Approx Price Accuracy:** {approx_price_accuracy:.2f}%")

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.caption("💹 Accuracy values based on test data from training phase.")

except Exception as e:
    st.sidebar.error("⚠️ Error computing accuracy metrics.")
    st.sidebar.code(str(e))

# =========================================
# 🧑‍💻 Footer
# =========================================
st.markdown(
    "<div class='footer'>© 2025 <b style='color:#00eaff;'>CFX</b> | Developed by <b style='color:#38bdf8;'>Shirin & Shruti</b></div>",
    unsafe_allow_html=True
)
