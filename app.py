"""
app.py
The Streamlit web app. Run with:  streamlit run app.py
The user pastes a URL; the app extracts domain features, asks the saved model
for a phishing probability and shows the verdict, a risk meter and notes.
"""
import html                                   # html.escape makes user text safe to show in HTML
from urllib.parse import urlparse             # used to read the scheme/path for the info notes

import joblib                                 # loads the saved model
import pandas as pd
import streamlit as st                        # the web app framework

from features import extract_domain_features  # same feature function used in training

# Must be the first Streamlit call: sets the browser tab title, icon and page width.
st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="centered")

# Custom CSS (styling). unsafe_allow_html=True lets Streamlit accept raw HTML/CSS.
st.markdown(
    """
<style>
/* Gradient banner at the top of the page */
.hero {
    padding: 2rem 1.5rem; border-radius: 20px; text-align: center;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35);
}
.hero h1 { margin: 0; font-size: 2.4rem; color: white; }
.hero p { margin: .4rem 0 0; opacity: .9; font-size: 1.05rem; }
/* Result card: red for phishing (.bad), green for legitimate (.good) */
.result { padding: 1.4rem 1.6rem; border-radius: 16px; margin: 1rem 0; }
.result.bad  { background: rgba(239,68,68,.12);  border: 1px solid #ef4444; }
.result.good { background: rgba(34,197,94,.12);  border: 1px solid #22c55e; }
.result h2 { margin: 0 0 .3rem; }
.result .url { font-family: monospace; opacity: .8; word-break: break-all; font-size: .9rem; }
/* Risk meter: a green-to-red bar with a white marker showing the probability */
.meter { position: relative; height: 14px; border-radius: 99px; margin: 1.6rem 0 .4rem;
    background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); }
.marker { position: absolute; top: -6px; width: 6px; height: 26px; border-radius: 4px;
    background: white; box-shadow: 0 0 8px rgba(0,0,0,.6); transform: translateX(-50%); }
.meter-labels { display: flex; justify-content: space-between; font-size: .8rem; opacity: .7; }
/* Small note boxes: red = warning, green = good sign, gray = information */
.flag { padding: .55rem .9rem; border-radius: 10px; margin: .35rem 0; font-size: .95rem; }
.flag.red   { background: rgba(239,68,68,.10); border-left: 4px solid #ef4444; }
.flag.green { background: rgba(34,197,94,.10); border-left: 4px solid #22c55e; }
.flag.gray  { background: rgba(148,163,184,.12); border-left: 4px solid #94a3b8; }
div.stButton > button { border-radius: 12px; }  /* rounded buttons */
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource            # load the model only once and reuse it on every rerun
def load_model():
    bundle = joblib.load("models/phishing_model.pkl")
    return bundle["model"], bundle["features"]


model, feature_names = load_model()

# Example buttons: label shown on the button -> URL placed in the input box.
SAMPLES = {
    "✅ Google": "https://www.google.com",
    "✅ GitHub": "https://github.com/microsoft/vscode",
    "⚠️ Fake PayPal": "http://secure-login-paypal-verify.com/account/update",
    "⚠️ IP address": "http://192.168.1.5/login.php",
    "⚠️ Crypto bonus": "http://free-crypto-bonus-claim.top/wallet/confirm",
}


def explain(f: dict, url: str):
    """Plain-English notes about the domain, plus info that is NOT scored."""
    reds, greens, infos = [], [], []             # three lists: warnings, good signs, information
    if f["has_ip"]:
        reds.append("Uses a raw IP address instead of a domain name")
    if f["has_userinfo"]:
        reds.append("Contains '@', which can hide the real destination")
    if f["suspicious_word_count"] > 0:
        reds.append(f"Domain contains {f['suspicious_word_count']} suspicious keyword(s) such as login, verify or secure")
    if f["num_hyphens_domain"] >= 2:
        reds.append("Domain name has several hyphens")
    if f["num_subdomains"] >= 3:
        reds.append("Unusually many subdomains")
    if f["hostname_length"] > 30:
        reds.append(f"Long hostname ({f['hostname_length']} characters)")
    if f["domain_has_digits"]:
        reds.append("Domain name contains digits")
    if f["has_port"]:
        reds.append("Uses a non-standard port")
    if not reds:                                  # nothing suspicious was found
        greens.append("No common domain-level red flags found")

    try:
        p = urlparse(url if "://" in url else "http://" + url)   # split the URL (add http:// if missing)
        infos.append("Uses HTTPS (not scored; phishing sites can use HTTPS too)"
                     if p.scheme == "https" else "Does not use HTTPS (not scored)")
        if len(p.path) > 1 or p.query:
            infos.append("This link has a path or query string; the model does not score that part")
    except ValueError:                            # ignore URLs that cannot be parsed
        pass
    return reds, greens, infos


# session_state keeps values between reruns (Streamlit reruns the script on every click).
if "history" not in st.session_state:
    st.session_state.history = []                 # list of past checks


def use_sample(u):
    """Called when an example button is clicked: fill the box and trigger a check."""
    st.session_state.url_input = u
    st.session_state.auto_check = True


# Sidebar: explanation, limitation, threshold slider and a clear-history button.
with st.sidebar:
    st.header("🛡️ About")
    st.write(
        "PhishGuard analyses the **domain name** of a link (length, digits, "
        "hyphens, subdomains, keywords, TLD) and a machine learning model "
        "estimates how likely it is to be phishing."
    )
    st.subheader("How it works")
    st.markdown(
        "1. You paste a URL\n"
        "2. The hostname is extracted\n"
        f"3. {len(feature_names)} domain features are computed\n"
        "4. An XGBoost / Random Forest model gives a probability"
    )
    st.warning("Limitation: the path and query part of a link is not scored.")
    st.info("It never opens the link, so checking is safe.")
    # Probabilities at or above this value are called phishing (default 0.5).
    threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Lower = catches more phishing but raises more false alarms.")
    if st.button("🗑️ Clear history"):
        st.session_state.history = []

# Gradient header banner (uses the .hero CSS above).
st.markdown(
    """
<div class="hero">
  <h1>🛡️ PhishGuard</h1>
  <p>Paste a link. Find out if it's a trap before you click.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Text box for the URL. key="url_input" lets the example buttons fill it.
url = st.text_input("Enter a URL", key="url_input", placeholder="https://example.com/login")

st.caption("Try an example:")
cols = st.columns(len(SAMPLES))                   # one column per example button
for col, (label, u) in zip(cols, SAMPLES.items()):
    col.button(label, on_click=use_sample, args=(u,), use_container_width=True)

check = st.button("🔍 Check URL", type="primary", use_container_width=True)
auto = st.session_state.pop("auto_check", False)  # True right after an example button was clicked

if check or auto:                                 # run when either button was used
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        feats_dict = extract_domain_features(url)  # numbers computed from the hostname
        if feats_dict["hostname_length"] == 0:     # no hostname found = invalid input
            st.error("That doesn't look like a valid URL. Try something like https://example.com")
            st.stop()                              # stop the app here

        # One-row table in the exact column order the model was trained with.
        feats = pd.DataFrame([feats_dict])[feature_names]
        prob = float(model.predict_proba(feats)[0][1])   # probability that it is phishing (0 to 1)
        is_bad = prob >= threshold                 # compare with the slider value
        safe_url = html.escape(url.strip())        # escape so the URL cannot inject HTML

        if is_bad:                                 # show the red or green result card
            st.markdown(f'<div class="result bad"><h2>🚨 Likely PHISHING</h2>'
                        f'<div class="url">{safe_url}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result good"><h2>✅ Likely LEGITIMATE</h2>'
                        f'<div class="url">{safe_url}</div></div>', unsafe_allow_html=True)

        # Three number tiles: probability, confidence and risk level.
        c1, c2, c3 = st.columns(3)
        c1.metric("Phishing probability", f"{prob:.1%}")
        c2.metric("Confidence", f"{max(prob, 1 - prob):.1%}")
        c3.metric("Risk level", "High" if prob >= 0.75 else "Medium" if prob >= 0.4 else "Low")

        # Risk meter: the marker's left position is the probability as a percentage.
        st.markdown(
            f'<div class="meter"><div class="marker" style="left:{prob * 100:.1f}%"></div></div>'
            '<div class="meter-labels"><span>Safe</span><span>Suspicious</span><span>Dangerous</span></div>',
            unsafe_allow_html=True,
        )

        reds, greens, infos = explain(feats_dict, url.strip())
        st.subheader("Domain red flags")
        st.caption("Simple rule-based notes, shown alongside the model's score.")
        for r in reds:
            st.markdown(f'<div class="flag red">🚩 {html.escape(r)}</div>', unsafe_allow_html=True)
        for g in greens:
            st.markdown(f'<div class="flag green">👍 {html.escape(g)}</div>', unsafe_allow_html=True)
        for i in infos:
            st.markdown(f'<div class="flag gray">ℹ️ {html.escape(i)}</div>', unsafe_allow_html=True)

        with st.expander("🔬 All extracted features"):   # collapsible table of raw feature values
            st.dataframe(feats.T.rename(columns={0: "value"}), use_container_width=True)

        # Add this check to the top of the history list.
        st.session_state.history.insert(
            0, {"URL": url.strip()[:70], "Verdict": "Phishing" if is_bad else "Legitimate",
                "Phishing %": round(prob * 100, 1)}
        )

if st.session_state.history:                      # show the last 10 checks, if any
    st.subheader("🕘 Recent checks")
    st.dataframe(pd.DataFrame(st.session_state.history[:10]), use_container_width=True, hide_index=True)

st.caption("⚠️ This is a machine learning estimate, not a guarantee. "
           "Never enter passwords on a site you are unsure about.")