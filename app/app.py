import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import html

import pandas as pd
import streamlit as st

from admet.predict import Predictor, draw_molecule
from assays import ASSAYS, SOURCES

EXAMPLES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "bisphenol A": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    "DDT": "ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
}

TABLE_CSS = """
<style>
.res { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.res th, .res td { padding: 6px 10px; text-align: left; border-bottom: 1px solid rgba(128,128,128,0.25); }
.res th { font-weight: 600; opacity: 0.7; }
.bar { position: relative; height: 10px; width: 140px; background: rgba(128,128,128,0.2); border-radius: 5px; display: inline-block; vertical-align: middle; margin-right: 8px; }
.bar > span { position: absolute; left: 0; top: 0; bottom: 0; background: #ff4b4b; border-radius: 5px; }
.tt { position: relative; border-bottom: 1px dotted rgba(128,128,128,0.8); cursor: help; }
.tt .tip { visibility: hidden; opacity: 0; position: absolute; left: 0; top: 130%; z-index: 20; width: 340px;
           background: rgba(38,39,48,0.97); color: #fafafa; padding: 9px 11px; border-radius: 6px;
           font-size: 0.82rem; line-height: 1.4; box-shadow: 0 2px 8px rgba(0,0,0,0.3); transition: opacity 0.12s; }
.tt .tip .refs { display: block; margin-top: 5px; opacity: 0.7; font-size: 0.75rem; }
.tt:hover .tip { visibility: visible; opacity: 1; }
.res tr:nth-last-child(-n+3) .tt .tip { top: auto; bottom: 130%; }
</style>
"""


def render_results_table(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        info = ASSAYS.get(r["assay"], {})
        target = html.escape(info.get("target", ""))
        about = html.escape(info.get("about", ""))
        refs = ", ".join(f"[{i}]" for i in info.get("refs", []))
        tip = f'<span class="tip">{about}<span class="refs">sources: {refs}</span></span>' if about else ""
        p = r["p(active)"]
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(r['assay'])}</b></td>"
            f'<td><span class="tt">{target}{tip}</span></td>'
            f'<td><span class="bar"><span style="width:{p * 100:.0f}%"></span></span>{p:.3f}</td>'
            f"<td>{r['call']}</td>"
            "</tr>"
        )
    header = "<tr><th>assay</th><th>target (hover for details)</th><th>p(active)</th><th>call</th></tr>"
    return f'{TABLE_CSS}<table class="res">{header}{"".join(rows)}</table>'


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


st.set_page_config(page_title="Tox21 predictor", layout="wide")
st.title("Tox21 toxicity predictor")
st.write(
    "Paste a SMILES string and get the predicted probability of activity "
    "across the 12 Tox21 assays."
)

try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

example = st.selectbox("try an example", ["(type your own)"] + list(EXAMPLES))
default = EXAMPLES.get(example, "")
smiles = st.text_input("SMILES", value=default, key=f"smiles_{example}").strip()

if not smiles:
    st.info("enter a SMILES string above")
    st.stop()

try:
    img = draw_molecule(smiles)
    probs = predictor.predict(smiles)
except ValueError:
    st.error(f"could not parse SMILES: `{smiles}`")
    st.stop()

df = pd.DataFrame({"assay": list(probs), "p(active)": list(probs.values())})
df["call"] = df["p(active)"].apply(lambda p: "active" if p >= 0.5 else "inactive")
df = df.sort_values("p(active)", ascending=False).reset_index(drop=True)

left, right = st.columns([1, 2])
with left:
    st.image(img, caption=smiles)
    n_active = int((df["p(active)"] >= 0.5).sum())
    st.metric("assays flagged active", f"{n_active} / {len(df)}")

with right:
    st.bar_chart(df.set_index("assay")["p(active)"], horizontal=True)

st.markdown(render_results_table(df), unsafe_allow_html=True)

st.caption(
    f"Feedforward net on 2048-bit Morgan fingerprints. "
    f"Validation mean ROC-AUC {predictor.val_auc:.3f} (epoch {predictor.epoch}). "
    "Probabilities are raw sigmoid outputs, not calibrated confidence. "
    "Not for real safety decisions."
)

st.divider()
st.subheader("Sources")
st.caption(
    "Assay panel and PubChem AIDs follow [1]. Tooltip descriptions summarise the cited "
    "Tox21 papers plus standard toxicology background; they are educational, not a "
    "substitute for the primary literature."
)
for i, (citation, url) in SOURCES.items():
    st.markdown(f"[{i}] {citation} [{url}]({url})")
