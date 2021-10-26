import streamlit as st
import viz_interface as interface
from bokeh.models import Label

import pandas as pd
import numpy as np
import scipy.special
from slugify import slugify
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans

project_title = "R Friends demo, AG news dataset"

# Display options
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title=project_title,
)


@st.experimental_memo
def load_raw_data(load_dest):

    load_dest = Path(load_dest)

    f_umap = load_dest / "umap.npy"

    assert f_umap.exists()
    umap = np.load(f_umap)
    umap -= umap.mean(axis=0)
    umap /= umap.std(axis=0)

    df = pd.read_csv(load_dest / "dataset.csv")
    df["ux"], df["uy"] = umap[:, 0], umap[:, 1]

    df = df[:]
    umap = umap[:]

    return df, umap


@st.experimental_memo
def cluster_data(umap, n_clusters):

    clf = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000)
    clusters = clf.fit_predict(umap)
    return clusters

    return clusters


@st.experimental_memo
def compute_keywords(df):

    import yake

    n_keywords = 2
    kw_extractor = yake.KeywordExtractor()
    custom_kw_extractor = yake.KeywordExtractor(
        lan="en", n=2, dedupLim=0.9, top=n_keywords, features=None
    )

    keywords = {}
    for i, dx in df.groupby(df["cluster"]):
        text = "\n".join(dx["text"].values)
        kw = [x[0] for x in custom_kw_extractor.extract_keywords(text)]
        keywords[i] = "; ".join(kw)
    return keywords


@st.experimental_memo
def load_model_and_vectors(load_dest):
    from transformers import AutoModel, AutoTokenizer

    load_dest = Path(load_dest)

    f_vectors = load_dest / "embeddings.npy"
    V = np.load(f_vectors)

    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    return tokenizer, model, V


def compute_distance(text):

    tokens = tokenizer([text], padding=True, truncation=True, return_tensors="pt")

    output = model(**tokens, output_hidden_states=True, return_dict=True)
    z = output.pooler_output.detach().cpu().numpy().ravel()

    dist = V.dot(z)

    from scipy.spatial.distance import cdist

    dist = cdist(V, [z], metric="cosine").ravel()

    return dist


def convert_plot_for_download(p):
    from bokeh.io import save
    import tempfile

    with tempfile.NamedTemporaryFile() as FOUT:
        save(p, title=project_title, filename=FOUT.name)
        FOUT.flush()

        with open(FOUT.name, "rb") as FIN:
            raw_bytes = FIN.read()

    return raw_bytes


# Fixed colors for the labeling (taken from ColorBrewer)
colors = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
] * 1000


df_org, umap = load_raw_data('data')
df = df_org.copy()
df["fill_color"] = df["color"] = "#3288bd"

show_colors = st.sidebar.checkbox("Color by label", value=False)

if show_colors:
    df["fill_color"] = [colors[i] if i >= 0 else "black" for i in df.label]
    st.sidebar.write("World (Red), Sports (Blue), Business (Green), Sci/Tech (Purple)")

radius = st.sidebar.slider("Point Radius", 0.005, 0.1, value=0.02, step=0.001)

show_labels = st.sidebar.checkbox("Show text labels", value=False)
if show_labels:
    n_text_labels = st.sidebar.slider("Number of text labels", 1, 50, value=10, step=1)
    df["cluster"] = cluster_data(umap, n_text_labels)
    keywords = compute_keywords(df)

show_search = st.sidebar.checkbox("Search by phrase", value=False)

if show_search:
    search_query = st.sidebar.text_input("Enter search terms", value="Baseball")
    tokenizer, model, V = load_model_and_vectors('data')
    df["dist"] = compute_distance(search_query)

df["size"] = radius
df["line_width"] = 0


viz_cols = [
    "text",
    "label",
]

st.header(project_title)

if show_search:
    subset = df.sort_values("dist")[["text", "dist"]]
    st.table(subset.head())

p = interface.plot_data_bokeh(df, hover_columns=viz_cols)
plot_placeholder = st.empty()


if show_labels:

    for col in range(n_text_labels):
        dx = df[df.cluster == col]
        dx = dx.reset_index()

        cmx, cmy = dx.ux.mean(), dx.uy.mean()

        if not len(dx):
            continue

        if not show_search:
            st.markdown(f"### {keywords[col]}")
            st.table(dx["text"].head())

        label_args = {
            "x": cmx,
            "y": cmy,
            "text": keywords[col],
            "text_color": "black",
            "background_fill_color": "white",
            "background_fill_alpha": 0.25,
        }
        p.add_layout(Label(**label_args))


plot_placeholder.bokeh_chart(p)


st.sidebar.download_button(
    label="📥 export plot HTML",
    data=convert_plot_for_download(p),
    file_name="streamlit_plot.html",
)
