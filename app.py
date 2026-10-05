import streamlit as st
import joblib
from gensim.models import Word2Vec


st.set_page_config(
    page_title="Klasifikasi Berita Detik.com",
    page_icon="📰",
    layout="centered"
)


st.title("📰 Klasifikasi Berita Detik.com")

st.write(
    "Aplikasi ini menggunakan Word2Vec Skip-gram "
    "dan Naive Bayes untuk mengklasifikasikan berita "
    "ke dalam kategori Sport atau Finance."
)


@st.cache_resource
def load_models():

    model_w2v = Word2Vec.load(
        "w2v_skenario1_final.model"
    )

    model_nb = joblib.load(
        "naive_bayes_skenario1.pkl"
    )

    return model_w2v, model_nb


try:

    model_w2v, model_nb = load_models()

    st.success("Model berhasil dimuat.")

except Exception as e:

    st.error("Model gagal dimuat.")

    st.exception(e)


st.divider()

st.info(
    "Model Word2Vec dan Naive Bayes berhasil "
    "dihubungkan ke aplikasi."
)