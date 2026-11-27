
import pandas as pd
import streamlit as st
import url_features as uf

st.set_page_config(
    page_title="Detektor URL Phishing (PhiUSIIL)",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def get_bundle():
    return uf.load_bundle("phishing_model.joblib")


@st.cache_resource
def get_lookup():
    return uf.load_lookup("url_lookup.joblib")


bundle, lk = get_bundle(), get_lookup()
m = bundle["metrics"]

THRESHOLD = 0.50


# ==============================
# HEADER
# ==============================

st.title("🛡️ Detektor URL Phishing")

st.caption(
   "Tidak Mencakup Pemblokiran URL maupun Deteksi Phishing secara realtime."
)


# ==============================
# FUNGSI HASIL PREDIKSI
# ==============================

def show_result(row):

    p = float(row["proba_phishing"])

    if row["prediksi"] == "PHISHING":
        st.error(f"⚠️ **PHISHING** — Probabilitas {p:.1%}")
    else:
        st.success(
            f"✅ **LEGITIMATE** — "
            f"Probabilitas phishing {p:.1%}"
        )

    st.progress(min(max(p, 0.0), 1.0))


def show_comparison(row):

    prediksi = row["prediksi"]
    label = row["label_dataset"]
    sesuai = bool(row["sesuai"])

    st.markdown(f"**Prediksi Aplikasi :** {prediksi}")
    st.markdown(f"**Label Dataset :** {label}")

    if sesuai:
        st.markdown(
            '<p style="color:#16a34a; font-weight:bold;">'
            'PREDIKSI SESUAI ✅'
            '</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:#dc2626; font-weight:bold;">'
            'PREDIKSI TIDAK SESUAI ❌ '
            '</p>',
            unsafe_allow_html=True
        )


# ==============================
# TABS
# ==============================

tab1, tab2 = st.tabs([
    "Cek Satu URL",
    "Uji dengan URL Dataset PhiUSIIL"
])


# ==============================
# TAB 1: CEK SATU URL
# ==============================

with tab1:

    url = st.text_input(
        "Masukkan URL yang ada di dataset PhiUSIIL",
        placeholder="https://www.example.com"
    )

    if st.button("Periksa", type="primary") and url.strip():

        r = uf.predict_dataset_urls(
            [url],
            bundle,
            lk,
            THRESHOLD
        ).iloc[0]

        if not r["ditemukan"]:

            st.error(
                "URL tidak ditemukan di dataset PhiUSIIL, "
                "sehingga tidak dapat diproses."
            )

            sug = uf.suggest_urls(url, lk, 5)

            if sug:
                st.write("URL dataset yang mirip:")

                for s in sug:
                    st.code(s, language=None)

        else:

            show_result(r)

            show_comparison(r)

            with st.expander(
                "Lihat fitur yang diberikan ke model"
            ):

                i = lk["index"][r["url_dataset"]]

                st.dataframe(
                    pd.Series(
                        lk["features"][i],
                        index=lk["columns"]
                    )[
                        bundle["feature_names"]
                    ].rename("nilai").to_frame(),
                    use_container_width=True
                )


# ==============================
# TAB 2: UJI 20 URL ACAK
# ==============================

with tab2:

    st.write(
        "Pengujian menggunakan 20 URL acak dari "
        "**data uji PhiUSIIL**"
    )

    if st.button(
        "🎲 Uji 20 URL Acak",
        type="primary"
    ):

        out = uf.predict_dataset_urls(
            uf.sample_test_urls(lk, 20),
            bundle,
            lk,
            THRESHOLD
        )

        ok = out["sesuai"].astype(bool)

        st.metric(
            "Kesesuaian prediksi dengan label dataset",
            f"{ok.mean():.2%}",
            f"{int(ok.sum())}/{len(out)} URL"
        )

        view = out[
            [
                "url_dataset",
                "proba_phishing",
                "prediksi",
                "label_dataset",
                "sesuai"
            ]
        ].rename(
            columns={
                "url_dataset": "URL",
                "proba_phishing": "Probabilitas Phishing",
                "prediksi": "Prediksi Aplikasi",
                "label_dataset": "Label Dataset",
                "sesuai": "Prediksi Sesuai"
            }
        )

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True
        )

        # ==============================
        # RINCIAN KESESUAIAN
        # ==============================

        st.subheader("Hasil Perbandingan Prediksi")

        for _, row in out.iterrows():

            with st.expander(row["url_dataset"]):

                st.write(
                    f"Probabilitas Phishing: "
                    f"**{row['proba_phishing']:.2%}**"
                )

                show_comparison(row)

                i = lk["index"][row["url_dataset"]]

                with st.expander(
                    "Lihat fitur yang diberikan ke model"
                ):

                    st.dataframe(
                        pd.Series(
                            lk["features"][i],
                            index=lk["columns"]
                        )[
                            bundle["feature_names"]
                        ].rename("nilai").to_frame(),
                        use_container_width=True
                    )
