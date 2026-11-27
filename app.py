
import pandas as pd
import streamlit as st
import url_features as uf


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Detektor URL Phishing (PhiUSIIL)",
    page_icon="🛡️",
    layout="centered"
)


# ==========================================
# LOAD MODEL DAN DATASET
# ==========================================

@st.cache_resource
def get_bundle():
    return uf.load_bundle("phishing_model.joblib")


@st.cache_resource
def get_lookup():
    return uf.load_lookup("url_lookup.joblib")


bundle, lk = get_bundle(), get_lookup()

m = bundle["metrics"]

THRESHOLD = 0.50


# ==========================================
# HEADER APLIKASI
# ==========================================

st.title("🛡️ Detektor URL Phishing")

st.caption(
    f"Model terbaik: **{bundle['model_name']}** · "
    f"Fitur: **{bundle['feature_set']}** · "
    f"{bundle['seleksi']} · "
    f"{bundle['n_features_used']} fitur · "
    f"F1 uji {m['f1']:.4f} · "
    f"Akurasi uji {m['accuracy']:.4f}"
)


# ==========================================
# FUNGSI MENAMPILKAN HASIL PREDIKSI
# ==========================================

def show_result(row):

    p = float(row["proba_phishing"])

    if row["prediksi"] == "PHISHING":

        st.error(
            f"⚠️ **PHISHING** — "
            f"Probabilitas {p:.1%}"
        )

    else:

        st.success(
            f"✅ **LEGITIMATE** — "
            f"Probabilitas phishing {p:.1%}"
        )

    st.progress(
        min(max(p, 0.0), 1.0)
    )


# ==========================================
# FUNGSI PERBANDINGAN PREDIKSI
# HANYA DIGUNAKAN DI TAB CEK SATU URL
# ==========================================

def show_comparison(row):

    prediksi = row["prediksi"]
    label = row["label_dataset"]
    sesuai = bool(row["sesuai"])

    st.markdown(
        f"**Prediksi Aplikasi :** {prediksi}"
    )

    st.markdown(
        f"**Label Dataset :** {label}"
    )

    if sesuai:

        st.markdown(
            '<p style="color:#16a34a; font-weight:bold;">'
            '✅ PREDIKSI SESUAI'
            '</p>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<p style="color:#dc2626; font-weight:bold;">'
            '❌ PREDIKSI TIDAK SESUAI'
            '</p>',
            unsafe_allow_html=True
        )


# ==========================================
# TAB NAVIGASI
# ==========================================

tab1, tab2 = st.tabs([
    "Cek Satu URL",
    "Uji dengan URL Dataset PhiUSIIL"
])


# ==========================================
# TAB 1: CEK SATU URL
# ==========================================

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

        # URL tidak ditemukan
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

        # URL ditemukan
        else:

            # Hasil prediksi
            show_result(r)

            # Perbandingan prediksi dengan label dataset
            show_comparison(r)

            # Fitur yang digunakan model
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


# ==========================================
# TAB 2: UJI 20 URL ACAK
# ==========================================

with tab2:

    st.write(
        "Pengujian menggunakan 20 URL acak dari "
        "**data uji PhiUSIIL** yang tidak digunakan "
        "saat melatih model."
    )

    if st.button(
        "🎲 Uji 20 URL Acak",
        type="primary"
    ):

        # Mengambil 20 URL acak dari data uji
        out = uf.predict_dataset_urls(
            uf.sample_test_urls(lk, 20),
            bundle,
            lk,
            THRESHOLD
        )

        # Menghitung kesesuaian prediksi
        ok = out["sesuai"].astype(bool)

        # Menampilkan metrik pengujian
        st.metric(
            "Kesesuaian prediksi dengan label dataset",
            f"{ok.mean():.2%}",
            f"{int(ok.sum())}/{len(out)} URL"
        )

        # Menampilkan tabel hasil pengujian
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
