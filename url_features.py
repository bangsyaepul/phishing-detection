"""
url_features.py
================
Ekstraksi fitur untuk deteksi phishing. Dipakai oleh DUA pihak yang sama persis:
  1. notebook -> membentuk fitur leksikal, tabel referensi, dan kalibrasi
  2. app.py   -> mengubah URL masukan pengguna menjadi 50 fitur numerik (+ 14 leksikal)

Empat kelompok fitur:
  URL_FEATURES      (19) : fitur PhiUSIIL yang dihitung dari TEKS URL
  REF_FEATURES      (3)  : URLSimilarityIndex, TLDLegitimateProb, URLCharProb
                           -> butuh data referensi; di sini didekati dengan tabel
                              yang dibangun dari DATA LATIH (url_reference.json)
  HTML_FEATURES     (28) : fitur dari isi halaman web -> halaman DIAMBIL (fetch)
                           saat prediksi
  LEXICAL_FEATURES  (14) : fitur leksikal tambahan dari teks URL (khusus Skenario 2)

URL19 + REF3 + HTML28 = 50 fitur numerik dataset PhiUSIIL.
"""
import ipaddress
import json
import math
import os
import re
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Daftar fitur (nama kolom sama dengan PhiUSIIL, termasuk salah ejaan
# 'Degits' dan 'Spacial', agar mudah dibandingkan)
# --------------------------------------------------------------------------
URL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    "IsHTTPS", "CharContinuationRate",
]
REF_FEATURES = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb"]
HTML_FEATURES = [
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
]
LEXICAL_FEATURES = [
    "NoOfDots", "NoOfHyphens", "NoOfDotsInDomain", "NoOfHyphensInDomain",
    "PathLength", "QueryLength", "PathDepth",
    "URLEntropy", "DomainEntropy",
    "NoOfDigitsInDomain", "DigitRatioInDomain",
    "SuspiciousKeywordCount", "HasSuspiciousKeyword", "HasAtSymbol",
]
NUMERIC50 = URL_FEATURES + REF_FEATURES + HTML_FEATURES
ALL_URL_BASED_FEATURES = URL_FEATURES + LEXICAL_FEATURES      # yang murni dari teks URL
ALL_FEATURES = NUMERIC50 + LEXICAL_FEATURES                   # 64 fitur

SUSPICIOUS_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "password", "passwd", "banking", "webscr",
    "wallet", "support", "billing", "invoice", "suspend", "unlock", "recover",
    "authenticate", "security", "validate",
]
SOCIAL_DOMAINS = ("facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com",
                  "youtube.com", "pinterest.com", "tiktok.com", "whatsapp.com", "t.me",
                  "telegram.org", "reddit.com", "snapchat.com")

# --------------------------------------------------------------------------
# Konfigurasi varian definisi (dikalibrasi notebook terhadap kolom asli dataset)
# --------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "subdomain": "labels_minus_2",   # labels_minus_2 | dots | dots_minus_1 | labels_minus_2_no_www
    "ccr": "runs_sum",               # runs_sum | pair_ratio | longest_run
    "special": "all_non_alnum",      # all_non_alnum | excl_scheme_dot
    "special_ratio": "other_only",   # other_only | all_special
    "obfuscation": "seq_count",      # seq_count | seq_count_x3
    "tld_ip": "last_label",          # last_label | zero
    "urlcharprob": "legit",          # legit | all   (tabel probabilitas karakter)
    "title_match": "seqmatch",       # seqmatch | token | contains_sld
}
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_HERE, "url_features_config.json")
REF_FILE = os.path.join(_HERE, "url_reference.json")
CONFIG = dict(DEFAULT_CONFIG)
REF = {}                              # tabel referensi (diisi dari url_reference.json)


def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        try:
            with open(path) as f:
                CONFIG.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CONFIG})
        except Exception:
            pass
    return CONFIG


def save_config(cfg, path=CONFIG_FILE):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_reference(path=REF_FILE):
    if os.path.exists(path):
        try:
            with open(path) as f:
                REF.clear()
                REF.update(json.load(f))
        except Exception:
            pass
    return REF


def save_reference(ref, path=REF_FILE):
    with open(path, "w") as f:
        json.dump(ref, f)


load_config()
load_reference()

# --------------------------------------------------------------------------
# Parsing URL (manual, tidak melempar error pada URL yang aneh)
# --------------------------------------------------------------------------
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://(.*)$", re.S)
_IPV4_RE = re.compile(r"^(\d{1,3})(\.\d{1,3}){3}$")
_OBF_RE = re.compile(r"%[0-9a-fA-F]{2}")
_LETTER_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")


def normalize_url(url, assume_scheme="https://"):
    """Bersihkan spasi; tambahkan skema bila pengguna tidak mengetikkannya."""
    url = str(url).strip()
    if not _SCHEME_RE.match(url):
        url = assume_scheme + url.lstrip("/")
    return url


def split_url(url):
    """Pecah URL menjadi (scheme, host, path, query)."""
    url = str(url).strip()
    m = _SCHEME_RE.match(url)
    scheme, rest = (m.group(1).lower(), m.group(2)) if m else ("", url)
    cut = re.search(r"[/?#]", rest)
    authority = rest[:cut.start()] if cut else rest
    tail = rest[cut.start():] if cut else ""
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    if authority.startswith("["):                      # IPv6 literal
        end = authority.find("]")
        host = authority[:end + 1] if end != -1 else authority
    else:
        host = re.sub(r":\d*$", "", authority)
    q = tail.find("?")
    h = tail.find("#")
    path_end = min([i for i in (q, h) if i != -1], default=len(tail))
    path = tail[:path_end]
    query = ""
    if q != -1 and (h == -1 or q < h):
        query = tail[q + 1:(h if h != -1 else len(tail))]
    return scheme, host, path, query


def _entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _char_type(ch):
    if ch.isascii() and ch.isalpha():
        return "L"
    if ch.isdigit():
        return "D"
    return "S"


def _char_continuation(url, variant):
    n = len(url)
    if n == 0:
        return 0.0
    types = [_char_type(c) for c in url]
    if variant == "pair_ratio":
        if n == 1:
            return 1.0
        return sum(types[i] == types[i + 1] for i in range(n - 1)) / (n - 1)
    longest = {"L": 0, "D": 0, "S": 0}
    run = 1
    for i in range(1, n + 1):
        if i < n and types[i] == types[i - 1]:
            run += 1
        else:
            longest[types[i - 1]] = max(longest[types[i - 1]], run)
            run = 1
    if variant == "longest_run":
        return max(longest.values()) / n
    return sum(longest.values()) / n          # runs_sum


# --------------------------------------------------------------------------
# (1) 19 fitur URL + 14 fitur leksikal  (murni dari teks URL)
# --------------------------------------------------------------------------
def extract_features(url, cfg=None):
    """Hitung 19 fitur URL + 14 fitur leksikal dari satu string URL."""
    cfg = cfg or CONFIG
    url = str(url).strip()
    scheme, host, path, query = split_url(url)
    n = max(len(url), 1)

    is_ip = int(bool(_IPV4_RE.match(host)) or host.startswith("["))
    labels = [l for l in host.split(".") if l != ""]
    if is_ip and cfg["tld_ip"] == "zero":
        tld_len = 0
    else:
        tld_len = len(labels[-1]) if len(labels) > 1 or (labels and is_ip) else 0

    sub = cfg["subdomain"]
    if is_ip:
        n_sub = 0
    elif sub == "dots":
        n_sub = host.count(".")
    elif sub == "dots_minus_1":
        n_sub = max(host.count(".") - 1, 0)
    elif sub == "labels_minus_2_no_www":
        n_sub = max(len(labels) - (3 if host.lower().startswith("www.") else 2), 0)
    else:
        n_sub = max(len(labels) - 2, 0)

    n_obf = len(_OBF_RE.findall(url))
    if cfg["obfuscation"] == "seq_count_x3":
        n_obf *= 3
    n_letters = len(_LETTER_RE.findall(url))
    n_digits = len(_DIGIT_RE.findall(url))
    n_eq, n_q, n_amp = url.count("="), url.count("?"), url.count("&")
    n_other = len(url) - n_letters - n_digits - n_eq - n_q - n_amp
    if cfg["special"] == "excl_scheme_dot":
        n_other -= url.count(":") + url.count("/") + url.count(".")
    n_other = max(n_other, 0)
    special_total = n_other + (n_eq + n_q + n_amp if cfg["special_ratio"] == "all_special" else 0)

    f = {
        "URLLength": len(url),
        "DomainLength": len(host),
        "IsDomainIP": is_ip,
        "TLDLength": tld_len,
        "NoOfSubDomain": n_sub,
        "HasObfuscation": int(n_obf > 0),
        "NoOfObfuscatedChar": n_obf,
        "ObfuscationRatio": n_obf / n,
        "NoOfLettersInURL": n_letters,
        "LetterRatioInURL": n_letters / n,
        "NoOfDegitsInURL": n_digits,
        "DegitRatioInURL": n_digits / n,
        "NoOfEqualsInURL": n_eq,
        "NoOfQMarkInURL": n_q,
        "NoOfAmpersandInURL": n_amp,
        "NoOfOtherSpecialCharsInURL": n_other,
        "SpacialCharRatioInURL": special_total / n,
        "IsHTTPS": int(scheme == "https"),
        "CharContinuationRate": _char_continuation(url, cfg["ccr"]),
    }

    low = url.lower()
    host_digits = len(_DIGIT_RE.findall(host))
    kw_count = sum(1 for k in SUSPICIOUS_KEYWORDS if k in low)
    f.update({
        "NoOfDots": url.count("."),
        "NoOfHyphens": url.count("-"),
        "NoOfDotsInDomain": host.count("."),
        "NoOfHyphensInDomain": host.count("-"),
        "PathLength": len(path),
        "QueryLength": len(query),
        "PathDepth": len([s for s in path.split("/") if s]),
        "URLEntropy": _entropy(url),
        "DomainEntropy": _entropy(host),
        "NoOfDigitsInDomain": host_digits,
        "DigitRatioInDomain": host_digits / max(len(host), 1),
        "SuspiciousKeywordCount": kw_count,
        "HasSuspiciousKeyword": int(kw_count > 0),
        "HasAtSymbol": int("@" in url),
    })
    return f


def extract_dataframe(urls, cfg=None):
    """Ekstraksi 19 + 14 fitur untuk banyak URL -> DataFrame (urutan kolom tetap)."""
    rows = [extract_features(u, cfg) for u in urls]
    return pd.DataFrame(rows, columns=ALL_URL_BASED_FEATURES)


# --------------------------------------------------------------------------
# (2) 3 fitur referensi: pendekatan dari tabel yang dibangun dari DATA LATIH
# --------------------------------------------------------------------------
def _tld_of(url):
    host = split_url(url)[1]
    labels = [l for l in host.split(".") if l]
    return labels[-1].lower() if len(labels) > 1 else ""


def _strip_for_sim(url):
    u = _SCHEME_RE.sub(lambda m: m.group(2), str(url).strip().lower())
    return (u[4:] if u.startswith("www.") else u)[:200]


def build_reference(urls, tld_probs, is_phish, n_ref=2000, seed=42):
    """Bangun tabel referensi HANYA dari data latih (jangan masukkan data uji)."""
    urls = pd.Series(list(urls)).astype(str).reset_index(drop=True)
    probs = pd.Series(np.asarray(tld_probs, float))
    phish = np.asarray(is_phish).astype(bool)
    tlds = urls.map(_tld_of)
    tld_prob = {k: float(v) for k, v in probs.groupby(tlds.values).mean().items() if k != ""}
    tld_default = float(np.quantile(list(tld_prob.values()), 0.10)) if tld_prob else 0.0

    def char_table(sub):
        c = Counter("".join(sub))
        tot = max(sum(c.values()), 1)
        return {ch: v / tot for ch, v in c.items()}

    legit = urls[~phish]
    refs = legit.sample(min(n_ref, len(legit)), random_state=seed).map(_strip_for_sim).tolist()
    return {"tld_prob": tld_prob, "tld_default": tld_default,
            "char_prob_all": char_table(urls), "char_prob_legit": char_table(legit),
            "legit_refs": refs}


def _char_prob_mean(url, table):
    if not url or not table:
        return 0.0
    floor = min(table.values())
    return float(np.mean([table.get(c, floor) for c in url]))


def _similarity_index(url, refs):
    """0-100: kemiripan tertinggi URL dengan kumpulan URL legitimate referensi."""
    s = _strip_for_sim(url)
    if not s or not refs:
        return 0.0
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(s)
    best = 0.0
    for r in refs:
        sm.set_seq1(r)
        if sm.real_quick_ratio() > best and sm.quick_ratio() > best:
            best = max(best, sm.ratio())
            if best >= 1.0:
                break
    return 100.0 * best


def ref_features(url, cfg=None, needed=None):
    """3 fitur referensi; yang tidak dibutuhkan model (needed) dilewati agar cepat."""
    cfg = cfg or CONFIG
    if not REF:
        return {k: 0.0 for k in REF_FEATURES}
    want = lambda k: needed is None or k in needed
    table = REF["char_prob_legit"] if cfg.get("urlcharprob") == "legit" else REF["char_prob_all"]
    return {"URLSimilarityIndex": _similarity_index(url, REF["legit_refs"]) if want("URLSimilarityIndex") else 0.0,
            "TLDLegitimateProb": float(REF["tld_prob"].get(_tld_of(url), REF["tld_default"])),
            "URLCharProb": _char_prob_mean(url, table) if want("URLCharProb") else 0.0}


# --------------------------------------------------------------------------
# (3) 28 fitur HTML: halaman diambil lalu diparsing
# --------------------------------------------------------------------------
def _norm_host(host):
    h = (host or "").lower()
    return h[4:] if h.startswith("www.") else h


def _same_site(h1, h2):
    a, b = _norm_host(h1), _norm_host(h2)
    return bool(a) and bool(b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def title_scores(host, url, title, variant="seqmatch"):
    """(DomainTitleMatchScore, URLTitleMatchScore) dalam skala 0-100."""
    title = (title or "").strip().lower()[:300]
    if not title:
        return 0.0, 0.0
    d = _norm_host(host)[:300]
    u = _strip_for_sim(url)[:300]
    if variant == "token":
        toks = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 1]

        def sc(s):
            return 100.0 * sum(w in s for w in toks) / len(toks) if toks else 0.0
        return sc(d), sc(u)
    if variant == "contains_sld":
        labels = d.split(".")
        sld = labels[-2] if len(labels) >= 2 else labels[0]
        v = 100.0 if sld and sld in re.sub(r"[^a-z0-9]", "", title) else 0.0
        return v, v
    return (100.0 * SequenceMatcher(None, d, title).ratio(),
            100.0 * SequenceMatcher(None, u, title).ratio())


def html_features(html, base_url, orig_url, n_redirect=0, n_self_redirect=0, cfg=None):
    """Hitung 28 fitur HTML dari teks HTML halaman."""
    from bs4 import BeautifulSoup
    cfg = cfg or CONFIG
    soup = BeautifulSoup(html, "html.parser")
    lines = html.splitlines()
    low = html.lower()
    text = soup.get_text(" ", strip=True).lower()
    base_host = urlparse(base_url).hostname or ""
    orig_host = split_url(orig_url)[1]

    title = soup.title.get_text(strip=True) if soup.title else ""
    dts, uts = title_scores(orig_host, orig_url, title, cfg.get("title_match", "seqmatch"))

    def meta(name, need_content=False):
        for m in soup.find_all("meta"):
            if (m.get("name") or "").strip().lower() == name and (not need_content or (m.get("content") or "").strip()):
                return 1
        return 0

    def host_of(href):
        try:
            return urlparse(urljoin(base_url, href)).hostname or ""
        except ValueError:
            return ""

    n_self = n_empty = n_ext = 0
    has_social = 0
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        hl = href.lower()
        if href == "" or href.startswith("#") or hl.startswith(("javascript:", "void(")):
            n_empty += 1
            continue
        if hl.startswith(("mailto:", "tel:")):
            continue
        h = host_of(href)
        if _same_site(h, base_host):
            n_self += 1
        else:
            n_ext += 1
            if any(h.lower().endswith(s) for s in SOCIAL_DOMAINS):
                has_social = 1

    ext_form = 0
    for fm in soup.find_all("form"):
        act = (fm.get("action") or "").strip()
        if act and not act.startswith("#") and not act.lower().startswith("javascript:"):
            h = host_of(act)
            if h and not _same_site(h, base_host):
                ext_form = 1

    def input_type(t):
        return int(any((i.get("type") or "").strip().lower() == t for i in soup.find_all("input")))

    has_submit = int(any((i.get("type") or "").strip().lower() in ("submit", "image") for i in soup.find_all("input"))
                     or any((b.get("type") or "submit").strip().lower() == "submit" for b in soup.find_all("button")))
    rels = [" ".join(l.get("rel") or []).lower() for l in soup.find_all("link")]

    return {
        "LineOfCode": len(lines),
        "LargestLineLength": max((len(l) for l in lines), default=0),
        "HasTitle": int(bool(title)),
        "DomainTitleMatchScore": dts,
        "URLTitleMatchScore": uts,
        "HasFavicon": int(any("icon" in r for r in rels)),
        "Robots": meta("robots"),
        "IsResponsive": meta("viewport"),
        "NoOfURLRedirect": int(n_redirect),
        "NoOfSelfRedirect": int(n_self_redirect),
        "HasDescription": meta("description", need_content=True),
        "NoOfPopup": len(re.findall(r"window\.open\s*\(|\balert\s*\(", low)),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": ext_form,
        "HasSocialNet": has_social,
        "HasSubmitButton": has_submit,
        "HasHiddenFields": input_type("hidden"),
        "HasPasswordField": input_type("password"),
        "Bank": int(bool(re.search(r"\bbank", text))),
        "Pay": int(bool(re.search(r"\bpay", text))),
        "Crypto": int(bool(re.search(r"crypto|bitcoin|ethereum|blockchain|\bbtc\b", text))),
        "HasCopyrightInfo": int("\u00a9" in html or "&copy;" in low or "copyright" in low),
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": sum("stylesheet" in r for r in rels),
        "NoOfJS": len(soup.find_all("script", src=True)),
        "NoOfSelfRef": n_self,
        "NoOfEmptyRef": n_empty,
        "NoOfExternalRef": n_ext,
    }


_UA = {"User-Agent": "Mozilla/5.0 (compatible; PhishDetectorResearch/1.0)", "Accept": "text/html,*/*;q=0.8"}


def _safe_target(url, allow_private=False):
    """Cegah aplikasi dipakai untuk mengakses alamat internal (SSRF)."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, "skema bukan http/https"
        if not p.hostname:
            return False, "host kosong"
        if allow_private:
            return True, ""
        port = p.port or (443 if p.scheme == "https" else 80)
        for info in socket.getaddrinfo(p.hostname, port, proto=socket.IPPROTO_TCP):
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified):
                return False, "alamat internal/privat diblokir"
        return True, ""
    except Exception as e:
        return False, f"DNS/URL gagal ({e.__class__.__name__})"


def fetch_page(url, timeout=8, max_redirects=5, max_bytes=2_000_000, allow_private=False):
    """Ambil HTML halaman; redirect diikuti manual agar tiap hop diperiksa."""
    import requests
    cur, hops = url, []

    def fail(msg):
        return {"ok": False, "error": msg, "n_redirect": len(hops), "n_self_redirect": 0}
    try:
        for _ in range(max_redirects + 1):
            ok, why = _safe_target(cur, allow_private)
            if not ok:
                return fail(why)
            r = requests.get(cur, headers=_UA, timeout=timeout, allow_redirects=False, stream=True)
            try:
                loc = r.headers.get("location")
                if r.status_code in (301, 302, 303, 307, 308) and loc:
                    nxt = urljoin(cur, loc)
                    hops.append((cur, nxt))
                    cur = nxt
                    continue
                if r.status_code >= 400:
                    return fail(f"HTTP {r.status_code}")
                buf = bytearray()
                for chunk in r.iter_content(16384):
                    buf.extend(chunk)
                    if len(buf) >= max_bytes:
                        break
                try:
                    html = bytes(buf).decode(r.encoding or "utf-8", errors="replace")
                except LookupError:
                    html = bytes(buf).decode("utf-8", errors="replace")
                n_self = sum(_same_site(urlparse(a).hostname, urlparse(b).hostname) for a, b in hops)
                return {"ok": True, "html": html, "final_url": cur, "n_redirect": len(hops),
                        "n_self_redirect": n_self, "error": ""}
            finally:
                r.close()
        return fail("terlalu banyak redirect")
    except Exception as e:
        return fail(e.__class__.__name__)


# --------------------------------------------------------------------------
# Ekstraksi lengkap (64 fitur) & prediksi
# --------------------------------------------------------------------------
def extract_all(url, fetch=True, timeout=8, allow_private=False, cfg=None, needed=None):
    """URL mentah -> (dict 64 fitur, info halaman).
    needed = nama fitur yang dipakai model; halaman hanya diambil bila ada fitur HTML di dalamnya.
    Jika halaman gagal diambil, fitur HTML = 0."""
    cfg = cfg or CONFIG
    u = normalize_url(url)
    need_html = needed is None or bool(set(needed) & set(HTML_FEATURES))
    f = extract_features(u, cfg)
    f.update(ref_features(u, cfg, needed))
    if fetch and need_html:
        page = fetch_page(u, timeout, allow_private=allow_private)
    else:
        page = {"ok": False, "error": "" if not need_html else "halaman tidak diambil",
                "n_redirect": 0, "n_self_redirect": 0}
    if page["ok"]:
        try:
            f.update(html_features(page["html"], page["final_url"], u, page["n_redirect"], page["n_self_redirect"], cfg))
        except Exception as e:                       # HTML rusak -> perlakukan sebagai gagal
            page = {"ok": False, "error": f"parsing HTML gagal ({e.__class__.__name__})",
                    "n_redirect": 0, "n_self_redirect": 0}
    if not page["ok"]:
        f.update({k: 0 for k in HTML_FEATURES})
    info = {"html_dibutuhkan": need_html, "html_tersedia": bool(page["ok"]), "catatan": page.get("error", ""),
            "final_url": page.get("final_url", u)}
    return f, info


def load_bundle(path="phishing_model.joblib"):
    import joblib
    return joblib.load(path)


def predict_features(X, bundle, threshold=0.5):
    """Prediksi dari tabel fitur yang sudah jadi (mis. fitur asli dataset)."""
    X = X[bundle["feature_names"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    return pd.DataFrame({"proba_phishing": proba,
                         "prediksi": np.where(proba >= threshold, "PHISHING", "LEGITIMATE")}, index=X.index)


def predict_one(url, bundle, threshold=0.5, fetch=True, timeout=8, allow_private=False):
    f, info = extract_all(url, fetch=fetch, timeout=timeout, allow_private=allow_private,
                          needed=set(bundle["feature_names"]))
    X = pd.DataFrame([f])[bundle["feature_names"]]
    proba = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    return {"url": str(url), "proba_phishing": proba,
            "prediksi": "PHISHING" if proba >= threshold else "LEGITIMATE",
            "fitur": X.iloc[0], **info}


def predict_urls(urls, bundle, threshold=0.5, fetch=True, timeout=8, workers=8, allow_private=False):
    """Banyak URL mentah -> DataFrame [url, proba_phishing, prediksi, html_tersedia, catatan]."""
    urls = [str(u) for u in urls]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(lambda u: predict_one(u, bundle, threshold, fetch, timeout, allow_private), urls))
    return pd.DataFrame([{k: r[k] for k in ("url", "proba_phishing", "prediksi", "html_dibutuhkan", "html_tersedia", "catatan")} for r in res])


# --------------------------------------------------------------------------
# Jalur APLIKASI: URL dicari di dataset PhiUSIIL, lalu dipakai NILAI FITUR ASLI dataset
# (sama persis dengan yang dipakai model saat training & evaluasi di notebook)
# --------------------------------------------------------------------------
LOOKUP_FILE = os.path.join(_HERE, "url_lookup.joblib")


def build_lookup(urls, X, y, is_test, path=LOOKUP_FILE):
    """Simpan tabel URL -> fitur asli (float64, tanpa pembulatan) + label + penanda data uji."""
    import joblib
    obj = {"urls": [str(u) for u in urls], "columns": list(X.columns),
           "features": np.ascontiguousarray(X.to_numpy(dtype=np.float64)),
           "label": np.asarray(y).astype(np.int8), "is_test": np.asarray(is_test).astype(bool)}
    joblib.dump(obj, path, compress=3)
    return obj


def load_lookup(path=LOOKUP_FILE):
    import joblib
    lk = joblib.load(path)
    lk["index"] = {u: i for i, u in enumerate(lk["urls"])}
    return lk


def _candidates(q):
    q = str(q).strip()
    c = [q]
    if q and not _SCHEME_RE.match(q):                      # pengguna tidak mengetik skema
        for sch in ("https://", "http://"):
            c += [sch + q, sch + "www." + q]
    c += [x + "/" for x in c if not x.endswith("/")] + [x.rstrip("/") for x in c if x.endswith("/")]
    return list(dict.fromkeys(x for x in c if x))


def find_url(q, lk):
    """Indeks baris URL di dataset (pencocokan persis; skema/garis miring akhir dicoba otomatis) atau None."""
    for cand in _candidates(q):
        i = lk["index"].get(cand)
        if i is not None:
            return i
    return None


def suggest_urls(q, lk, n=5):
    """URL dataset yang mengandung teks yang diketik (untuk saran jika tidak ditemukan)."""
    ql = str(q).strip().lower()
    if len(ql) < 4:
        return []
    out = []
    for u in lk["urls"]:
        if ql in u.lower():
            out.append(u)
            if len(out) >= n:
                break
    return out


def predict_dataset_urls(urls, bundle, lk, threshold=0.5):
    """SATU-SATUNYA jalur prediksi aplikasi (tab 'Cek satu URL' dan tab uji memakai fungsi ini).
    Kolom: url_input, url_dataset, ditemukan, proba_phishing, prediksi, label_dataset, sesuai, data."""
    urls = [str(u) for u in urls]
    idx = [find_url(u, lk) for u in urls]
    found = np.array([i is not None for i in idx])
    res = pd.DataFrame({"url_input": urls, "url_dataset": None, "ditemukan": found, "proba_phishing": np.nan,
                        "prediksi": None, "label_dataset": None, "sesuai": None, "data": None}, dtype=object)
    if found.any():
        rows = np.array([i for i in idx if i is not None])
        X = pd.DataFrame(lk["features"][rows], columns=lk["columns"])[bundle["feature_names"]]
        proba = bundle["pipeline"].predict_proba(X)[:, 1]
        pred = np.where(proba >= threshold, "PHISHING", "LEGITIMATE")
        lab = np.where(lk["label"][rows] == 1, "PHISHING", "LEGITIMATE")
        pos = np.where(found)[0]
        res.loc[pos, "url_dataset"] = [lk["urls"][i] for i in rows]
        res.loc[pos, "proba_phishing"] = proba
        res.loc[pos, "prediksi"] = pred
        res.loc[pos, "label_dataset"] = lab
        res.loc[pos, "sesuai"] = pred == lab
        res.loc[pos, "data"] = np.where(lk["is_test"][rows], "uji", "latih")
    res["proba_phishing"] = res["proba_phishing"].astype(float)
    return res


def sample_test_urls(lk, n=50, seed=None):
    """Ambil n URL acak dari DATA UJI dataset."""
    rng = np.random.RandomState(seed)
    pool = np.where(lk["is_test"])[0]
    return [lk["urls"][i] for i in rng.choice(pool, min(n, len(pool)), replace=False)]
