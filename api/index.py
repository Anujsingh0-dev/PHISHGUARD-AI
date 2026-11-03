"""
api/index.py
────────────
Ultra-lightweight Vercel Serverless Function Handler for PHISHGUARD AI.

Key Design:
  - 100% Zero heavy external dependencies (no scikit-learn, no pandas, no scipy).
  - Uses direct mathematical traversal of the trained Decision Tree from tree_rules.json.
  - Complete Lambda size is <1 MB (well under AWS Lambda 500 MB limit).
  - Built-in cybersecurity pipeline: Canonicalization, Authority Check, Brand Impersonation.
"""

import sys
import json
import pathlib
import math
import ipaddress
import re
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

# Load pre-exported tree rules once at container cold-start
RULES_PATH = ROOT_DIR / "models" / "url_only_model" / "tree_rules.json"
with open(RULES_PATH, "r", encoding="utf-8") as f:
    TREE_RULES = json.load(f)

TOP_DOMAINS = {
    "google.com", "google.co.in", "youtube.com", "wikipedia.org", "amazon.com",
    "amazon.in", "github.com", "microsoft.com", "apple.com", "netflix.com",
    "facebook.com", "instagram.com", "linkedin.com", "twitter.com", "x.com",
    "reddit.com", "yahoo.com", "bing.com", "chase.com", "paypal.com",
    "cloudflare.com", "openai.com", "stackoverflow.com", "cgc.ac.in",
    "cgc.edu.in", "ptu.ac.in", "w3schools.com", "medium.com", "kaggle.com",
    "python.org", "scikit-learn.org", "streamlit.io", "zoom.us", "dropbox.com",
    "quora.com", "spotify.com", "canva.com", "adobe.com", "whatsapp.com",
}

BRAND_OFFICIAL_DOMAINS = {
    "paypal": "paypal.com",
    "apple": "apple.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "chase": "chase.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "bankofamerica": "bankofamerica.com",
    "wellsfargo": "wellsfargo.com",
    "github": "github.com",
    "dropbox": "dropbox.com",
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "account", "update", "verify", "secure",
    "banking", "paypal", "ebay", "amazon", "apple", "google",
    "microsoft", "confirm", "password", "credential", "free",
    "winner", "lucky", "prize", "click", "urgent", "alert",
    "wallet", "recover", "authenticate", "billing", "invoice",
]


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        h = hostname.split(":")[0].strip("[]")
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _normalize_url(url: str) -> str:
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        if path == "/":
            path = ""
        query = parsed.query
        return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")
    except Exception:
        return url


def _extract_features(url: str) -> dict:
    url = _normalize_url(url)
    try:
        parsed = urlparse(url)
    except Exception:
        return {name: 0 for name in TREE_RULES["features"]}

    raw_hostname = parsed.hostname or ""
    hostname = raw_hostname.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    scheme = parsed.scheme or ""

    domain_clean = hostname[4:] if hostname.startswith("www.") else hostname
    domain_length = len(domain_clean)

    num_digits = sum(1 for ch in domain_clean if ch.isdigit())
    digit_ratio = num_digits / domain_length if domain_length > 0 else 0.0

    special_chars = set("@!$%^&*")
    num_special_chars = sum(1 for ch in url if ch in special_chars)

    domain_parts = domain_clean.split(".")
    num_subdomains = max(0, len(domain_parts) - 2)

    try:
        query_params = parse_qs(query)
        num_query_params = len(query_params)
    except Exception:
        num_query_params = 0

    path_segments = [s for s in path.split("/") if s]
    url_lower = url.lower()
    suspicious_words_count = sum(1 for word in SUSPICIOUS_KEYWORDS if word in url_lower)

    return {
        "url_length": len(url),
        "domain_length": domain_length,
        "path_length": len(path),
        "num_dots": domain_clean.count("."),
        "num_hyphens": domain_clean.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": path.count("/"),
        "num_digits": num_digits,
        "digit_ratio": round(digit_ratio, 6),
        "num_special_chars": num_special_chars,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_double_slash": 1 if "//" in url[7:] else 0,
        "has_ip_as_host": 1 if _is_ip_address(hostname) else 0,
        "is_https": 1 if scheme == "https" else 0,
        "num_subdomains": num_subdomains,
        "num_query_params": num_query_params,
        "has_port": 1 if parsed.port is not None else 0,
        "has_percent_encoding": 1 if re.search(r"%[0-9A-Fa-f]{2}", url) else 0,
        "has_punycode": 1 if "xn--" in hostname else 0,
        "num_path_segments": len(path_segments),
        "suspicious_words_count": suspicious_words_count,
        "entropy": round(_shannon_entropy(domain_clean), 6),
    }


def _predict_pure_tree(features_dict: dict):
    # Step 1: Impute and standardize
    scaled = []
    for i, name in enumerate(TREE_RULES["features"]):
        val = features_dict.get(name, 0)
        mean = TREE_RULES["scaler_mean"][i]
        scale = TREE_RULES["scaler_scale"][i]
        scaled.append((val - mean) / scale if scale != 0 else 0.0)

    # Step 2: Traverse decision tree nodes
    node = 0
    while TREE_RULES["tree_feature"][node] != -2:
        feat_idx = TREE_RULES["tree_feature"][node]
        thresh = TREE_RULES["tree_threshold"][node]
        if scaled[feat_idx] <= thresh:
            node = TREE_RULES["tree_children_left"][node]
        else:
            node = TREE_RULES["tree_children_right"][node]

    val = TREE_RULES["tree_value"][node]
    total = sum(val)
    prob_0 = round(float(val[0] / total), 4) if total > 0 else 0.5
    prob_1 = round(float(val[1] / total), 4) if total > 0 else 0.5
    pred = 0 if prob_0 >= prob_1 else 1
    return pred, prob_0, prob_1


def _analyze_url(url: str) -> dict:
    url = url.strip()
    if not url:
        return {"error": "URL cannot be empty."}

    clean_url = _normalize_url(url)
    parsed = urlparse(clean_url)
    raw_hostname = parsed.hostname or ""
    hostname = raw_hostname.lower()

    parts = hostname[4:].split(".") if hostname.startswith("www.") else hostname.split(".")
    base_domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname

    features = _extract_features(clean_url)
    ml_pred, ml_p0, ml_p1 = _predict_pure_tree(features)

    reasons = []
    is_top = base_domain in TOP_DOMAINS
    is_ip = _is_ip_address(hostname)
    impersonation = None

    if not is_top and not is_ip:
        for brand, official in BRAND_OFFICIAL_DOMAINS.items():
            if brand in hostname and base_domain != official:
                impersonation = f"🚨 Brand Impersonation Alert: Domain references '{brand.upper()}' but registered domain is '{base_domain}', not '{official}'."
                reasons.append(impersonation)
                break

    if is_ip:
        reasons.append("🚨 Raw IP Address Host: Public websites use registered domains, not bare IP addresses.")

    if features["has_at_symbol"] == 1:
        reasons.append("⚠️ '@' Symbol in URL: Frequently used in phishing authority spoofing.")

    if features["has_punycode"] == 1:
        reasons.append("⚠️ Punycode Obfuscation (xn--): Internationalized domain commonly abused in lookalike attacks.")

    if features["suspicious_words_count"] >= 2:
        reasons.append(f"⚠️ High Suspicious Keywords: Contains {features['suspicious_words_count']} phishing trigger words.")

    if is_top and not impersonation and not is_ip:
        final_pred = 1
        label = "Legitimate"
        p0 = min(ml_p0, 0.05)
        p1 = round(1.0 - p0, 4)
        risk = "Safe / Verified Authority"
        reasons.append(f"✅ Verified Authority Domain: '{base_domain}' is an established high-reputation domain.")
    elif impersonation or is_ip:
        final_pred = 0
        label = "Phishing"
        p0 = max(ml_p0, 0.95)
        p1 = round(1.0 - p0, 4)
        risk = "High Risk (Phishing)"
    else:
        final_pred = ml_pred
        label = "Legitimate" if final_pred == 1 else "Phishing"
        p0 = ml_p0
        p1 = ml_p1
        risk = "Suspicious / Potential Phishing" if final_pred == 0 else "Low Risk / Predicted Legitimate"
        if not reasons:
            reasons.append("✅ URL structure is consistent with legitimate web domain patterns." if final_pred == 1 else "⚠️ Machine Learning structural analysis detected anomalies consistent with phishing URLs.")

    return {
        "prediction": final_pred,
        "prediction_label": label,
        "confidence": p1 if final_pred == 1 else p0,
        "phishing_prob": p0,
        "legitimate_prob": p1,
        "features": features,
        "url": clean_url,
        "reasons": reasons,
        "risk_level": risk,
        "base_domain": base_domain,
        "is_verified_top": is_top,
        "error": None,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        response_body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.endswith("/api/predict"):
            url_param = query.get("url", [""])[0]
            if not url_param:
                self._send_json(400, {"error": "Missing 'url' query parameter"})
                return
            result = _analyze_url(url_param)
            self._send_json(200, result)
        else:
            self._send_json(200, {"status": "ok", "app": "PHISHGUARD AI Serverless"})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            data = json.loads(body) if body else {}
            result = _analyze_url(data.get("url", ""))
            self._send_json(200, result)
        except Exception as e:
            self._send_json(400, {"error": str(e)})
