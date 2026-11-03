"""
src/predict.py
───────────────
Prediction module for PHISHGUARD AI.

This module implements a defense-in-depth cybersecurity classification pipeline:
  1. Input Validation & Canonicalization (URL normalization)
  2. Verified Authority / High-Reputation Domain Check (Whitelisting layer)
  3. Brand Impersonation & Typosquatting Detection (Heuristic layer)
  4. IP Host & Protocol Security Check (Transport layer)
  5. Machine Learning Model Inference (Random Forest / Decision Tree)

This ensures live real-world URLs (e.g., google.com, github.com) are properly
recognized, while malicious phishing URLs (e.g. paypal-security-update.com)
are caught with clear, explainable diagnostic evidence.
"""

import sys
import pathlib
import re
import ipaddress
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.utils import (
    FULL_MODEL_DIR, URL_MODEL_DIR, label_to_text,
)
from src.feature_extraction import (
    extract_url_features, features_to_ordered_list, URL_FEATURE_NAMES,
    normalize_url, _is_ip_address,
)

# ── Module-level caches ────────────────────────────────────────────────────────
_url_model          = None
_url_preprocessor   = None
_full_model         = None
_full_preprocessor  = None
_full_feature_names = None

# Top verified global apex domains (Authority Knowledge Base)
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

# Major brands commonly targeted by phishing attackers
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


def _load_url_model():
    """Load URL model and preprocessor into module-level cache."""
    global _url_model, _url_preprocessor

    if _url_model is None:
        model_path = URL_MODEL_DIR / "url_model.pkl"
        prep_path  = URL_MODEL_DIR / "url_preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"URL model not found: {model_path}\n"
                f"Run first: python src/train_url_model.py"
            )
        if not prep_path.exists():
            raise FileNotFoundError(
                f"URL preprocessor not found: {prep_path}\n"
                f"Run first: python src/train_url_model.py"
            )

        _url_model        = joblib.load(model_path)
        _url_preprocessor = joblib.load(prep_path)

    return _url_model, _url_preprocessor


def _load_full_model():
    """Load full-dataset model, preprocessor, and feature names into cache."""
    global _full_model, _full_preprocessor, _full_feature_names

    if _full_model is None:
        model_path = FULL_MODEL_DIR / "best_model.pkl"
        prep_path  = FULL_MODEL_DIR / "preprocessor.pkl"
        feat_path  = FULL_MODEL_DIR / "feature_names.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Full model not found: {model_path}\n"
                f"Run first: python src/train_full_model.py"
            )
        if not prep_path.exists():
            raise FileNotFoundError(
                f"Full preprocessor not found: {prep_path}\n"
                f"Run first: python src/preprocessing.py"
            )

        _full_model         = joblib.load(model_path)
        _full_preprocessor  = joblib.load(prep_path)
        _full_feature_names = joblib.load(feat_path) if feat_path.exists() else []

    return _full_model, _full_preprocessor, _full_feature_names


def _extract_base_domain(hostname: str) -> str:
    """Extract registered second-level apex domain (e.g. google.com from mail.google.com)."""
    h = hostname.lower().strip()
    if h.startswith("www."):
        h = h[4:]
    parts = h.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return h


def _basic_url_validation(url: str) -> Tuple[bool, str]:
    """Basic validation of user input."""
    url = url.strip()

    if not url:
        return False, "URL cannot be empty. Please enter a URL to analyse."

    if len(url) < 3:
        return False, f"URL is too short ({len(url)} characters). Please enter a valid URL."

    if "." not in url and not re.match(r"^\d{1,3}(\.\d{1,3}){3}", url):
        return False, "URL does not appear valid — it must contain a domain name with a dot."

    if any(ch in url for ch in ["\n", "\r", "\t"]):
        return False, "URL contains invalid whitespace characters."

    return True, ""


def predict_url(url: str) -> dict:
    """
    Predict whether a URL is phishing or legitimate using multi-layered ML and heuristics.

    Parameters
    ----------
    url : str — The raw URL entered by the user.

    Returns
    -------
    dict with structured diagnostic information.
    """
    is_valid, error_msg = _basic_url_validation(url)
    if not is_valid:
        return {
            "prediction"       : None,
            "prediction_label" : None,
            "confidence"       : None,
            "phishing_prob"    : None,
            "legitimate_prob"  : None,
            "features"         : {},
            "url"              : url,
            "error"            : error_msg,
            "reasons"          : [error_msg],
            "risk_level"       : "Unknown",
        }

    clean_url = normalize_url(url)

    try:
        model, preprocessor = _load_url_model()

        # Step 1: Extract features identically to training
        features_dict = extract_url_features(clean_url)
        feature_values = features_to_ordered_list(features_dict)
        X = np.array(feature_values).reshape(1, -1)

        # Step 2: Machine Learning inference
        X_proc = preprocessor.transform(X)
        ml_pred = int(model.predict(X_proc)[0])

        ml_phish_prob = 0.5
        ml_legit_prob = 0.5
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_proc)[0]
            ml_phish_prob = round(float(proba[0]), 4)
            ml_legit_prob = round(float(proba[1]), 4)

        # Step 3: Cybersecurity Domain & Heuristic Analysis
        parsed = urlparse(clean_url)
        raw_hostname = parsed.hostname or ""
        hostname = raw_hostname.lower()
        base_domain = _extract_base_domain(hostname)

        reasons = []
        is_verified_top = base_domain in TOP_DOMAINS
        is_ip_host = _is_ip_address(hostname)
        impersonated_brand = None

        # Check brand impersonation
        if not is_verified_top and not is_ip_host:
            for brand, official_domain in BRAND_OFFICIAL_DOMAINS.items():
                if brand in hostname and base_domain != official_domain:
                    impersonated_brand = brand
                    reasons.append(
                        f"🚨 Brand Impersonation Alert: Domain references '{brand.upper()}' "
                        f"but registered domain is '{base_domain}', not '{official_domain}'."
                    )
                    break

        if is_ip_host:
            reasons.append("🚨 Raw IP Address Host: Legitimate public websites use domain names, not bare IPs.")

        if features_dict.get("has_at_symbol") == 1:
            reasons.append("⚠️ '@' Symbol in URL: Often used to confuse users about the actual destination domain.")

        if features_dict.get("has_punycode") == 1:
            reasons.append("⚠️ Punycode Obfuscation (xn--): Internationalized domain commonly used for homograph attacks.")

        if features_dict.get("suspicious_words_count", 0) >= 2:
            reasons.append(f"⚠️ High Suspicious Keywords: Contains {features_dict['suspicious_words_count']} phishing trigger words.")

        # Step 4: Multi-Layer Decision Fusion
        if is_verified_top and not impersonated_brand and not is_ip_host:
            # Verified Authority Domain (e.g. google.com, github.com, wikipedia.org)
            final_pred = 1
            final_label = "Legitimate"
            phishing_prob = min(ml_phish_prob, 0.05)
            legitimate_prob = 1.0 - phishing_prob
            confidence = legitimate_prob
            risk_level = "Safe / Verified Authority"
            reasons.append(f"✅ Verified Authority Domain: '{base_domain}' is an established high-reputation domain.")

        elif impersonated_brand or is_ip_host:
            # High-confidence malicious indicator
            final_pred = 0
            final_label = "Phishing"
            phishing_prob = max(ml_phish_prob, 0.95)
            legitimate_prob = round(1.0 - phishing_prob, 4)
            confidence = phishing_prob
            risk_level = "High Risk (Phishing)"

        else:
            # General ML classification
            final_pred = ml_pred
            final_label = label_to_text(final_pred)
            phishing_prob = ml_phish_prob
            legitimate_prob = ml_legit_prob
            confidence = round(float(ml_phish_prob if final_pred == 0 else ml_legit_prob), 4)

            if final_pred == 0:
                risk_level = "Suspicious / Potential Phishing"
                if not reasons:
                    reasons.append("⚠️ Machine Learning structural analysis detected anomalies consistent with phishing URLs.")
            else:
                risk_level = "Low Risk / Predicted Legitimate"
                if not reasons:
                    reasons.append("✅ URL structure is consistent with legitimate web domain patterns.")

        return {
            "prediction"       : final_pred,
            "prediction_label" : final_label,
            "confidence"       : round(float(confidence), 4),
            "phishing_prob"    : round(float(phishing_prob), 4),
            "legitimate_prob"  : round(float(legitimate_prob), 4),
            "features"         : features_dict,
            "url"              : clean_url,
            "error"            : None,
            "reasons"          : reasons,
            "risk_level"       : risk_level,
            "base_domain"      : base_domain,
            "is_verified_top"  : is_verified_top,
        }

    except FileNotFoundError as exc:
        return {
            "prediction"       : None,
            "prediction_label" : None,
            "confidence"       : None,
            "phishing_prob"    : None,
            "legitimate_prob"  : None,
            "features"         : {},
            "url"              : clean_url,
            "error"            : f"Model not loaded: {exc}",
            "reasons"          : [str(exc)],
            "risk_level"       : "Error",
        }
    except Exception as exc:
        return {
            "prediction"       : None,
            "prediction_label" : None,
            "confidence"       : None,
            "phishing_prob"    : None,
            "legitimate_prob"  : None,
            "features"         : {},
            "url"              : clean_url,
            "error"            : f"Prediction error: {exc}",
            "reasons"          : [str(exc)],
            "risk_level"       : "Error",
        }


def predict_from_features(feature_dict: dict) -> dict:
    """Predict using the FULL dataset model (MODE A — Research Demo)."""
    try:
        model, preprocessor, feature_names = _load_full_model()
        df = pd.DataFrame([feature_dict])

        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            return {
                "prediction"       : None,
                "prediction_label" : None,
                "confidence"       : None,
                "error"            : f"Missing features: {missing}",
            }

        X = df[feature_names]
        X_proc = preprocessor.transform(X)
        prediction = int(model.predict(X_proc)[0])

        phishing_prob   = None
        legitimate_prob = None
        confidence      = None

        if hasattr(model, "predict_proba"):
            proba           = model.predict_proba(X_proc)[0]
            phishing_prob   = round(float(proba[0]), 4)
            legitimate_prob = round(float(proba[1]), 4)
            confidence      = round(float(proba[prediction]), 4)

        return {
            "prediction"       : prediction,
            "prediction_label" : label_to_text(prediction),
            "confidence"       : confidence,
            "phishing_prob"    : phishing_prob,
            "legitimate_prob"  : legitimate_prob,
            "error"            : None,
        }

    except Exception as exc:
        return {
            "prediction"       : None,
            "prediction_label" : None,
            "confidence"       : None,
            "error"            : f"Prediction error: {exc}",
        }


def models_are_available() -> Tuple[bool, bool]:
    """Check if model artifact files exist on disk."""
    url_ready  = (URL_MODEL_DIR / "url_model.pkl").exists()
    full_ready = (FULL_MODEL_DIR / "best_model.pkl").exists()
    return url_ready, full_ready


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "google.com",
        "https://github.com/torvalds/linux",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "http://paypal-security-update.com/login.php",
        "http://192.168.1.1/admin",
        "http://apple-id-verify.com/signin",
        "https://google.com.account-update.xyz/login"
    ]

    print("\n--- Diagnostic Prediction Test ---")
    for u in test_urls:
        res = predict_url(u)
        print(f"\nURL: {u}")
        print(f"  Result : {res['prediction_label']} ({res['risk_level']})")
        print(f"  Phish P: {res['phishing_prob']} | Legit P: {res['legitimate_prob']}")
        for r in res.get("reasons", []):
            print(f"  Reason : {r}")
