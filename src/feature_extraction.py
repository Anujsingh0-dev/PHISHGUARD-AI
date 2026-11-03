"""
src/feature_extraction.py
──────────────────────────
URL-Only Feature Extraction for MODE B (Live URL Prediction).

This module extracts features ONLY from the URL string itself —
no webpage downloading, no JavaScript, no third-party APIs.

All features here can be computed offline from a raw URL string.
The SAME extract_url_features() function is used both during:
  • Training the URL-only model (on dataset URLs)
  • Predicting on a new URL entered by the user

This guarantees that training and prediction use identical features
in the exact same order — which is required for correct ML predictions.

IMPORTANT:
  This module does NOT cover webpage-derived features
  (e.g. HTML content, DOM structure, SSL certificate details).
  Those features are available only in the FULL dataset model (MODE A).
"""

import re
import math
import pathlib
import ipaddress
from urllib.parse import urlparse, parse_qs, unquote


# ── Feature names (in order) ──────────────────────────────────────────────────
# This list MUST match the exact order that extract_url_features() returns.
# If you add or remove features here, update extract_url_features() to match.
URL_FEATURE_NAMES = [
    "url_length",            # Total length of the URL string
    "domain_length",         # Length of the domain (hostname) part (excluding www.)
    "path_length",           # Length of the URL path
    "num_dots",              # Number of '.' characters in the domain
    "num_hyphens",           # Number of '-' characters in the domain
    "num_underscores",       # Number of '_' characters in the full URL
    "num_slashes",           # Number of '/' in the URL path
    "num_digits",            # Number of digit characters (0-9) in domain
    "digit_ratio",           # Proportion of digits relative to domain length
    "num_special_chars",     # Count of special characters: @, !, $, %, ^, &, *
    "has_at_symbol",         # 1 if '@' is present in URL (common phishing trick)
    "has_double_slash",      # 1 if '//' appears after the protocol
    "has_ip_as_host",        # 1 if the hostname looks like an IP address
    "is_https",              # 1 if the scheme is 'https', else 0
    "num_subdomains",        # Number of subdomains (excluding www.)
    "num_query_params",      # Number of query string parameters
    "has_port",              # 1 if a port number is specified in the URL
    "has_percent_encoding",  # 1 if '%XX' encoding is present (obfuscation)
    "has_punycode",          # 1 if 'xn--' appears (internationalised domain trick)
    "num_path_segments",     # Number of segments in the URL path
    "suspicious_words_count",# Count of suspicious keywords in the URL
    "entropy",               # Shannon entropy of the domain (high = random/suspicious)
]

# Words commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "account", "update", "verify", "secure",
    "banking", "paypal", "ebay", "amazon", "apple", "google",
    "microsoft", "confirm", "password", "credential", "free",
    "winner", "lucky", "prize", "click", "urgent", "alert",
    "wallet", "recover", "authenticate", "billing", "invoice",
]


def _shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    High entropy means the string looks random — common in obfuscated phishing URLs.
    Low entropy means the string has patterns — common in legitimate domain names.

    Formula: H = -sum(p * log2(p)) for each unique character p.

    Parameters
    ----------
    text : str

    Returns
    -------
    float — entropy value (higher = more random)
    """
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def _is_ip_address(hostname: str) -> bool:
    """
    Returns True if the hostname is a valid IPv4 or IPv6 address.

    Using an IP address instead of a domain name is a common phishing trick
    to avoid registering a recognisable domain.

    Parameters
    ----------
    hostname : str

    Returns
    -------
    bool
    """
    if not hostname:
        return False
    try:
        # Strip port or brackets if any
        h = hostname.split(":")[0].strip("[]")
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def normalize_url(url: str) -> str:
    """
    Standardize a raw URL string before parsing and feature extraction.

    Ensures that 'google.com', 'https://google.com/', and 'https://www.google.com'
    are canonicalized consistently to avoid trivial syntactic mismatches.
    """
    url = str(url).strip()
    if not url:
        return ""

    # If scheme is missing, default to https:// (standard modern web traffic)
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


def extract_url_features(url: str) -> dict:
    """
    Extract a fixed set of numerical features from a URL string.

    This function is the CORE of the URL-only model.
    It is called identically during training and live prediction.

    Parameters
    ----------
    url : str
        The raw URL string entered by the user or read from the dataset.
        Example: 'https://www.example.com/login?user=test'

    Returns
    -------
    dict
        A dictionary mapping feature names to numerical values.
        Keys match URL_FEATURE_NAMES in the same order.
    """
    url = normalize_url(url)
    if not url:
        return {name: 0 for name in URL_FEATURE_NAMES}

    try:
        parsed = urlparse(url)
    except Exception:
        return {name: 0 for name in URL_FEATURE_NAMES}

    raw_hostname = parsed.hostname or ""
    hostname = raw_hostname.lower()
    path     = parsed.path or ""
    query    = parsed.query or ""
    scheme   = parsed.scheme or ""

    # Canonicalize domain without 'www.' prefix for structural metrics
    # so 'google.com' and 'www.google.com' yield identical domain metrics
    if hostname.startswith("www."):
        domain_for_metrics = hostname[4:]
    else:
        domain_for_metrics = hostname

    # ── Compute each feature ───────────────────────────────────────────────────

    url_length    = len(url)
    domain_length = len(domain_for_metrics)
    path_length   = len(path)

    # Dots and hyphens in the domain (clean domain excludes www.)
    num_dots        = domain_for_metrics.count(".")
    num_hyphens     = domain_for_metrics.count("-")
    num_underscores = url.count("_")
    num_slashes     = path.count("/")

    # Digits in domain
    num_digits  = sum(1 for ch in domain_for_metrics if ch.isdigit())
    digit_ratio = num_digits / domain_length if domain_length > 0 else 0.0

    # Special characters in full URL
    special_chars     = set("@!$%^&*")
    num_special_chars = sum(1 for ch in url if ch in special_chars)

    has_at_symbol    = 1 if "@" in url else 0
    has_double_slash = 1 if "//" in url[7:] else 0  # Skip the protocol '//'
    has_ip_as_host   = 1 if _is_ip_address(hostname) else 0
    is_https         = 1 if scheme.lower() == "https" else 0

    # Count subdomains in the clean domain (excluding 'www')
    # e.g. 'google.com' -> 0 subdomains; 'mail.google.com' -> 1 subdomain
    domain_parts = domain_for_metrics.split(".")
    num_subdomains = max(0, len(domain_parts) - 2)

    # Query parameters
    try:
        query_params     = parse_qs(query)
        num_query_params = len(query_params)
    except Exception:
        num_query_params = 0

    has_port = 1 if parsed.port is not None else 0

    # Percent encoding & punycode
    has_percent_encoding = 1 if re.search(r"%[0-9A-Fa-f]{2}", url) else 0
    has_punycode         = 1 if "xn--" in hostname else 0

    # Path segments
    path_segments     = [s for s in path.split("/") if s]
    num_path_segments = len(path_segments)

    # Count suspicious keywords in entire URL
    url_lower = url.lower()
    suspicious_words_count = sum(
        1 for word in SUSPICIOUS_KEYWORDS if word in url_lower
    )

    # Domain entropy (Shannon entropy of clean domain)
    entropy = _shannon_entropy(domain_for_metrics)

    # ── Build the result dictionary in the exact order of URL_FEATURE_NAMES ────
    features = {
        "url_length"            : url_length,
        "domain_length"         : domain_length,
        "path_length"           : path_length,
        "num_dots"              : num_dots,
        "num_hyphens"           : num_hyphens,
        "num_underscores"       : num_underscores,
        "num_slashes"           : num_slashes,
        "num_digits"            : num_digits,
        "digit_ratio"           : round(digit_ratio, 6),
        "num_special_chars"     : num_special_chars,
        "has_at_symbol"         : has_at_symbol,
        "has_double_slash"      : has_double_slash,
        "has_ip_as_host"        : has_ip_as_host,
        "is_https"              : is_https,
        "num_subdomains"        : num_subdomains,
        "num_query_params"      : num_query_params,
        "has_port"              : has_port,
        "has_percent_encoding"  : has_percent_encoding,
        "has_punycode"          : has_punycode,
        "num_path_segments"     : num_path_segments,
        "suspicious_words_count": suspicious_words_count,
        "entropy"               : round(entropy, 6),
    }

    assert set(features.keys()) == set(URL_FEATURE_NAMES), (
        "Feature dictionary keys do not match URL_FEATURE_NAMES."
    )

    return features


def features_to_ordered_list(features: dict) -> list:
    """
    Convert the feature dictionary to a list in the correct column order.

    Parameters
    ----------
    features : dict — output of extract_url_features()

    Returns
    -------
    list — values in the same order as URL_FEATURE_NAMES
    """
    return [features[name] for name in URL_FEATURE_NAMES]


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "google.com",
        "https://google.com/",
        "https://github.com/torvalds/linux",
        "http://192.168.1.1/login/verify?user=admin",
        "https://paypal-secure-update.com/account/login",
        "http://xn--pypal-4ve.com/",
    ]

    print("URL Feature Extraction Test")
    print("=" * 80)
    for url in test_urls:
        feats = extract_url_features(url)
        print(f"\nURL: {url}")
        for name in ["url_length", "domain_length", "num_dots", "num_subdomains", "is_https", "has_ip_as_host", "suspicious_words_count"]:
            print(f"  {name:<25} = {feats[name]}")
