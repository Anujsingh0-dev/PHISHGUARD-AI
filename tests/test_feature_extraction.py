"""
tests/test_feature_extraction.py
──────────────────────────────────
Tests for URL feature extraction (src/feature_extraction.py).

Run with:
  pytest tests/test_feature_extraction.py -v

These tests verify that:
  - Features are extracted correctly from known URLs
  - Edge cases are handled gracefully
  - Feature names and count are consistent
  - The feature extraction is deterministic
"""

import sys
import pathlib
import pytest

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.feature_extraction import (
    extract_url_features,
    URL_FEATURE_NAMES,
    features_to_ordered_list,
    _is_ip_address,
    _shannon_entropy,
)


class TestFeatureNames:
    """Verify the feature name list is consistent."""

    def test_feature_count(self):
        """There should be exactly 22 URL features."""
        assert len(URL_FEATURE_NAMES) == 22

    def test_no_duplicate_feature_names(self):
        """Every feature name must be unique."""
        assert len(URL_FEATURE_NAMES) == len(set(URL_FEATURE_NAMES))

    def test_required_features_present(self):
        """Key features must be present in the list."""
        required = [
            "url_length", "is_https", "has_ip_as_host",
            "has_at_symbol", "entropy", "num_dots",
        ]
        for feat in required:
            assert feat in URL_FEATURE_NAMES, f"Missing required feature: {feat}"


class TestExtractUrlFeatures:
    """Test extract_url_features() on various URL types."""

    def test_returns_dict(self):
        """Function must return a dictionary."""
        result = extract_url_features("https://www.google.com")
        assert isinstance(result, dict)

    def test_keys_match_feature_names(self):
        """Returned keys must exactly match URL_FEATURE_NAMES."""
        result = extract_url_features("https://www.example.com")
        assert set(result.keys()) == set(URL_FEATURE_NAMES)

    def test_https_url_detected(self):
        """HTTPS scheme should set is_https = 1."""
        result = extract_url_features("https://www.example.com")
        assert result["is_https"] == 1

    def test_http_url_not_https(self):
        """HTTP scheme should set is_https = 0."""
        result = extract_url_features("http://www.example.com")
        assert result["is_https"] == 0

    def test_ip_address_host_detected(self):
        """IP address as hostname should set has_ip_as_host = 1."""
        result = extract_url_features("http://192.168.1.1/login")
        assert result["has_ip_as_host"] == 1

    def test_domain_host_not_ip(self):
        """Normal domain name should set has_ip_as_host = 0."""
        result = extract_url_features("https://www.google.com")
        assert result["has_ip_as_host"] == 0

    def test_at_symbol_detected(self):
        """@ in URL should set has_at_symbol = 1."""
        result = extract_url_features("http://user@evil.com/login")
        assert result["has_at_symbol"] == 1

    def test_no_at_symbol(self):
        """Normal URL without @ should set has_at_symbol = 0."""
        result = extract_url_features("https://www.google.com")
        assert result["has_at_symbol"] == 0

    def test_url_length_is_correct(self):
        """url_length should match the actual length of the input."""
        url = "https://www.example.com"
        result = extract_url_features(url)
        assert result["url_length"] == len(url)

    def test_suspicious_keyword_detected(self):
        """URL with 'login' should have suspicious_words_count >= 1."""
        result = extract_url_features("http://evil.com/login/verify")
        assert result["suspicious_words_count"] >= 1

    def test_no_suspicious_words_in_clean_url(self):
        """A clean URL without phishing keywords should have count = 0 or low."""
        result = extract_url_features("https://www.weather.gov")
        assert result["suspicious_words_count"] == 0

    def test_query_params_counted(self):
        """Query parameters should be counted correctly."""
        result = extract_url_features("https://example.com/page?a=1&b=2&c=3")
        assert result["num_query_params"] == 3

    def test_no_query_params(self):
        """URL without query string should have 0 params."""
        result = extract_url_features("https://example.com/about")
        assert result["num_query_params"] == 0

    def test_subdomain_counted(self):
        """sub.example.com should have num_subdomains = 1."""
        result = extract_url_features("https://sub.example.com")
        assert result["num_subdomains"] == 1

    def test_no_subdomain(self):
        """example.com should have num_subdomains = 0."""
        result = extract_url_features("https://example.com")
        assert result["num_subdomains"] == 0

    def test_punycode_detected(self):
        """xn-- in hostname should set has_punycode = 1."""
        result = extract_url_features("http://xn--pypal-4ve.com/login")
        assert result["has_punycode"] == 1

    def test_percent_encoding_detected(self):
        """Percent-encoded characters should set has_percent_encoding = 1."""
        result = extract_url_features("https://example.com/path%20with%20spaces")
        assert result["has_percent_encoding"] == 1

    def test_entropy_is_float(self):
        """Entropy should be a non-negative float."""
        result = extract_url_features("https://www.google.com")
        assert isinstance(result["entropy"], float)
        assert result["entropy"] >= 0.0

    def test_all_values_are_numeric(self):
        """All feature values must be int or float."""
        result = extract_url_features("https://paypal-secure.com/account/verify")
        for key, val in result.items():
            assert isinstance(val, (int, float)), f"Feature {key} is not numeric: {type(val)}"

    def test_empty_url_returns_dict(self):
        """Empty URL should not crash — returns a dict of zeros/defaults."""
        result = extract_url_features("")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(URL_FEATURE_NAMES)

    def test_malformed_url_does_not_crash(self):
        """Malformed input should not raise an exception."""
        malformed_urls = [
            "not_a_url",
            "://missing-scheme",
            "http://",
            "https://" + "a" * 500,
            "http://user:pass@host.com:8080/path?q=1#fragment",
        ]
        for url in malformed_urls:
            result = extract_url_features(url)
            assert isinstance(result, dict), f"Failed for: {url}"

    def test_features_to_ordered_list(self):
        """features_to_ordered_list should return values in URL_FEATURE_NAMES order."""
        result = extract_url_features("https://www.example.com")
        ordered = features_to_ordered_list(result)
        assert len(ordered) == len(URL_FEATURE_NAMES)
        assert ordered[URL_FEATURE_NAMES.index("is_https")] == 1

    def test_deterministic(self):
        """Same URL must always return identical features."""
        url = "https://www.paypal.com/signin"
        r1  = extract_url_features(url)
        r2  = extract_url_features(url)
        assert r1 == r2


class TestHelpers:
    """Test internal helper functions."""

    def test_ip_detection_ipv4(self):
        assert _is_ip_address("192.168.1.1") is True

    def test_ip_detection_not_ip(self):
        assert _is_ip_address("www.google.com") is False

    def test_ip_detection_empty(self):
        assert _is_ip_address("") is False

    def test_entropy_empty_string(self):
        """Empty string entropy should be 0."""
        assert _shannon_entropy("") == 0.0

    def test_entropy_uniform(self):
        """All-same-character string has entropy 0."""
        assert _shannon_entropy("aaaa") == 0.0

    def test_entropy_varied(self):
        """Varied string should have entropy > 0."""
        assert _shannon_entropy("abcde12345") > 0.0
