"""
tests/test_prediction.py
─────────────────────────
Tests for prediction module (src/predict.py).

Run with:
  pytest tests/test_prediction.py -v

These tests check input validation and response structure.
They do NOT require trained models — model-dependent tests are skipped
automatically if models are not available.
"""

import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.predict import predict_url, models_are_available


class TestPredictUrlValidation:
    """Test input validation in predict_url()."""

    def test_empty_url_returns_error(self):
        """Empty URL should return an error, not crash."""
        result = predict_url("")
        assert result["error"] is not None
        assert result["prediction"] is None

    def test_whitespace_url_returns_error(self):
        """Whitespace-only URL should return an error."""
        result = predict_url("   ")
        assert result["error"] is not None

    def test_too_short_url_returns_error(self):
        """Very short string should return an error."""
        result = predict_url("a")
        assert result["error"] is not None

    def test_result_has_required_keys(self):
        """Every result dict must contain the required keys."""
        result = predict_url("https://example.com")
        required_keys = [
            "prediction", "prediction_label", "confidence",
            "phishing_prob", "legitimate_prob", "features",
            "url", "error",
        ]
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"


class TestPredictUrlWithModel:
    """Tests that require the URL model — skipped if not trained."""

    def setup_method(self):
        url_ready, _ = models_are_available()
        if not url_ready:
            pytest.skip("URL model not trained — skipping model-dependent tests.")

    def test_https_google_prediction(self):
        """A well-known HTTPS URL should return a valid prediction."""
        result = predict_url("https://www.google.com")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]
        assert result["prediction_label"] in ["Phishing", "Legitimate"]

    def test_http_url_prediction(self):
        """HTTP URL should also return a valid prediction."""
        result = predict_url("http://example.com/page")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]

    def test_ip_address_url_prediction(self):
        """URL with IP address should return a valid prediction."""
        result = predict_url("http://192.168.1.1/admin")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]

    def test_suspicious_url_prediction(self):
        """A suspicious-looking URL should return a valid (not necessarily phishing) result."""
        result = predict_url("http://paypal-secure-login-verify.com/account?user=1234")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]

    def test_probabilities_sum_to_one(self):
        """Phishing probability + Legitimate probability should equal ~1.0."""
        result = predict_url("https://www.amazon.com")
        if result["phishing_prob"] is not None:
            total = result["phishing_prob"] + result["legitimate_prob"]
            assert abs(total - 1.0) < 0.01, f"Probabilities do not sum to 1: {total}"

    def test_features_returned(self):
        """Prediction result should include extracted URL features."""
        result = predict_url("https://www.example.com")
        assert isinstance(result["features"], dict)
        assert len(result["features"]) > 0

    def test_subdomain_url_prediction(self):
        """URL with subdomain should return valid prediction."""
        result = predict_url("https://mail.google.com/inbox")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]

    def test_url_with_special_chars(self):
        """URL with special characters should be handled gracefully."""
        result = predict_url("https://example.com/path?q=hello+world&lang=en")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]

    def test_punycode_url_prediction(self):
        """Punycode URL should return valid prediction."""
        result = predict_url("http://xn--pypal-4ve.com/login")
        assert result["error"] is None
        assert result["prediction"] in [0, 1]
