"""
tests/test_relay_client.py — Unit tests for relay_client.PostCarClient.

Uses unittest.mock.patch to mock httpx.Client so no real network calls are made.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relay_client import PostCarClient, load_env


def _make_response(status_code: int = 200, json_data=None):
    """Helper: build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    # raise_for_status does nothing on success, raises on 4xx/5xx
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestPostCarClientHeartbeat(unittest.TestCase):

    def _client(self):
        with patch("httpx.Client"):
            return PostCarClient(
                relay_url="https://relay.example.com",
                agent_id="agt_test",
                agent_key="key_secret",
            )

    def test_heartbeat_returns_true_on_200(self):
        client = self._client()
        mock_resp = _make_response(200)
        client._http.post.return_value = mock_resp

        result = client.heartbeat(stress=0.5, version="1.0.0")

        self.assertTrue(result)
        client._http.post.assert_called_once()
        call_kwargs = client._http.post.call_args
        self.assertIn("/agents/agt_test/heartbeat", call_kwargs[0][0])

    def test_heartbeat_returns_false_on_non_200(self):
        client = self._client()
        mock_resp = _make_response(503)
        client._http.post.return_value = mock_resp

        result = client.heartbeat(stress=0.1, version="1.0.0")

        self.assertFalse(result)

    def test_heartbeat_returns_false_when_httpx_raises(self):
        client = self._client()
        import httpx
        client._http.post.side_effect = httpx.ConnectError("connection refused")

        result = client.heartbeat(stress=0.9, version="1.0.0")

        self.assertFalse(result)

    def test_heartbeat_returns_false_on_any_exception(self):
        client = self._client()
        client._http.post.side_effect = RuntimeError("unexpected")

        result = client.heartbeat(stress=0.0, version="0.1.0")

        self.assertFalse(result)


class TestPostCarClientSendQuery(unittest.TestCase):

    def _client(self):
        with patch("httpx.Client"):
            return PostCarClient(
                relay_url="https://relay.example.com",
                agent_id="agt_test",
                agent_key="key_secret",
            )

    def test_send_query_returns_query_id(self):
        client = self._client()
        mock_resp = _make_response(200, json_data={"query_id": "qry_abc123"})
        client._http.post.return_value = mock_resp

        result = client.send_query(
            tags=["trading", "macro"],
            question="What is the current trend?",
        )

        self.assertEqual(result, "qry_abc123")

    def test_send_query_passes_correct_body(self):
        client = self._client()
        mock_resp = _make_response(200, json_data={"query_id": "qry_xyz"})
        client._http.post.return_value = mock_resp

        client.send_query(
            tags=["risk"],
            question="Is this safe?",
            context="Market is volatile",
            urgency="high",
        )

        _, kwargs = client._http.post.call_args
        body = kwargs["json"]
        self.assertEqual(body["tags"], ["risk"])
        self.assertEqual(body["question"], "Is this safe?")
        self.assertEqual(body["context"], "Market is volatile")
        self.assertEqual(body["urgency"], "high")

    def test_send_query_returns_none_on_exception(self):
        client = self._client()
        client._http.post.side_effect = Exception("network down")

        result = client.send_query(tags=["test"], question="test?")

        self.assertIsNone(result)

    def test_send_query_omits_context_when_none(self):
        client = self._client()
        mock_resp = _make_response(200, json_data={"query_id": "qry_no_ctx"})
        client._http.post.return_value = mock_resp

        client.send_query(tags=["tag"], question="q?")

        _, kwargs = client._http.post.call_args
        self.assertNotIn("context", kwargs["json"])


class TestPostCarClientRateOffer(unittest.TestCase):

    def _client(self):
        with patch("httpx.Client"):
            return PostCarClient(
                relay_url="https://relay.example.com",
                agent_id="agt_test",
                agent_key="key_secret",
            )

    def test_rate_offer_sends_correct_body(self):
        client = self._client()
        mock_resp = _make_response(200)
        client._http.post.return_value = mock_resp

        result = client.rate_offer(offer_id="off_001", rating="useful")

        self.assertTrue(result)
        _, kwargs = client._http.post.call_args
        self.assertEqual(kwargs["json"], {"rating": "useful"})
        self.assertIn("/offers/off_001/rate", client._http.post.call_args[0][0])

    def test_rate_offer_returns_false_on_error(self):
        client = self._client()
        client._http.post.side_effect = Exception("timeout")

        result = client.rate_offer(offer_id="off_001", rating="negative")

        self.assertFalse(result)

    def test_rate_offer_all_valid_ratings(self):
        for rating in ("useful", "related", "unrelated", "negative"):
            client = self._client()
            mock_resp = _make_response(200)
            client._http.post.return_value = mock_resp

            result = client.rate_offer(offer_id=f"off_{rating}", rating=rating)
            self.assertTrue(result, f"Expected True for rating={rating}")


class TestPostCarClientFromEnv(unittest.TestCase):

    def test_from_env_loads_dotenv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as fh:
                fh.write("POSTCAR_RELAY_URL=https://example.relay.io\n")
                fh.write("POSTCAR_AGENT_ID=agt_loaded\n")
                fh.write("POSTCAR_AGENT_KEY=secret_loaded\n")

            with patch("httpx.Client"):
                client = PostCarClient.from_env(env_dir=tmpdir)

            self.assertEqual(client.relay_url, "https://example.relay.io")
            self.assertEqual(client.agent_id, "agt_loaded")
            self.assertEqual(client.agent_key, "secret_loaded")

    def test_from_env_falls_back_to_os_environ(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No .env file — should fall back to os.environ
            env_vars = {
                "POSTCAR_RELAY_URL": "https://fallback.relay.io",
                "POSTCAR_AGENT_ID": "agt_env",
                "POSTCAR_AGENT_KEY": "key_env",
            }
            with patch.dict(os.environ, env_vars):
                with patch("httpx.Client"):
                    client = PostCarClient.from_env(env_dir=tmpdir)

            self.assertEqual(client.relay_url, "https://fallback.relay.io")
            self.assertEqual(client.agent_id, "agt_env")
            self.assertEqual(client.agent_key, "key_env")

    def test_load_env_parses_file_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as fh:
                fh.write("# comment line\n")
                fh.write("KEY_A=value_a\n")
                fh.write('KEY_B="quoted value"\n')
                fh.write("KEY_C=value_c  # inline comment\n")
                fh.write("\n")

            result = load_env(tmpdir)

            self.assertEqual(result.get("KEY_A"), "value_a")
            self.assertEqual(result.get("KEY_B"), "quoted value")
            # inline comment stripped
            self.assertNotIn("#", result.get("KEY_C", ""))

    def test_load_env_returns_empty_dict_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_env(tmpdir)
            self.assertEqual(result, {})


class TestPostCarClientHeaders(unittest.TestCase):

    def test_headers_returns_correct_dict(self):
        with patch("httpx.Client"):
            client = PostCarClient(
                relay_url="https://relay.example.com",
                agent_id="agt_hdr",
                agent_key="key_hdr",
            )

        headers = client._headers()
        self.assertEqual(headers["X-PostCar-Agent"], "agt_hdr")
        self.assertEqual(headers["X-PostCar-Key"], "key_hdr")


class TestPostCarClientGetMethods(unittest.TestCase):

    def _client(self):
        with patch("httpx.Client"):
            return PostCarClient(
                relay_url="https://relay.example.com",
                agent_id="agt_test",
                agent_key="key_secret",
            )

    def test_get_offers_returns_list(self):
        client = self._client()
        mock_resp = _make_response(200, json_data=[{"offer_id": "o1"}, {"offer_id": "o2"}])
        client._http.get.return_value = mock_resp

        result = client.get_offers(limit=5)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_get_offers_returns_empty_list_on_error(self):
        client = self._client()
        client._http.get.side_effect = Exception("network error")

        result = client.get_offers()

        self.assertEqual(result, [])

    def test_get_queries_returns_list(self):
        client = self._client()
        mock_resp = _make_response(200, json_data=[{"query_id": "q1"}])
        client._http.get.return_value = mock_resp

        result = client.get_queries(limit=10)

        self.assertIsInstance(result, list)

    def test_get_gaps_returns_list(self):
        client = self._client()
        mock_resp = _make_response(200, json_data=["gap_a", "gap_b"])
        client._http.get.return_value = mock_resp

        result = client.get_gaps()

        self.assertIsInstance(result, list)

    def test_get_version_returns_dict(self):
        client = self._client()
        mock_resp = _make_response(200, json_data={"version": "1.2.3"})
        client._http.get.return_value = mock_resp

        result = client.get_version()

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("version"), "1.2.3")

    def test_get_version_returns_empty_dict_on_error(self):
        client = self._client()
        client._http.get.side_effect = Exception("connection refused")

        result = client.get_version()

        self.assertEqual(result, {})


class TestPostCarClientRespondToQuery(unittest.TestCase):

    def _client(self):
        with patch("httpx.Client"):
            return PostCarClient(
                relay_url="https://relay.example.com",
                agent_id="agt_test",
                agent_key="key_secret",
            )

    def test_respond_to_query_returns_offer_id(self):
        client = self._client()
        mock_resp = _make_response(200, json_data={"offer_id": "off_resp_001"})
        client._http.post.return_value = mock_resp

        result = client.respond_to_query(
            query_id="qry_001",
            content="My response content",
            confidence=0.85,
        )

        self.assertEqual(result, "off_resp_001")

    def test_respond_to_query_returns_none_on_error(self):
        client = self._client()
        client._http.post.side_effect = Exception("timeout")

        result = client.respond_to_query(query_id="qry_001", content="test")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
