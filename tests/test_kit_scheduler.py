"""
Tests for postcar_kit.py — main 5-minute scheduler entry point.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure project root is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import postcar_kit  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(
    heartbeat_ok: bool = True,
    send_query_return: str = "q123",
    get_offers_return=None,
    get_version_return=None,
):
    """Return a MagicMock that quacks like PostCarClient."""
    client = MagicMock()
    client.heartbeat.return_value = heartbeat_ok
    client.send_query.return_value = send_query_return
    client.get_offers.return_value = get_offers_return or []
    client.get_version.return_value = get_version_return or {}
    client.rate_offer.return_value = True
    return client


# ---------------------------------------------------------------------------
# load_client
# ---------------------------------------------------------------------------


class TestLoadClient:
    def test_returns_none_when_no_env_file(self, tmp_path):
        """load_client returns None if no .env file exists and env vars are absent."""
        # Strip any lingering POSTCAR_* vars from the environment so the test
        # is hermetic even when a developer has them set locally.
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("POSTCAR_")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            result = postcar_kit.load_client(str(tmp_path))
        assert result is None

    def test_returns_none_when_env_vars_blank(self, tmp_path):
        """load_client returns None when env vars are empty strings."""
        env = {"POSTCAR_RELAY_URL": "", "POSTCAR_AGENT_ID": "", "POSTCAR_AGENT_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            result = postcar_kit.load_client(str(tmp_path))
        assert result is None

    def test_returns_client_when_all_vars_present(self, tmp_path):
        """load_client returns a PostCarClient when all credentials are set."""
        env = {
            "POSTCAR_RELAY_URL": "https://relay.example.com",
            "POSTCAR_AGENT_ID": "agt_test",
            "POSTCAR_AGENT_KEY": "key_test",
        }
        with patch.dict(os.environ, env, clear=True):
            result = postcar_kit.load_client(str(tmp_path))
        # Should be a PostCarClient (or mock-compatible object), not None.
        assert result is not None

    def test_returns_none_on_exception(self, tmp_path):
        """load_client returns None when PostCarClient.from_env raises."""
        with patch("postcar_kit.PostCarClient.from_env", side_effect=RuntimeError("boom")):
            result = postcar_kit.load_client(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_creates_log_file_in_agent_dir(self, tmp_path):
        """setup_logging creates .postcar.log inside agent_dir."""
        # Use a fresh logger name to avoid state from other tests.
        with patch("postcar_kit.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock(spec=logging.Logger)
            mock_logger.handlers = []
            mock_get_logger.return_value = mock_logger
            postcar_kit.setup_logging(str(tmp_path))

        # Call the real function so the file is actually created.
        # Reset logger to avoid duplicate handlers from other tests.
        real_logger_name = "postcar_kit_test_" + tmp_path.name
        with patch("postcar_kit.logging") as mock_logging_module:
            # Redirect to a real logger with a unique name.
            real_logger = logging.getLogger(real_logger_name)
            real_logger.handlers.clear()
            mock_logging_module.getLogger.return_value = real_logger
            mock_logging_module.INFO = logging.INFO
            mock_logging_module.Formatter = logging.Formatter
            mock_logging_module.FileHandler = logging.FileHandler
            mock_logging_module.StreamHandler = logging.StreamHandler
            postcar_kit.setup_logging(str(tmp_path))

        log_file = tmp_path / ".postcar.log"
        assert log_file.exists(), ".postcar.log was not created in agent_dir"

    def test_creates_log_file_directly(self, tmp_path):
        """setup_logging (real call) creates .postcar.log in tmp_path."""
        # Use a unique logger name to avoid handler accumulation.
        unique_name = "postcar_kit_direct_" + tmp_path.name
        with patch("postcar_kit.logging.getLogger", return_value=logging.getLogger(unique_name)):
            logger = postcar_kit.setup_logging(str(tmp_path))

        log_file = tmp_path / ".postcar.log"
        assert log_file.exists(), ".postcar.log should be created by setup_logging"

    def test_returns_logger(self, tmp_path):
        """setup_logging returns a Logger instance."""
        unique_name = "postcar_kit_ret_" + tmp_path.name
        with patch("postcar_kit.logging.getLogger", return_value=logging.getLogger(unique_name)):
            logger = postcar_kit.setup_logging(str(tmp_path))
        assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


class TestRunOnce:
    def _run_with_mocks(self, tmp_path, fired_triggers=None, dedup_ok=True, version_info=None):
        """Helper: call run_once with all heavy dependencies mocked."""
        if fired_triggers is None:
            fired_triggers = []

        stress_summary = {
            "level": "normal",
            "indicators": {"failure_streak": 0, "error_rate": 0.0, "performance_delta": 0.0},
            "framework": "generic",
        }
        inbox_result = {"applied": 0, "deferred": 0, "rejected": 0, "guidance_path": None}
        client = _make_mock_client(get_version_return=version_info or {})
        logger = logging.getLogger("test_run_once")

        with (
            patch("stress.compute_stress_summary", return_value=stress_summary),
            patch("trigger.load_triggers", return_value=[]),
            patch("trigger.eval_triggers", return_value=fired_triggers),
            patch("trigger.dedup_check", return_value=dedup_ok),
            patch("trigger.mark_triggered"),
            patch("trigger.generate_query", return_value={
                "tags": ["skill:debugging"],
                "question": "What happened?",
                "urgency": "high",
            }),
            patch("llm.call_llm", return_value="guidance text"),
            patch("inbox.execute_inbox_cycle", return_value=inbox_result),
        ):
            postcar_kit.run_once(str(tmp_path), client, logger)

        return client

    def test_run_once_no_triggers(self, tmp_path):
        """run_once with no fired triggers completes without error."""
        client = self._run_with_mocks(tmp_path, fired_triggers=[])
        client.heartbeat.assert_called_once()

    def test_run_once_with_fired_trigger_sends_query(self, tmp_path):
        """run_once sends a query when a trigger fires and passes dedup."""
        trigger = {
            "id": "T001",
            "name": "high_failure_streak",
            "condition_key": "failure_streak",
            "threshold": 5,
            "urgency": "high",
            "tags": ["skill:debugging"],
            "topic": "high failure streak",
            "window_h": 12,
            "condition": "gt",
        }
        client = self._run_with_mocks(tmp_path, fired_triggers=[trigger], dedup_ok=True)
        client.send_query.assert_called_once()

    def test_run_once_skips_query_when_dedup_blocks(self, tmp_path):
        """run_once does NOT send a query when dedup_check returns False."""
        trigger = {
            "id": "T001",
            "name": "high_failure_streak",
            "urgency": "high",
            "tags": [],
            "topic": "streak",
            "window_h": 12,
        }
        client = self._run_with_mocks(tmp_path, fired_triggers=[trigger], dedup_ok=False)
        client.send_query.assert_not_called()

    def test_run_once_logs_new_version(self, tmp_path):
        """run_once logs a message when relay reports a newer version."""
        logger = MagicMock(spec=logging.Logger)
        stress_summary = {
            "level": "normal",
            "indicators": {},
            "framework": "generic",
        }
        inbox_result = {"applied": 0, "deferred": 0, "rejected": 0, "guidance_path": None}
        client = _make_mock_client(get_version_return={"version": "9.9.9"})

        with (
            patch("stress.compute_stress_summary", return_value=stress_summary),
            patch("trigger.load_triggers", return_value=[]),
            patch("trigger.eval_triggers", return_value=[]),
            patch("inbox.execute_inbox_cycle", return_value=inbox_result),
        ):
            postcar_kit.run_once(str(tmp_path), client, logger)

        # Expect an info log mentioning the new version.
        info_calls = [str(c) for c in logger.info.call_args_list]
        assert any("9.9.9" in c for c in info_calls), (
            "Expected logger.info to mention the new version '9.9.9'"
        )

    def test_run_once_inbox_cycle_called(self, tmp_path):
        """run_once always calls execute_inbox_cycle."""
        stress_summary = {
            "level": "normal",
            "indicators": {},
            "framework": "generic",
        }
        inbox_result = {"applied": 3, "deferred": 1, "rejected": 0, "guidance_path": None}
        client = _make_mock_client()

        mock_inbox = MagicMock(return_value=inbox_result)
        with (
            patch("stress.compute_stress_summary", return_value=stress_summary),
            patch("trigger.load_triggers", return_value=[]),
            patch("trigger.eval_triggers", return_value=[]),
            patch("inbox.execute_inbox_cycle", mock_inbox),
        ):
            postcar_kit.run_once(str(tmp_path), client, logger=logging.getLogger("test"))

        mock_inbox.assert_called_once_with(client, str(tmp_path))


# ---------------------------------------------------------------------------
# --once flag
# ---------------------------------------------------------------------------


class TestOnceFlag:
    def test_once_flag_calls_run_once_and_exits(self, tmp_path):
        """--once mode: run_once is called exactly once and main returns."""
        env = {
            "POSTCAR_RELAY_URL": "https://relay.example.com",
            "POSTCAR_AGENT_ID": "agt_test",
            "POSTCAR_AGENT_KEY": "key_test",
        }

        mock_run_once = MagicMock()
        unique_logger_name = "postcar_kit_once_" + tmp_path.name

        with (
            patch.dict(os.environ, env, clear=True),
            patch("postcar_kit.run_once", mock_run_once),
            patch("postcar_kit.logging.getLogger", return_value=logging.getLogger(unique_logger_name)),
            patch("sys.argv", ["postcar_kit.py", "--agent-dir", str(tmp_path), "--once"]),
        ):
            postcar_kit.main()

        mock_run_once.assert_called_once()

    def test_once_flag_does_not_call_run_once_when_no_client(self, tmp_path):
        """--once mode: run_once is NOT called when client is None."""
        env_clean = {k: v for k, v in os.environ.items() if not k.startswith("POSTCAR_")}
        mock_run_once = MagicMock()
        unique_logger_name = "postcar_kit_once_no_creds_" + tmp_path.name

        with (
            patch.dict(os.environ, env_clean, clear=True),
            patch("postcar_kit.run_once", mock_run_once),
            patch("postcar_kit.logging.getLogger", return_value=logging.getLogger(unique_logger_name)),
            patch("sys.argv", ["postcar_kit.py", "--agent-dir", str(tmp_path), "--once"]),
        ):
            postcar_kit.main()

        mock_run_once.assert_not_called()

    def test_daemon_loop_stops_on_stop_event(self, tmp_path):
        """Daemon loop exits when stop_event is set immediately."""
        env = {
            "POSTCAR_RELAY_URL": "https://relay.example.com",
            "POSTCAR_AGENT_ID": "agt_test",
            "POSTCAR_AGENT_KEY": "key_test",
        }
        mock_run_once = MagicMock()
        unique_logger_name = "postcar_kit_daemon_" + tmp_path.name

        real_threading_Event = threading.Event

        def patched_Event():
            ev = real_threading_Event()
            ev.set()  # Pre-set so the loop exits immediately after first check.
            return ev

        with (
            patch.dict(os.environ, env, clear=True),
            patch("postcar_kit.run_once", mock_run_once),
            patch("postcar_kit.logging.getLogger", return_value=logging.getLogger(unique_logger_name)),
            patch("postcar_kit.threading.Event", patched_Event),
            patch("sys.argv", ["postcar_kit.py", "--agent-dir", str(tmp_path)]),
        ):
            postcar_kit.main()

        # run_once should NOT have been called because stop_event was already set.
        mock_run_once.assert_not_called()
        # PID file should have been cleaned up.
        assert not (tmp_path / ".postcar_running.pid").exists()
