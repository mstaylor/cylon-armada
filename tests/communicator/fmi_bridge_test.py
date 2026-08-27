"""Tests for FMIBridge — channel-type passthrough and port resolution."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'shared', 'scripts'))


class TestFromPayloadChannelTypePassthrough:
    """direct-redis must reach FMIConfig unchanged from Step Functions payloads."""

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_payload_passes_direct_redis_channel_type_through(self, mock_import_fmi):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        payload = {"fmi_channel_type": "direct-redis", "world_size": 2, "rank": 0}
        FMIBridge.from_payload(payload)

        assert mock_fmi_config.call_args.kwargs["channel_type"] == "direct-redis"

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_payload_maps_tcpunch_to_direct(self, mock_import_fmi):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        payload = {"fmi_channel_type": "tcpunch", "world_size": 2, "rank": 0}
        FMIBridge.from_payload(payload)

        assert mock_fmi_config.call_args.kwargs["channel_type"] == "direct"


class TestPortResolutionByChannelType:
    """direct-redis has no rendezvous server — its 'port' argument must come from
    FMI_LISTEN_PORT (own listen port), not RENDEZVOUS_PORT."""

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_env_direct_redis_reads_fmi_listen_port(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("FMI_CHANNEL_TYPE", "direct-redis")
        monkeypatch.setenv("FMI_LISTEN_PORT", "50055")
        monkeypatch.setenv("RENDEZVOUS_PORT", "10000")

        FMIBridge.from_env()

        assert mock_fmi_config.call_args.kwargs["port"] == 50055

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_env_direct_channel_still_reads_rendezvous_port(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("FMI_CHANNEL_TYPE", "direct")
        monkeypatch.setenv("FMI_LISTEN_PORT", "50055")
        monkeypatch.setenv("RENDEZVOUS_PORT", "10000")

        FMIBridge.from_env()

        assert mock_fmi_config.call_args.kwargs["port"] == 10000

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_payload_direct_redis_reads_fmi_listen_port(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.setenv("FMI_LISTEN_PORT", "50055")
        monkeypatch.setenv("RENDEZVOUS_PORT", "10000")

        payload = {"fmi_channel_type": "direct-redis", "world_size": 2, "rank": 0}
        FMIBridge.from_payload(payload)

        assert mock_fmi_config.call_args.kwargs["port"] == 50055

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_payload_direct_channel_still_reads_rendezvous_port(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.setenv("FMI_LISTEN_PORT", "50055")
        monkeypatch.setenv("RENDEZVOUS_PORT", "10000")

        payload = {"fmi_channel_type": "direct", "world_size": 2, "rank": 0}
        FMIBridge.from_payload(payload)

        assert mock_fmi_config.call_args.kwargs["port"] == 10000


class TestAdvertiseHostIsSeparateFromRendezvousHost:
    """direct-redis must not silently advertise the TCPunch rendezvous host —
    advertise_host is a distinct config axis, empty by default so ECS metadata
    auto-discovery can run."""

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_env_defaults_advertise_host_to_empty(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("FMI_CHANNEL_TYPE", "direct-redis")
        monkeypatch.setenv("RENDEZVOUS_HOST", "cylon-rendezvous.aws-cylondata.com")
        monkeypatch.delenv("ADVERTISE_HOST", raising=False)

        FMIBridge.from_env()

        assert mock_fmi_config.call_args.kwargs["advertise_host"] == ""
        assert mock_fmi_config.call_args.kwargs["host"] == "cylon-rendezvous.aws-cylondata.com"

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_env_reads_advertise_host_when_explicitly_set(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("FMI_CHANNEL_TYPE", "direct-redis")
        monkeypatch.setenv("ADVERTISE_HOST", "10.0.3.17")

        FMIBridge.from_env()

        assert mock_fmi_config.call_args.kwargs["advertise_host"] == "10.0.3.17"

    @patch('communicator.fmi_bridge._import_fmi')
    def test_from_payload_defaults_advertise_host_to_empty(self, mock_import_fmi, monkeypatch):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        monkeypatch.delenv("ADVERTISE_HOST", raising=False)

        payload = {"fmi_channel_type": "direct-redis", "world_size": 2, "rank": 0}
        FMIBridge.from_payload(payload)

        assert mock_fmi_config.call_args.kwargs["advertise_host"] == ""
