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


class TestListenPortIsSeparateFromRendezvousPort:
    """listen_port and rendezvous_port are distinct constructor parameters —
    direct-redis must never fall back to rendezvous_port's default, and direct
    must never be affected by a caller-supplied listen_port."""

    @patch('communicator.fmi_bridge._import_fmi')
    def test_direct_redis_uses_listen_port_not_rendezvous_port_default(self, mock_import_fmi):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        FMIBridge(world_size=4, rank=1, channel_type="direct-redis",
                  listen_port=54321, rendezvous_port=10000)

        assert mock_fmi_config.call_args.kwargs["port"] == 54321

    @patch('communicator.fmi_bridge._import_fmi')
    def test_direct_channel_uses_rendezvous_port_ignores_listen_port(self, mock_import_fmi):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        FMIBridge(world_size=4, rank=1, channel_type="direct",
                  listen_port=54321, rendezvous_port=10000)

        assert mock_fmi_config.call_args.kwargs["port"] == 10000

    @patch('communicator.fmi_bridge._import_fmi')
    def test_two_direct_redis_ranks_with_distinct_listen_ports_get_distinct_ports(self, mock_import_fmi):
        from communicator.fmi_bridge import FMIBridge

        mock_fmi_config = MagicMock()
        mock_import_fmi.return_value = (mock_fmi_config, MagicMock(), MagicMock())

        FMIBridge(world_size=4, rank=0, channel_type="direct-redis", listen_port=50100)
        port_rank0 = mock_fmi_config.call_args.kwargs["port"]

        FMIBridge(world_size=4, rank=1, channel_type="direct-redis", listen_port=50101)
        port_rank1 = mock_fmi_config.call_args.kwargs["port"]

        assert port_rank0 == 50100
        assert port_rank1 == 50101
        assert port_rank0 != port_rank1
