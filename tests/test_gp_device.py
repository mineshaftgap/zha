"""Tests for the ZHA GP device wrapper (P2 acceptance criteria).

All tests run without hardware - a synthetic zigpy.zgp.device.GPDevice is
sufficient. The gateway is a minimal mock; real gateway integration is P3.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from zigpy.quirks import _GP_REGISTRY, CustomGreenPowerDevice
from zigpy.zgp.device import GPDevice as ZigpyGPDevice
from zigpy.zgp.types import SecurityKeyType, SecurityLevel

from zha.application.const import POWER_BATTERY_OR_UNKNOWN, UNKNOWN
from zha.zigbee.device import DeviceStatus, ExtendedDeviceInfo
from zha.zigbee.gp_device import GPDevice

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_zigpy_gpd(
    source_id: int = 0x0040F4E4,
    device_id: int = 0x02,
    last_seen: datetime | None = None,
) -> ZigpyGPDevice:
    """Synthetic Hue Tap GPD (NoSecurity / NoKey)."""
    return ZigpyGPDevice(
        source_id=source_id,
        device_id=device_id,
        security_level=SecurityLevel.NoSecurity,
        security_key_type=SecurityKeyType.NoKey,
        frame_counter=0,
        last_seen=last_seen,
    )


def _make_gateway() -> MagicMock:
    """Minimal mock gateway (P3 will wire the real one)."""
    gw = MagicMock()
    return gw


# ---------------------------------------------------------------------------
# Stub quirk - registered per-test via the _register_stub_quirk fixture below
# ---------------------------------------------------------------------------


class _TestHueTapStub(CustomGreenPowerDevice, priority=3):
    """Lightweight stub; the real HueTap quirk has the same shape."""

    manufacturer = "Philips"
    model = "Hue Tap"
    device_automation_triggers = {
        ("pressed", "button_1"): {
            "command": "notification",
            "cluster_id": 0x0021,
            "params": {"command_id": 0x22},
        },
    }

    @classmethod
    def match(cls, gpd: ZigpyGPDevice) -> bool:
        return (
            gpd.security_level is SecurityLevel.NoSecurity
            and gpd.security_key_type is SecurityKeyType.NoKey
            and (gpd.source_id & 0xFFFF0000) == 0x00400000
        )


# Subclassing CustomGreenPowerDevice self-registers the stub into the global
# _GP_REGISTRY at import time.  Undo that side effect so registration is owned
# entirely by the _register_stub_quirk fixture and never leaks across modules.
_GP_REGISTRY.remove(_TestHueTapStub)


@pytest.fixture(autouse=True)
def _register_stub_quirk():
    """Register the stub GP quirk for the duration of each test only.

    Adds the stub to the global _GP_REGISTRY on setup and removes it on
    teardown, mirroring CustomGreenPowerDevice.__init_subclass__ (append + sort
    by priority) so there is no session-wide registry pollution.
    """
    _GP_REGISTRY.append(_TestHueTapStub)
    _GP_REGISTRY.sort(key=lambda c: c.priority)
    try:
        yield
    finally:
        if _TestHueTapStub in _GP_REGISTRY:
            _GP_REGISTRY.remove(_TestHueTapStub)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGPDeviceConstruction:
    """Construction of the GPDevice wrapper."""

    def test_creates_without_crashing(self):
        """A GPDevice can be built from a synthetic zigpy GPD."""
        gpd = _make_zigpy_gpd()
        dev = GPDevice(gpd, _make_gateway())
        assert dev is not None

    def test_quirk_matched(self):
        """A recognised GPD resolves to the stub quirk."""
        gpd = _make_zigpy_gpd()
        dev = GPDevice(gpd, _make_gateway())
        assert dev._quirk is _TestHueTapStub
        assert dev.quirk_applied is True

    def test_no_quirk_for_unknown_device(self):
        """An unrecognised GPD has no quirk applied."""
        gpd = _make_zigpy_gpd(source_id=0x11223344)
        dev = GPDevice(gpd, _make_gateway())
        assert dev._quirk is None
        assert dev.quirk_applied is False


# ---------------------------------------------------------------------------
# Identity properties
# ---------------------------------------------------------------------------


class TestGPDeviceIdentity:
    """Identity properties overridden for a GPD."""

    def setup_method(self):
        """Build a wrapper around a default synthetic GPD."""
        self.gpd = _make_zigpy_gpd()
        self.dev = GPDevice(self.gpd, _make_gateway())

    def test_ieee_matches_synthetic(self):
        """The ieee mirrors the synthetic EUI64 of the zigpy GPD."""
        assert self.dev.ieee == self.gpd.ieee

    def test_nwk_is_sentinel(self):
        """The nwk reports the GP sentinel address."""
        assert self.dev.nwk == 0xFFFE

    def test_manufacturer_from_quirk(self):
        """The manufacturer comes from the matched quirk."""
        assert self.dev.manufacturer == "Philips"

    def test_model_from_quirk(self):
        """The model comes from the matched quirk."""
        assert self.dev.model == "Hue Tap"

    def test_name_concatenated(self):
        """The name is "<manufacturer> <model>"."""
        assert self.dev.name == "Philips Hue Tap"

    def test_is_end_device(self):
        """A GPD presents as an end device, not router/coordinator."""
        assert self.dev.is_end_device is True
        assert self.dev.is_coordinator is False
        assert self.dev.is_router is False

    def test_skip_configuration(self):
        """A receive-only GPD skips configuration."""
        assert self.dev.skip_configuration is True

    def test_power_source(self):
        """power_source reports the battery/unknown label."""
        assert self.dev.power_source == POWER_BATTERY_OR_UNKNOWN

    def test_device_type_unknown(self):
        """device_type is unknown for a generic GPD."""
        assert self.dev.device_type == UNKNOWN

    def test_always_available(self):
        """A GPD is always reported as available."""
        assert self.dev.available is True

    def test_last_seen_none_when_not_seen(self):
        """last_seen is None when the GPD has never been seen."""
        assert self.dev.last_seen is None

    def test_last_seen_converts_datetime_to_float(self):
        """last_seen converts the zigpy datetime to an epoch float."""
        ts = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)
        gpd = _make_zigpy_gpd(last_seen=ts)
        dev = GPDevice(gpd, _make_gateway())
        assert dev.last_seen == pytest.approx(ts.timestamp())


# ---------------------------------------------------------------------------
# P2 acceptance #1: device_info builds without crashing (join hot path)
# ---------------------------------------------------------------------------


class TestGPDeviceInfo:
    """device_info / extended_device_info build on the join hot path."""

    def setup_method(self):
        """Build a wrapper around a default synthetic GPD."""
        self.gpd = _make_zigpy_gpd()
        self.dev = GPDevice(self.gpd, _make_gateway())

    def test_device_info_builds(self):
        """device_info exposes the quirk identity without crashing."""
        info = self.dev.device_info
        assert info.manufacturer == "Philips"
        assert info.model == "Hue Tap"
        assert info.ieee == self.gpd.ieee

    def test_extended_device_info_builds(self):
        """extended_device_info returns empty topology for a GPD."""
        ext = self.dev.extended_device_info
        assert isinstance(ext, ExtendedDeviceInfo)
        assert ext.neighbors == []
        assert ext.routes == []
        assert ext.endpoint_names == []
        assert ext.entities == {}
        assert ext.active_coordinator is False

    def test_extended_device_info_inherits_device_info_fields(self):
        """extended_device_info carries the base device_info fields through."""
        ext = self.dev.extended_device_info
        assert ext.manufacturer == "Philips"
        assert ext.ieee == self.gpd.ieee


# ---------------------------------------------------------------------------
# P2 acceptance #2: device_automation_triggers returns quirk map
# ---------------------------------------------------------------------------


class TestGPDeviceAutomationTriggers:
    """device_automation_triggers merge quirk and built-in entries."""

    def setup_method(self):
        """Build a wrapper around a default synthetic GPD."""
        self.gpd = _make_zigpy_gpd()
        self.dev = GPDevice(self.gpd, _make_gateway())

    def test_includes_quirk_triggers(self):
        """Triggers include the quirk-provided entries."""
        triggers = self.dev.device_automation_triggers
        assert ("pressed", "button_1") in triggers

    def test_includes_device_offline(self):
        """Triggers include the built-in device_offline entry."""
        triggers = self.dev.device_automation_triggers
        assert ("device_offline", "device_offline") in triggers

    def test_more_than_offline_only(self):
        """Triggers contain more than just device_offline."""
        triggers = self.dev.device_automation_triggers
        assert len(triggers) > 1

    def test_trigger_command_shape(self):
        """Quirk contract: command=notification, cluster_id=0x0021, params.command_id."""
        entry = self.dev.device_automation_triggers[("pressed", "button_1")]
        assert entry["command"] == "notification"
        assert entry["cluster_id"] == 0x0021
        assert "command_id" in entry["params"]


# ---------------------------------------------------------------------------
# P2 acceptance (bonus): handle_gp_command emits via ZHA event bus
# ---------------------------------------------------------------------------


class TestHandleGpCommand:
    """handle_gp_command emits onto the ZHA event bus."""

    def setup_method(self):
        """Build a wrapper around a default synthetic GPD."""
        self.gpd = _make_zigpy_gpd()
        self.dev = GPDevice(self.gpd, _make_gateway())

    def test_handle_gp_command_emits_zha_event(self):
        """A GP command emits a zha_event carrying the quirk contract."""
        received = []
        self.dev.on_event("zha_event", received.append)

        event = MagicMock()
        event.command_id = 0x22

        self.dev.handle_gp_command(event)

        assert len(received) == 1
        payload = received[0].data
        assert payload["command"] == "notification"
        assert payload["cluster_id"] == 0x0021
        assert payload["params"]["command_id"] == 0x22

    def test_handle_gp_command_marks_available(self):
        """Receiving a GP command marks the device available again."""
        self.dev._available = False
        event = MagicMock()
        event.command_id = 0x10
        self.dev.handle_gp_command(event)
        assert self.dev._available is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestGPDeviceLifecycle:
    """Lifecycle status transitions for a GPD."""

    def test_initial_status_is_created(self):
        """A freshly built GPDevice starts in the CREATED state."""
        gpd = _make_zigpy_gpd()
        dev = GPDevice(gpd, _make_gateway())
        assert dev.status == DeviceStatus.CREATED

    def test_async_initialize_sets_initialized(self):
        """async_initialize moves the device to INITIALIZED."""
        gpd = _make_zigpy_gpd()
        dev = GPDevice(gpd, _make_gateway())
        asyncio.get_event_loop().run_until_complete(dev.async_initialize())
        assert dev.status == DeviceStatus.INITIALIZED
        assert dev._initialized is True
