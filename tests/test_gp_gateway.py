"""Tests for the gateway Green Power integration (wiring and UI delete).

Tests run without hardware or a real zigpy app - the gateway methods are
exercised directly on a minimal stub.

Wiring acceptance criteria:
  1. _gp_add inserts into _devices[ieee] BEFORE emitting ZHA_GW_MSG_DEVICE_FULL_INIT.
  2. _gp_remove pops from _devices and emits ZHA_GW_MSG_DEVICE_REMOVED.
  3. _gp_command routes the event to the right GPDevice.
  4. _gp_setup subscribes to the three named GP events.
  5. Persisted GPDs are rebuilt on _gp_setup.
  6. Regression: a GP command does NOT silently kill the listener chain.

UI-delete acceptance criteria:
  7. async_remove_device on a GP IEEE calls gp.remove_device(source_id).
  8. async_remove_device emits DeviceLeft on the GP manager (triggers DB row delete + gateway cleanup).
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from zigpy.quirks import _GP_REGISTRY, CustomGreenPowerDevice
from zigpy.zgp.device import GPDevice as ZigpyGPDevice
from zigpy.zgp.types import SecurityKeyType, SecurityLevel

from zha.application.const import ZHA_GW_MSG_DEVICE_FULL_INIT, ZHA_GW_MSG_DEVICE_REMOVED
from zha.application.gateway import (
    DeviceFullInitEvent,
    DevicePairingStatus,
    DeviceRemovedEvent,
    ExtendedDeviceInfoWithPairingStatus,
    Gateway,
)
from zha.event import EventBase

# ---------------------------------------------------------------------------
# Stub quirk - registered per-test via the _register_stub_quirk fixture below
# ---------------------------------------------------------------------------


class _StubHueTap(CustomGreenPowerDevice, priority=10):
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
_GP_REGISTRY.remove(_StubHueTap)


@pytest.fixture(autouse=True)
def _register_stub_quirk():
    """Register the stub GP quirk for the duration of each test only.

    Adds the stub to the global _GP_REGISTRY on setup and removes it on
    teardown, mirroring CustomGreenPowerDevice.__init_subclass__ (append + sort
    by priority) so there is no session-wide registry pollution.
    """
    _GP_REGISTRY.append(_StubHueTap)
    _GP_REGISTRY.sort(key=lambda c: c.priority)
    try:
        yield
    finally:
        if _StubHueTap in _GP_REGISTRY:
            _GP_REGISTRY.remove(_StubHueTap)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zigpy_gpd(source_id: int = 0x0040F4E4) -> ZigpyGPDevice:
    return ZigpyGPDevice(
        source_id=source_id,
        device_id=0x02,
        security_level=SecurityLevel.NoSecurity,
        security_key_type=SecurityKeyType.NoKey,
        frame_counter=0,
        last_seen=None,
    )


def _make_stub_gateway(gp_devices: dict | None = None):
    """Bare Gateway instance with just what the GP methods need.

    Uses object.__new__ to bypass Gateway.__init__, then initialises the
    EventBase side so that emit() works properly.
    """
    gw = object.__new__(Gateway)
    EventBase.__init__(gw)  # initialises _listeners / _global_listeners
    gw._devices = {}
    gw._gp_by_source = {}

    # Mock application_controller with a green_power manager.
    gp_mock = MagicMock()
    gp_mock.devices = gp_devices or {}
    # on_event: capture subscription calls as (event_name, handler) tuples.
    subscriptions: list[tuple[str, Callable]] = []

    def _on_event(name, handler):
        subscriptions.append((name, handler))
        return lambda: None  # unsub stub

    gp_mock.on_event.side_effect = _on_event
    gw._gp_subscriptions = subscriptions

    app_ctrl = MagicMock()
    app_ctrl.green_power = gp_mock
    gw.application_controller = app_ctrl

    return gw, gp_mock


# ---------------------------------------------------------------------------
# _gp_add
# ---------------------------------------------------------------------------


class TestGpAdd:
    """_gp_add builds and registers a GPDevice wrapper."""

    def test_device_in_devices_before_emit(self):
        """Insert into _devices[ieee] happens BEFORE the DEVICE_FULL_INIT emit."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()

        observed_during_emit = {}

        def _record_emit(event_name, event_obj=None):
            # Capture _devices state at emit time.
            observed_during_emit["has_device"] = gpd.ieee in gw._devices

        gw.on_event(ZHA_GW_MSG_DEVICE_FULL_INIT, _record_emit)

        Gateway._gp_add(gw, gpd)

        assert observed_during_emit.get("has_device") is True, (
            "_devices must contain the GPDevice before ZHA_GW_MSG_DEVICE_FULL_INIT fires"
        )

    def test_emits_device_full_init(self):
        """_gp_add emits the CONFIGURED then INITIALIZED full-init pair."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()

        emitted = []
        gw.on_event(ZHA_GW_MSG_DEVICE_FULL_INIT, emitted.append)

        Gateway._gp_add(gw, gpd)

        # Two emits: CONFIGURED (new_join=True) then INITIALIZED (new_join=False),
        # mirroring the normal device-join sequence so the frontend dialog unlocks.
        assert len(emitted) == 2
        assert all(isinstance(e, DeviceFullInitEvent) for e in emitted)

    def test_pairing_status_sequence(self):
        """The two full-init emits carry CONFIGURED then INITIALIZED status."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()

        emitted = []
        gw.on_event(ZHA_GW_MSG_DEVICE_FULL_INIT, emitted.append)

        Gateway._gp_add(gw, gpd)

        assert isinstance(emitted[0].device_info, ExtendedDeviceInfoWithPairingStatus)
        assert emitted[0].device_info.pairing_status is DevicePairingStatus.CONFIGURED
        assert emitted[0].new_join is True
        assert emitted[1].device_info.pairing_status is DevicePairingStatus.INITIALIZED
        assert emitted[1].new_join is False

    def test_indexed_by_source_id(self):
        """_gp_add indexes the wrapper by source_id."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)
        assert gpd.source_id in gw._gp_by_source

    def test_skips_if_already_present(self):
        """A second _gp_add for the same GPD does not replace the wrapper."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)
        first_device = gw._devices[gpd.ieee]
        Gateway._gp_add(gw, gpd)  # second call
        assert gw._devices[gpd.ieee] is first_device, (
            "second _gp_add should not replace"
        )

    def test_skips_unrecognised_device(self):
        """GPDs with no matching quirk are not added."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd(source_id=0xDEADBEEF)  # no quirk match
        emitted = []
        gw.on_event(ZHA_GW_MSG_DEVICE_FULL_INIT, emitted.append)

        Gateway._gp_add(gw, gpd)

        assert len(emitted) == 0
        assert gpd.ieee not in gw._devices


# ---------------------------------------------------------------------------
# _gp_remove
# ---------------------------------------------------------------------------


class TestGpRemove:
    """_gp_remove tears down a GPDevice wrapper."""

    def test_removes_from_devices(self):
        """_gp_remove pops the wrapper out of _devices."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)
        assert gpd.ieee in gw._devices

        Gateway._gp_remove(gw, gpd)
        assert gpd.ieee not in gw._devices

    def test_removes_from_source_index(self):
        """_gp_remove pops the wrapper out of the source_id index."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)
        Gateway._gp_remove(gw, gpd)
        assert gpd.source_id not in gw._gp_by_source

    def test_emits_device_removed(self):
        """_gp_remove emits ZHA_GW_MSG_DEVICE_REMOVED."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)

        emitted = []
        gw.on_event(ZHA_GW_MSG_DEVICE_REMOVED, emitted.append)

        Gateway._gp_remove(gw, gpd)

        assert len(emitted) == 1
        assert isinstance(emitted[0], DeviceRemovedEvent)

    def test_remove_noop_if_not_present(self):
        """Removing a GPD that was never added should not raise."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_remove(gw, gpd)  # should not raise


# ---------------------------------------------------------------------------
# _gp_command - routing
# ---------------------------------------------------------------------------


class TestGpCommand:
    """_gp_command routes incoming GP commands to the right wrapper."""

    def test_routes_command_to_device(self):
        """A GP command is delivered to the matching GPDevice as a zha_event."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)

        received = []
        zha_dev = gw._devices[gpd.ieee]
        zha_dev.on_event("zha_event", received.append)

        event = MagicMock()
        event.device.source_id = gpd.source_id
        event.command_id = 0x22

        Gateway._gp_command(gw, event)

        assert len(received) == 1
        payload = received[0].data
        assert payload["command"] == "notification"
        assert payload["params"]["command_id"] == 0x22

    def test_command_for_unknown_source_is_ignored(self):
        """No error if the SrcID is not in _gp_by_source."""
        gw, _ = _make_stub_gateway()
        event = MagicMock()
        event.device.source_id = 0x11223344
        Gateway._gp_command(gw, event)  # should not raise


# ---------------------------------------------------------------------------
# Regression - listener chain survives a GP command
# ---------------------------------------------------------------------------


class TestListenerSurvival:
    """The named-event subscription survives repeated GP commands."""

    def test_gp_command_does_not_kill_listener_chain(self):
        """A second command after the first must still arrive."""
        gw, _ = _make_stub_gateway()
        gpd = _make_zigpy_gpd()
        Gateway._gp_add(gw, gpd)

        received = []
        zha_dev = gw._devices[gpd.ieee]
        zha_dev.on_event("zha_event", received.append)

        for cmd_id in [0x22, 0x10, 0x11]:
            event = MagicMock()
            event.device.source_id = gpd.source_id
            event.command_id = cmd_id
            Gateway._gp_command(gw, event)

        assert len(received) == 3, "Listener chain must survive multiple GP commands"


# ---------------------------------------------------------------------------
# _gp_setup - event subscription + persisted-device rebuild
# ---------------------------------------------------------------------------


class TestGpSetup:
    """_gp_setup subscribes to GP events and rebuilds persisted devices."""

    def test_subscribes_to_three_events(self):
        """_gp_setup subscribes to the three named GP manager events."""
        gw, gp_mock = _make_stub_gateway()

        Gateway._gp_setup(gw)

        subscribed_names = {name for name, _ in gw._gp_subscriptions}
        assert "gp_device_joined" in subscribed_names
        assert "gp_device_left" in subscribed_names
        assert "gp_command_received" in subscribed_names

    def test_rebuilds_persisted_gp_devices(self):
        """GPDs already in green_power.devices are wrapped on _gp_setup."""
        gpd = _make_zigpy_gpd()
        gw, _ = _make_stub_gateway(gp_devices={gpd.source_id: gpd})

        Gateway._gp_setup(gw)

        assert gpd.ieee in gw._devices

    def test_noop_when_no_green_power(self):
        """If the app controller has no green_power, _gp_setup is a no-op."""
        gw, _ = _make_stub_gateway()
        # spec=[] means the mock has no attributes, so hasattr(ctrl, "green_power") is False.
        ctrl = MagicMock(spec=[])
        gw.application_controller = ctrl
        Gateway._gp_setup(gw)  # should not raise
