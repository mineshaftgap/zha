"""ZHA device wrapper for a Green Power Device (GPD).

A commissioned GPD (e.g. a Philips Hue Tap) has no endpoints, no ZDO, and no
network address - it is a receive-only event source.  This wrapper presents the
same ``Device`` interface ha-core consumes, so a GPD can be registered without
any ha-core changes.

Implementation notes:
  * ``Device.__init__`` is bypassed because it iterates
    ``zigpy_device.endpoints`` and reads ``node_desc``/``is_mains_powered``,
    none of which a GPD has.  ``EventBase.__init__`` is called directly and only
    the attributes the rest of the class touches are set.
  * ``self._zigpy_device`` is a small stub (``endpoints={}``) so base-class
    methods that iterate endpoints directly (``async_get_clusters``,
    ``ZHADeviceProxy.zha_device_info``) iterate nothing instead of raising
    ``AttributeError``.  All identity properties are overridden.
  * ``device_info`` is inherited unchanged - it only reads the overridden
    properties.
  * ``extended_device_info`` is overridden to skip the topology/endpoints
    lookup, which has no meaning for a GPD and would otherwise raise.
  * Identity and automation triggers come from the matched GP quirk
    (``zigpy.quirks.get_green_power_quirk``); the wrapper is quirk-agnostic.

``last_seen`` is converted from the zigpy ``GPDevice.last_seen`` ``datetime`` to
an epoch ``float`` because the inherited ``device_info`` passes it to
``time.localtime``.  The identity properties (``name``/``manufacturer``/
``model``/``device_type``) override ``cached_property`` attributes on the base
class, so they are declared as plain ``@property`` rather than set as instance
attributes in ``__init__``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from zigpy.quirks import get_green_power_quirk
from zigpy.types.named import EUI64, NWK

from zha.application.const import (
    POWER_BATTERY_OR_UNKNOWN,
    UNKNOWN,
    UNKNOWN_MANUFACTURER,
    UNKNOWN_MODEL,
)
from zha.event import EventBase
from zha.zigbee.device import Device, DeviceStatus, ExtendedDeviceInfo

if TYPE_CHECKING:
    from zigpy.zgp.device import GPDevice as ZigpyGPDevice

    from zha.application.gateway import Gateway

_LOGGER = logging.getLogger(__name__)

# GP transport constants
_GP_ENDPOINT = 242
_GP_CLUSTER_ID = 0x0021  # Green Power cluster

# GP sentinel network address - a GPD has no real nwk; this mirrors the value
# zigpy uses for the GP endpoint.
_GP_SENTINEL_NWK = NWK(0xFFFE)

# A GPD is energy-harvesting: an untouched-but-healthy Tap and a removed Tap
# look identical on the wire, so timeout-based availability is meaningless.
# Use an effectively-infinite window (10 years, in seconds) so the device is
# always reported as available.
_AVAILABILITY_TIMEOUT_DISABLED = 60 * 60 * 24 * 365 * 10


class _ZigpyDeviceStub:
    """Minimal stand-in for ``self._zigpy_device``.

    Base-class methods that iterate endpoints (``async_get_clusters``,
    ``ZHADeviceProxy.zha_device_info``) access ``self._zigpy_device`` directly
    rather than through overridden properties.  Providing an empty stub means
    they iterate nothing instead of raising ``AttributeError``.
    """

    endpoints: dict = {}


_ZIGPY_DEVICE_STUB = _ZigpyDeviceStub()


class GPDevice(Device):
    """ZHA device wrapper for a commissioned Green Power Device."""

    def __init__(self, gpd: ZigpyGPDevice, gateway: Gateway) -> None:
        """Initialize the wrapper from a zigpy GPDevice and its gateway."""
        # Bypass Device.__init__ - it requires a real zigpy Device with endpoints.
        EventBase.__init__(self)

        self._gpd = gpd
        self._gateway = gateway
        # Satisfies base-class code that accesses self._zigpy_device directly
        # (e.g. async_get_clusters, ZHADeviceProxy.zha_device_info).
        self._zigpy_device = _ZIGPY_DEVICE_STUB

        quirk = get_green_power_quirk(gpd)
        self._quirk = quirk

        # Instance attrs the base class / consumers read via normal paths.
        self.unique_id = str(gpd.ieee)
        self._platform_entities: dict = {}
        self._pending_entities: list = []
        self._discovered_entities: list = []
        self._endpoints: dict = {}
        self._on_remove_callbacks: list = []
        self._initialized = False
        self._available = True
        self._on_network = True
        self._checkins_missed_count = 0
        self._firmware_version = None
        self.semaphore = asyncio.Semaphore(3)

        self.quirk_applied = quirk is not None
        self.quirk_class = f"{quirk.__module__}.{quirk.__name__}" if quirk else "none"
        self.exposes_features: set[str] = set()
        self.consider_unavailable_time = _AVAILABILITY_TIMEOUT_DISABLED
        self.status = DeviceStatus.CREATED

    @classmethod
    def new(cls, gpd: ZigpyGPDevice, gateway: Gateway) -> GPDevice:
        """Create a GPDevice wrapper, mirroring the ``Device.new`` signature."""
        return cls(gpd, gateway)

    # ---------------------------------------------------------------- identity
    # Every property below overrides a cached_property or property on the base
    # that would otherwise read self._zigpy_device (which doesn't exist here).

    @property
    def ieee(self) -> EUI64:
        """Return the synthetic EUI64 derived from the GP source_id."""
        return self._gpd.ieee

    @property
    def nwk(self) -> NWK:
        """Return the GP sentinel network address (GPDs have no real nwk)."""
        return _GP_SENTINEL_NWK

    @property
    def manufacturer(self) -> str:
        """Return the manufacturer reported by the matched quirk."""
        return self._quirk.manufacturer if self._quirk else UNKNOWN_MANUFACTURER

    @property
    def model(self) -> str:
        """Return the model reported by the matched quirk."""
        return self._quirk.model if self._quirk else UNKNOWN_MODEL

    @property
    def name(self) -> str:
        """Return the display name ("<manufacturer> <model>")."""
        return f"{self.manufacturer} {self.model}"

    @property
    def manufacturer_code(self) -> int | None:
        """Return the GP manufacturer id, if any."""
        return self._gpd.manufacturer_id

    @property
    def quirk_metadata(self) -> None:
        """Return None; GP quirks carry no v2 quirk metadata."""
        return None

    @property
    def device_alerts(self) -> list:
        """Return an empty alert list; GPDs report no alerts."""
        return []

    @property
    def is_mains_powered(self) -> bool:
        """Return False; a GPD is energy-harvesting, never mains powered."""
        return False

    @property
    def power_source(self) -> str:
        """Return the power source label (kinetic/energy-harvesting)."""
        return POWER_BATTERY_OR_UNKNOWN

    @property
    def device_type(self) -> str:
        """Return the device type (unknown for a generic GPD)."""
        return UNKNOWN

    @property
    def is_coordinator(self) -> bool:
        """Return False; a GPD is never the coordinator."""
        return False

    @property
    def is_active_coordinator(self) -> bool:
        """Return False; a GPD is never the active coordinator."""
        return False

    @property
    def is_router(self) -> bool:
        """Return False; a GPD does not route."""
        return False

    @property
    def is_end_device(self) -> bool:
        """Return True; a GPD behaves like an end device."""
        return True

    @property
    def skip_configuration(self) -> bool:
        """Return True; a receive-only GPD has nothing to configure."""
        return True

    @property
    def lqi(self) -> None:
        """Return None; no LQI is available for a GPD."""
        return None

    @property
    def rssi(self) -> None:
        """Return None; no RSSI is available for a GPD."""
        return None

    @property
    def last_seen(self) -> float | None:
        """Return last_seen as an epoch float (base passes it to time.localtime)."""
        # zigpy GPDevice.last_seen is a datetime; convert to epoch seconds.
        if self._gpd.last_seen is None:
            return None
        return self._gpd.last_seen.timestamp()

    @property
    def zigbee_signature(self) -> dict:
        """Return a signature dict identifying the GPD by source/device id."""
        return {
            "source_id": f"0x{self._gpd.source_id:08X}",
            "device_id": f"0x{self._gpd.device_id:02X}",
            "manufacturer": self.manufacturer,
            "model": self.model,
        }

    @property
    def device_automation_triggers(self) -> dict:
        """Return device automation triggers, merged from the matched quirk."""
        base = {
            ("device_offline", "device_offline"): {
                "device_event_type": "device_offline"
            }
        }
        if self._quirk:
            base.update(self._quirk.device_automation_triggers)
        return base

    @property
    def device_automation_commands(self) -> list:
        """Return an empty command list; GP triggers are event-only."""
        return []

    # ---------------------------------------------------------- extended info
    @property
    def extended_device_info(self) -> ExtendedDeviceInfo:
        """Return extended info without the topology/endpoints lookup.

        The base implementation reads ``topology.neighbors``/``routes`` and
        ``self.device.endpoints`` - all absent for a GPD.  This runs on the join
        hot path (the gateway fires ``DeviceFullInitEvent`` which ha-core
        unpacks), so it must not raise.
        """
        return ExtendedDeviceInfo(
            **self.device_info.__dict__,
            active_coordinator=False,
            entities={},
            neighbors=[],
            routes=[],
            endpoint_names=[],
        )

    # ---------------------------------------------------------------- gateway
    @property
    def gateway(self) -> Gateway:
        """Return the gateway this device is registered with."""
        return self._gateway

    # --------------------------------------------------------------- lifecycle
    async def async_initialize(self, from_cache: bool = False) -> None:
        """Mark the device initialized; a GPD needs no cluster/entity discovery."""
        self.debug("GP device initialized (no entities)")
        self.status = DeviceStatus.INITIALIZED
        self._initialized = True

    async def async_configure(self) -> None:
        """Do nothing; a receive-only GPD has nothing to configure."""

    # --------------------------------------------------------------- hot path
    def handle_gp_command(self, event) -> None:
        """Translate a zigpy GP CommandReceived into a ZHA bus event.

        Emits the quirk contract so the device automation triggers fire:
        ``command="notification"``, ``cluster_id=0x0021``,
        ``params.command_id=<N>``.
        """
        self.emit_zha_event(
            {
                "endpoint_id": _GP_ENDPOINT,
                "cluster_id": _GP_CLUSTER_ID,
                "command": "notification",
                "args": [],
                "params": {"command_id": int(event.command_id)},
            }
        )
        self._available = True

    # ------------------------------------------------------------- logging
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a message."""
        msg = f"[%s](%s): {msg}"
        args = (self.nwk, self.model) + args
        _LOGGER.log(level, msg, *args, **kwargs)
