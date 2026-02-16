from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from pymodbus.client import ModbusTcpClient

log = logging.getLogger(__name__)


class ModbusController:
    """
    Small wrapper around pymodbus client:
    - thread-offloads sync pymodbus calls via asyncio.to_thread()
    - serializes mask read/modify/write with a lock
    - keeps last_error for diagnostics
    """

    REG_STATE_MASK: int = 468
    REG_SET_MASK: int = 470

    def __init__(
        self,
        host: str,
        port: int,
        slave_id: int,
        timeout_sec: float = 1.0,
        connect_timeout_sec: float = 3.0,
        retries: int = 2,
        retry_delay_sec: float = 0.15,
    ):
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.client = ModbusTcpClient(host=self.host, port=self.port, timeout=timeout_sec)
        self.lock = asyncio.Lock()
        self.last_error: Optional[str] = None

        self.connect_timeout_sec = connect_timeout_sec
        self.retries = retries
        self.retry_delay_sec = retry_delay_sec

        # Track active pulse tasks per bit to avoid race conditions
        self._pulse_tasks: dict[int, asyncio.Task] = {}
        self._pulse_tasks_lock = asyncio.Lock()

        # Cache the correct unit kwarg for this pymodbus version
        self._unit_kw_cache: Optional[str] = None

    def is_connected(self) -> bool:
        return self.client.is_socket_open()

    def disconnect(self) -> None:
        if self.client.is_socket_open():
            self.client.close()
            log.info("Disconnected from Modbus server.")

    def _call_with_slave(self, fn: Callable[..., Any], /, **kwargs) -> Any:
        """
        Pymodbus compatibility shim:
        different versions use different kwarg names for unit/slave id.
        """
        if self._unit_kw_cache:
            return fn(**kwargs, **{self._unit_kw_cache: self.slave_id})

        last_type_error: Optional[TypeError] = None
        for unit_kw in ("device_id", "slave", "unit"):
            try:
                result = fn(**kwargs, **{unit_kw: self.slave_id})
                self._unit_kw_cache = unit_kw
                return result
            except TypeError as e:
                last_type_error = e
                continue
        if last_type_error:
            log.debug("Unit kwarg mismatch; falling back without unit: %s", last_type_error)
        return fn(**kwargs)

    def _backoff_sleep(self, attempt_index: int) -> None:
        # Exponential-ish backoff: delay, 2*delay, 4*delay...
        delay = self.retry_delay_sec * (2 ** attempt_index)
        time.sleep(delay)

    def _read_holding_registers_sync(self, address: int, count: int):
        return self._call_with_slave(
            self.client.read_holding_registers,
            address=address,
            count=count,
        )

    def _write_register_sync(self, address: int, value: int):
        return self._call_with_slave(
            self.client.write_register,
            address=address,
            value=value,
        )

    def _read_mask_sync(self) -> Optional[int]:
        last_rr = None
        for attempt in range(max(self.retries, 0) + 1):
            try:
                rr = self._read_holding_registers_sync(address=self.REG_STATE_MASK, count=1)
                last_rr = rr
                if not rr.isError():
                    return rr.registers[0]
                self.last_error = f"read_mask_error: {rr}"
                log.warning("Modbus read mask failed (attempt %s/%s): %s", attempt + 1, self.retries + 1, rr)
            except Exception as e:
                self.last_error = f"read_mask_exception: {e}"
                log.warning("Modbus read mask exception (attempt %s/%s): %s", attempt + 1, self.retries + 1, e)
            if attempt < self.retries:
                self._backoff_sleep(attempt)
        if last_rr is not None:
            log.error("Modbus error reading mask (%s) after retries: %s", self.REG_STATE_MASK, last_rr)
        return None

    def _write_mask_sync(self, new_mask: int) -> bool:
        last_wr = None
        for attempt in range(max(self.retries, 0) + 1):
            try:
                wr = self._write_register_sync(address=self.REG_SET_MASK, value=new_mask)
                last_wr = wr
                if not wr.isError():
                    return True
                self.last_error = f"write_mask_error: {wr}"
                log.warning("Modbus write mask failed (attempt %s/%s): %s", attempt + 1, self.retries + 1, wr)
            except Exception as e:
                self.last_error = f"write_mask_exception: {e}"
                log.warning("Modbus write mask exception (attempt %s/%s): %s", attempt + 1, self.retries + 1, e)
            if attempt < self.retries:
                self._backoff_sleep(attempt)
        if last_wr is not None:
            log.error("Modbus error writing mask (%s) after retries: %s", self.REG_SET_MASK, last_wr)
        return False

    def _set_do_sync(self, bit: int, state: bool) -> bool:
        if bit < 0 or bit > 15:
            self.last_error = f"invalid_bit: {bit}"
            log.error("Invalid DO bit: %s (expected 0..15)", bit)
            return False

        current_mask = self._read_mask_sync()
        if current_mask is None:
            return False

        if state:
            new_mask = current_mask | (1 << bit)
        else:
            new_mask = current_mask & ~(1 << bit)

        if self._write_mask_sync(new_mask):
            log.info("DO%s -> %s (mask=%s)", bit + 1, "ON" if state else "OFF", bin(new_mask))
            return True
        return False

    async def connect(self) -> bool:
        log.info("Connecting to Modbus server at %s:%s ...", self.host, self.port)
        try:
            # NOTE: to_thread cannot be force-killed; timeout protects startup from hanging.
            is_connected = await asyncio.wait_for(
                asyncio.to_thread(self.client.connect),
                timeout=self.connect_timeout_sec,
            )
        except Exception as e:
            self.last_error = f"connect_exception: {e}"
            log.exception("Modbus connect exception")
            return False

        if not is_connected:
            self.last_error = "connect_failed"
            log.error("Failed to connect to Modbus server (slave_id=%s).", self.slave_id)
            return False

        # Optional init: set DO3/DO4 to logic mode (same behavior as before)
        try:
            await asyncio.to_thread(self._write_register_sync, 274, 0)
            await asyncio.to_thread(self._write_register_sync, 275, 0)
            log.info("Init OK: Reg 274/275 set to 0 (logic mode).")
        except Exception as e:
            self.last_error = f"init_failed: {e}"
            log.exception("Initialization write failed")
        return True

    async def schedule_pulse(self, bit: int, duration: float) -> None:
        """
        Ensure only one active pulse task per bit.
        If a new request comes for the same bit, cancel the previous task.
        """
        async with self._pulse_tasks_lock:
            prev = self._pulse_tasks.get(bit)
            if prev is not None and not prev.done():
                prev.cancel()
            task = asyncio.create_task(self.pulse_do(bit, duration))

            # cleanup callback
            def _cleanup(t: asyncio.Task) -> None:
                if self._pulse_tasks.get(bit) is t:
                    self._pulse_tasks.pop(bit, None)

            task.add_done_callback(_cleanup)
            self._pulse_tasks[bit] = task

    async def pulse_do(self, bit: int, duration: float) -> None:
        """
        Pulse a digital output bit for `duration` seconds.
        """
        try:
            async with self.lock:
                await asyncio.to_thread(self._set_do_sync, bit, True)
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            log.warning("Pulse on DO%s cancelled.", bit + 1)
            raise
        except Exception as e:
            self.last_error = f"pulse_failed: {e}"
            log.exception("Pulse failed")
        finally:
            # Guaranteed cleanup: attempt to turn off the bit
            try:
                async with self.lock:
                    await asyncio.to_thread(self._set_do_sync, bit, False)
                log.info("Pulse on DO%s finished.", bit + 1)
            except Exception as e:
                self.last_error = f"pulse_cleanup_failed: {e}"
                log.exception("Failed to cleanup pulse on DO%s", bit + 1)

    async def ping(self) -> bool:
        """
        Lightweight "real" healthcheck: try reading mask register.
        """
        if not self.is_connected():
            return False
        try:
            val = await asyncio.to_thread(self._read_mask_sync)
            return val is not None
        except Exception as e:
            self.last_error = f"ping_failed: {e}"
            return False

    async def shutdown(self, bits_to_clear: list[int]) -> None:
        """
        Graceful shutdown:
        - cancel all pulse tasks
        - best-effort: clear configured bits so lamps do not remain ON
        """
        # cancel tasks
        async with self._pulse_tasks_lock:
            tasks = list(self._pulse_tasks.values())
            self._pulse_tasks.clear()
        for t in tasks:
            if not t.done():
                t.cancel()

        # best-effort cleanup
        if self.is_connected():
            for bit in bits_to_clear:
                try:
                    async with self.lock:
                        await asyncio.to_thread(self._set_do_sync, bit, False)
                except Exception as e:
                    self.last_error = f"shutdown_clear_failed: {e}"
                    log.exception("Failed to clear bit %s during shutdown", bit)
