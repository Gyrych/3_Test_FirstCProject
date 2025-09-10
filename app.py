import logging
import os
import threading
import json
import asyncio
from typing import Any, Dict, Literal, Optional, Tuple, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Lazy import to allow startup on systems without python gpiod until actually used
try:
    import gpiod  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    gpiod = None  # type: ignore

DEFAULT_CHIP_NAME = os.environ.get("GPIO_DEFAULT_CHIP", "gpiochip0")
CONSUMER_LABEL = os.environ.get("GPIO_CONSUMER_LABEL", "mcp-gpio")


class GPIOError(RuntimeError):
    pass


class GpiodV1Manager:
    """Manager for libgpiod v1.x Python bindings.

    - Caches requested lines per (chip_name, line_offset)
    - Ensures thread-safe access
    - Automatically requests/re-requests lines with the correct direction
    """

    def __init__(self, consumer: str = CONSUMER_LABEL) -> None:
        if gpiod is None:
            raise GPIOError(
                "Python gpiod module not found. Install python3-libgpiod (apt) or pip install gpiod."
            )
        if not hasattr(gpiod, "Chip"):
            raise GPIOError("Unsupported gpiod Python API (missing Chip). This server targets libgpiod v1.x.")

        self._consumer: str = consumer
        self._chips: Dict[str, "gpiod.Chip"] = {}
        self._lines: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _get_chip(self, chip_name: str) -> "gpiod.Chip":
        with self._lock:
            if chip_name in self._chips:
                return self._chips[chip_name]
            try:
                chip = gpiod.Chip(chip_name)
            except Exception as exc:  # noqa: BLE001
                raise GPIOError(f"Failed to open chip '{chip_name}': {exc}") from exc
            self._chips[chip_name] = chip
            return chip

    def _ensure_mode(self, chip_name: str, line_offset: int, mode: Literal["in", "out"]) -> None:
        key = (chip_name, line_offset)
        with self._lock:
            entry = self._lines.get(key)
            if entry and entry.get("direction") == mode:
                return

            # Release any existing request with wrong direction
            if entry:
                try:
                    line = entry["line"]
                    if hasattr(line, "release"):
                        line.release()  # type: ignore[attr-defined]
                except Exception:
                    pass
                finally:
                    self._lines.pop(key, None)

            chip = self._get_chip(chip_name)
            try:
                line = chip.get_line(line_offset)
            except Exception as exc:  # noqa: BLE001
                raise GPIOError(
                    f"Failed to get line offset {line_offset} on '{chip_name}': {exc}"
                ) from exc

            req_type = gpiod.LINE_REQ_DIR_IN if mode == "in" else gpiod.LINE_REQ_DIR_OUT
            try:
                line.request(consumer=self._consumer, type=req_type)
            except Exception as exc:  # noqa: BLE001
                raise GPIOError(
                    f"Request for {chip_name}:{line_offset} as {mode} failed: {exc}"
                ) from exc

            self._lines[key] = {"line": line, "direction": mode}

    def set_mode(self, chip_name: str, line_offset: int, mode: Literal["in", "out"]) -> None:
        if mode not in ("in", "out"):
            raise GPIOError("mode must be either 'in' or 'out'")
        self._ensure_mode(chip_name, line_offset, mode)

    def write(self, chip_name: str, line_offset: int, value: int) -> None:
        logical = 1 if int(value) else 0
        self._ensure_mode(chip_name, line_offset, "out")
        key = (chip_name, line_offset)
        with self._lock:
            entry = self._lines.get(key)
            if not entry:
                raise GPIOError("internal error: line not requested after ensure_mode")
            line = entry["line"]
            try:
                line.set_value(logical)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                raise GPIOError(
                    f"Failed to set value on {chip_name}:{line_offset}: {exc}"
                ) from exc

    def read(self, chip_name: str, line_offset: int) -> int:
        # Prefer input mode for reads to avoid contention
        self._ensure_mode(chip_name, line_offset, "in")
        key = (chip_name, line_offset)
        with self._lock:
            entry = self._lines.get(key)
            if not entry:
                raise GPIOError("internal error: line not requested after ensure_mode")
            line = entry["line"]
            try:
                value = line.get_value()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                raise GPIOError(
                    f"Failed to get value from {chip_name}:{line_offset}: {exc}"
                ) from exc
        return int(value)

    def cleanup(self) -> None:
        with self._lock:
            for key, entry in list(self._lines.items()):
                try:
                    line = entry.get("line")
                    if line is not None and hasattr(line, "release"):
                        line.release()  # type: ignore[attr-defined]
                except Exception:
                    pass
                finally:
                    self._lines.pop(key, None)
            for chip_name, chip in list(self._chips.items()):
                try:
                    if hasattr(chip, "close"):
                        chip.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                finally:
                    self._chips.pop(chip_name, None)


def create_gpio_manager() -> GpiodV1Manager:
    # Currently we target libgpiod v1.x which is prevalent on Debian/Ubuntu LTS.
    return GpiodV1Manager()


class PinBase(BaseModel):
    pin: int = Field(..., ge=0, description="GPIO line offset on the chip")
    chip: str = Field(DEFAULT_CHIP_NAME, description="gpiochip device name, e.g., gpiochip0")

    @field_validator("chip")
    @classmethod
    def validate_chip(cls, v: str) -> str:
        if not v.startswith("gpiochip"):
            raise ValueError("chip must look like 'gpiochipN'")
        return v


class WriteRequest(PinBase):
    value: int = Field(..., ge=0, le=1, description="Logical output value (0 or 1)")


class ModeRequest(PinBase):
    mode: Literal["in", "out"]


class ReadRequest(PinBase):
    pass


class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("radxa-gpio")

app = FastAPI(
    title="Radxa 3A GPIO API",
    description=(
        "Control RK3568 GPIO via libgpiod. Endpoints: gpio_write, gpio_read, gpio_set_mode."
    ),
    version="0.1.0",
)

# Instantiate manager at module load
try:
    gpio_manager = create_gpio_manager()
except Exception as exc:  # noqa: BLE001
    # Delay hard failure to first request, but log now
    logger.warning("GPIO manager not initialized: %s", exc)
    gpio_manager = None  # type: ignore


@app.get("/health", response_model=APIResponse, tags=["system"])
def health() -> APIResponse:
    return APIResponse(success=True, data={"status": "ok"})


@app.post("/gpio/set_mode", response_model=APIResponse, tags=["gpio"])
def gpio_set_mode(req: ModeRequest) -> APIResponse:
    try:
        if gpio_manager is None:
            raise GPIOError("gpiod module not available on this system")
        gpio_manager.set_mode(req.chip, req.pin, req.mode)
        return APIResponse(success=True, data={"chip": req.chip, "pin": req.pin, "mode": req.mode})
    except GPIOError as exc:
        logger.error("set_mode error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/gpio/write", response_model=APIResponse, tags=["gpio"])
def gpio_write(req: WriteRequest) -> APIResponse:
    try:
        if gpio_manager is None:
            raise GPIOError("gpiod module not available on this system")
        gpio_manager.write(req.chip, req.pin, req.value)
        return APIResponse(success=True, data={"chip": req.chip, "pin": req.pin, "value": int(req.value)})
    except GPIOError as exc:
        logger.error("write error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/gpio/read", response_model=APIResponse, tags=["gpio"])
def gpio_read(req: ReadRequest) -> APIResponse:
    try:
        if gpio_manager is None:
            raise GPIOError("gpiod module not available on this system")
        value = gpio_manager.read(req.chip, req.pin)
        return APIResponse(success=True, data={"chip": req.chip, "pin": req.pin, "value": int(value)})
    except GPIOError as exc:
        logger.error("read error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.on_event("shutdown")
def on_shutdown() -> None:
    try:
        if gpio_manager is not None:
            gpio_manager.cleanup()
    except Exception:
        pass


# --------------------------
# Minimal MCP SSE transport
# --------------------------

# Connected client queues for SSE messages
_mcp_clients: List[asyncio.Queue[str]] = []


def _mcp_broadcast(message: Dict[str, Any]) -> None:
    data = json.dumps(message, ensure_ascii=False)
    for q in list(_mcp_clients):
        try:
            q.put_nowait(data)
        except Exception:
            # Drop stalled clients
            try:
                _mcp_clients.remove(q)
            except ValueError:
                pass


async def _sse_event_stream(queue: asyncio.Queue[str]):
    try:
        while True:
            data = await queue.get()
            yield f"event: message\n".encode()
            # SSE requires each data line prefixed with 'data:'
            for line in data.splitlines() or [""]:
                yield f"data: {line}\n".encode()
            yield b"\n"
    except asyncio.CancelledError:  # pragma: no cover
        return


@app.get("/mcp")
async def mcp_sse(request: Request) -> StreamingResponse:
    queue: asyncio.Queue[str] = asyncio.Queue()
    _mcp_clients.append(queue)

    async def stream_wrapper():
        try:
            async for chunk in _sse_event_stream(queue):
                # Client disconnected?
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            try:
                _mcp_clients.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


@app.post("/mcp")
async def mcp_post(payload: Dict[str, Any]) -> Dict[str, str]:
    """Receive client->server JSON-RPC messages and respond via SSE."""
    try:
        if not isinstance(payload, dict):
            return {"status": "ignored"}
        method = payload.get("method")
        message_id = payload.get("id")
        params = payload.get("params") or {}

        # Handle MCP JSON-RPC methods
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "radxa-gpio", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
            _mcp_broadcast({"jsonrpc": "2.0", "id": message_id, "result": result})
        elif method == "tools/list":
            tools = [
                {
                    "name": "gpio_write",
                    "description": "Set GPIO pin level via libgpiod",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "pin": {"type": "integer", "minimum": 0},
                            "value": {"type": "integer", "enum": [0, 1]},
                            "chip": {"type": "string", "default": DEFAULT_CHIP_NAME},
                        },
                        "required": ["pin", "value"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "gpio_read",
                    "description": "Read GPIO pin level via libgpiod",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "pin": {"type": "integer", "minimum": 0},
                            "chip": {"type": "string", "default": DEFAULT_CHIP_NAME},
                        },
                        "required": ["pin"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "gpio_set_mode",
                    "description": "Configure GPIO pin direction (in/out) via libgpiod",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "pin": {"type": "integer", "minimum": 0},
                            "mode": {"type": "string", "enum": ["in", "out"]},
                            "chip": {"type": "string", "default": DEFAULT_CHIP_NAME},
                        },
                        "required": ["pin", "mode"],
                        "additionalProperties": False,
                    },
                },
            ]
            _mcp_broadcast({"jsonrpc": "2.0", "id": message_id, "result": {"tools": tools}})
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            try:
                if name == "gpio_write":
                    pin = int(arguments["pin"])  # type: ignore[index]
                    value = int(arguments["value"])  # type: ignore[index]
                    chip = str(arguments.get("chip", DEFAULT_CHIP_NAME))
                    if gpio_manager is None:
                        raise GPIOError("gpiod not available")
                    gpio_manager.write(chip, pin, value)
                    result_text = json.dumps({"chip": chip, "pin": pin, "value": int(value)})
                elif name == "gpio_read":
                    pin = int(arguments["pin"])  # type: ignore[index]
                    chip = str(arguments.get("chip", DEFAULT_CHIP_NAME))
                    if gpio_manager is None:
                        raise GPIOError("gpiod not available")
                    value = gpio_manager.read(chip, pin)
                    result_text = json.dumps({"chip": chip, "pin": pin, "value": int(value)})
                elif name == "gpio_set_mode":
                    pin = int(arguments["pin"])  # type: ignore[index]
                    mode = str(arguments["mode"])  # type: ignore[index]
                    chip = str(arguments.get("chip", DEFAULT_CHIP_NAME))
                    if mode not in ("in", "out"):
                        raise GPIOError("mode must be 'in' or 'out'")
                    if gpio_manager is None:
                        raise GPIOError("gpiod not available")
                    gpio_manager.set_mode(chip, pin, mode)  # type: ignore[arg-type]
                    result_text = json.dumps({"chip": chip, "pin": pin, "mode": mode})
                else:
                    raise GPIOError(f"unknown tool: {name}")

                tool_result = {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                }
                _mcp_broadcast({"jsonrpc": "2.0", "id": message_id, "result": tool_result})
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                tool_result = {"content": [{"type": "text", "text": error_text}], "isError": True}
                _mcp_broadcast({"jsonrpc": "2.0", "id": message_id, "result": tool_result})
        elif method == "ping":
            _mcp_broadcast({"jsonrpc": "2.0", "id": message_id, "result": {}})
        else:
            # Unknown method -> JSON-RPC error
            _mcp_broadcast(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )

        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.error("mcp_post error: %s", exc)
        return {"status": "error", "detail": str(exc)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)