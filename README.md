# Radxa 3A (RK3568) GPIO MCP Plugin

FastAPI service using libgpiod to control GPIO pins on Radxa 3A. Exposes both simple REST JSON endpoints and Model Context Protocol (MCP) tools over SSE so an LLM can operate GPIO directly.

## Requirements
- Debian/Ubuntu on Radxa 3A
- libgpiod v1.x userspace bindings

## Install
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-libgpiod gpiod
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3-libgpiod` is unavailable, you can instead:
```bash
pip install gpiod
```

## Run
```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```
Server listens on 0.0.0.0:8000.

Optional environment variables:
- `GPIO_DEFAULT_CHIP` (default: `gpiochip0`)
- `GPIO_CONSUMER_LABEL` (default: `mcp-gpio`)

## REST API
All endpoints accept/return JSON. Default demo pin is 12.

- POST `/gpio/set_mode`
  - body: `{ "pin": 12, "mode": "out", "chip": "gpiochip0" }`
- POST `/gpio/write`
  - body: `{ "pin": 12, "value": 1, "chip": "gpiochip0" }`
- POST `/gpio/read`
  - body: `{ "pin": 12, "chip": "gpiochip0" }`

### curl examples (pin 12)
```bash
# Set pin 12 as output
curl -s http://localhost:8000/gpio/set_mode -H 'Content-Type: application/json' \
  -d '{"pin":12,"mode":"out","chip":"gpiochip0"}' | jq

# Drive pin 12 high (LED on)
curl -s http://localhost:8000/gpio/write -H 'Content-Type: application/json' \
  -d '{"pin":12,"value":1,"chip":"gpiochip0"}' | jq

# Drive pin 12 low (LED off)
curl -s http://localhost:8000/gpio/write -H 'Content-Type: application/json' \
  -d '{"pin":12,"value":0,"chip":"gpiochip0"}' | jq

# Read pin 12
curl -s http://localhost:8000/gpio/read -H 'Content-Type: application/json' \
  -d '{"pin":12,"chip":"gpiochip0"}' | jq
```

## MCP (Model Context Protocol)
This server exposes MCP over Server-Sent Events (SSE) at `/mcp`. Tools provided:
- `gpio_write(pin, value, chip?)`
- `gpio_read(pin, chip?)`
- `gpio_set_mode(pin, mode, chip?)`

### Configure in Cursor
Add a custom MCP server pointing to this machine:
```json
{
  "name": "radxa-gpio",
  "server": {
    "type": "sse",
    "url": "http://<RADXA_IP>:8000/mcp"
  }
}
```
Or use the included `manifest.json` with the same transport and tools.

### JSON-RPC messages
- Client opens SSE stream at `/mcp`
- Client POSTs JSON-RPC to `/mcp` with methods: `initialize`, `tools/list`, `tools/call`
- Server responds via SSE events with matching `id`

Example `tools/call` body:
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/call",
  "params": { "name": "gpio_write", "arguments": { "pin": 12, "value": 1 } }
}
```

## Notes
- Uses libgpiod (not sysfs). Requires permission to access `/dev/gpiochip*`.
- On first read, server configures the line as input; writes configure as output.
- Lines are cached and reused safely; wrong-direction requests are re-requested.
