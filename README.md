## Prerequisite

Use `pyproject.toml` to manage the `uv` environment.

### 1. Install uv
```
sudo apt install curl
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository

Clone this repository.  

```
git clone https://github.com/smtamh/ssds_HL.git
```

### 3. Create uv Project
```
cd ssds_HL
uv sync             # uv reads `pyproject.toml` and downloads dependencies in 'ssds_HL/.venv'

# 'uv run ...' use 'ssds_HL/.venv' automatically.
```

<br>

## Usage with Web ChatGPT

A ChatGPT subscription is required to use MCP in Web ChatGPT.  

### 1. Run MCP Server

Start your MCP server:
```
uv run server.py --transport streamable-http
```
Your MCP server will run on a local port (e.g., http://127.0.0.1:8000)  

### 2. Expose Server with ngrok
Install ```ngrok``` from the [official website](https://ngrok.com).  
Sign up and follow the installation guide.

Expose your local server to the internet:
```
ngrok http 8000 --host-header=rewrite
```
After running, you will see a forwarding URL like:
```
https://abcd-1234.ngrok-free.dev -> http://localhost:8000
```
The HTTPS URL (`https://abcd-1234.ngrok-free.dev`) will be used in the ChatGPT website.

### 3. Enable Developer Mode
Go to:
- ChatGPT website
- Account → Apps → Advanced Settings

Enable ```Developer mode```

### 4. Create MCP App
Go to:
- ChatGPT website
- Account → Apps → ```Create app``` (next to Advanced Settings)

<br>

Fill in:
- **Name**: ssds_HL (or any name)
- **Description**: (optional)
- **MCP Server URL**:
    ```
    https://abcd-1234.ngrok-free.dev/mcp
    ```
- **Authentication**: No Auth
- Select the check box  

Then click ```Create```

### 5. Verify Connection
Go to: 
- ChatGPT website
- Account → Apps

Confirm your app is listed.  
Click ```Refresh``` to update the MCP server status

<br>

## Usage with local LLM/VLM
Run LLM/VLM on Ubuntu 24.04 with python 3.12.
```
UV_PROJECT_ENVIRONMENT=.venv/312 uv "$@" run inference_text.py --transport streamable-http

# or

UV_PROJECT_ENVIRONMENT=.venv/312 uv "$@" run inference_stt.py --input-source text --transport streamable-http
```

Run the server on Ubuntu 22.04 with python 3.10.  
You may use a Docker environment.
```
UV_PROJECT_ENVIRONMENT=.venv/310 uv "$@" run server.py --transport streamable-http
```

To use STT, run LLM/VLM with the following commands.  
Run `server.py` separately using the command above.
```
# 1st terminal
UV_PROJECT_ENVIRONMENT=.venv/312 uv "$@" run recognize_speech.py

# 2nd terminal
UV_PROJECT_ENVIRONMENT=.venv/312 uv "$@" run inference_stt.py --input-source speech --transport streamable-http
```

Check MCP endpoint (e.g., http://127.0.0.1:8000/mcp) on the server side, and then use that endpoint in inference_*.py

`recognize_speech.py` starts the localhost speech text server by default. Use `--no-server` to disable it.
