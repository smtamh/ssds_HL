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
uv run server.py
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

### 1. Download LLM and STT model from Huggingface

`download_model_from_hub.py` downloads the model specified by `model_id` into
`models/`, the default directory configured in `config.py`. Change `model_id`
to the Hugging Face model you want, save the file, and run:

```
uv run python download_model_from_hub.py
```

For example, set `model_id` to `Qwen/Qwen3-4B-AWQ` for the LLM or
`Systran/faster-whisper-medium` for STT. After downloading, make sure the
corresponding `LLM_PATH` or `STT_PATH` in `config.py` matches the saved folder
name.

### 2. Using one computer (Ubuntu 22.04 Docker environment)

Use this setup when the robot/MCP server runs in the Ubuntu 22.04 Docker
environment and inference runs on the Ubuntu 24.04 host. Docker uses host
networking, so the host can access the MCP endpoint at
`http://127.0.0.1:8000/mcp`.

Start the Docker environment and open a shell in the container:

```
docker compose up -d --build
docker exec -it ssds bash
```

In the container, initialize the Python 3.10 environment once and start the
MCP server:

```
source /root/.bashrc
cd /root/ssds_HL
uv-docker sync
uv-docker run server.py
```

On the Ubuntu 24.04 host, initialize the Python 3.12 environment once, then
run text inference in a separate terminal:

```
UV_PROJECT_ENVIRONMENT=.venv/312 uv sync --python 3.12
UV_PROJECT_ENVIRONMENT=.venv/312 uv run inference_text.py
```

For speech input, start the following commands in separate terminals:

```
# Terminal 1
UV_PROJECT_ENVIRONMENT=.venv/312 uv run recognize_speech.py

# Terminal 2
UV_PROJECT_ENVIRONMENT=.venv/312 uv run inference_stt.py --input-source speech
```

### 3. Using two computers (Ubuntu 22.04 server and Ubuntu 24.04 inference)

Use this setup when the robot and MCP server are on an Ubuntu 22.04 computer,
while the LLM/VLM inference runs on a separate Ubuntu 24.04 computer. Docker
is not required on either computer.

#### Ubuntu 22.04 computer: start the MCP server

Set up the robot/ROS workspace on this computer and clone this repository as
described in the prerequisites. Source the ROS workspace before starting the
MCP server. Replace `{ROS_WORKSPACE}` with the path to the workspace you want
to use:

```
source {ROS_WORKSPACE}/install/setup.bash
cd ssds_HL
uv run server.py
```

Allow TCP port `8000` through the server computer's firewall if necessary. The
Ubuntu 24.04 computer must be able to reach
`http://<SERVER_IP>:8000/mcp` on the same network.

#### Ubuntu 24.04 computer: run inference

Clone this repository, run `uv sync`, and download the models as described
above:

```
cd ssds_HL
uv sync
```

Run text inference by replacing `<SERVER_IP>` with the Ubuntu 22.04 computer's
IP address:

```
uv run inference_text.py --mcp-server http://<SERVER_IP>:8000/mcp
```

For speech input, start speech recognition and inference in separate terminals:

```
# Terminal 1
uv run recognize_speech.py

# Terminal 2
uv run inference_stt.py \
  --input-source speech \
  --mcp-server http://<SERVER_IP>:8000/mcp
```

You can check the MCP endpoint from the Ubuntu 24.04 computer with:

```
curl http://<SERVER_IP>:8000/mcp
```

`recognize_speech.py` starts the localhost speech text server by default. Use `--no-server` to disable it.
