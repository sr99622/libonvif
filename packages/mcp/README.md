<h2>MCP Runtime Example using libonvif</h2>

This example uses the UV runtime

Please note that you have to edit the json file manually. The file can be found using Claude Desktop. Go to File->Settings->Developer and click the Edit Config button. This will bring up a file browser highlighting the claude_dekstop_config.json file. Adjust these settings to your current situation and paste them into the json at the top.

you can get the git server from 

```
git clone https://github.com/modelcontextprotocol/servers.git
```

and the libonvif MCP at

```
git clone https://github.com/sr996222/local.mcpb.stephen-rhodes.camera
```

```
  "mcpServers": {
    "git": {
      "command": "uv",
      "args": [
        "--directory", 
        "C:\\Users\\sr996\\Projects\\servers\\src\\git",
        "run",
        "src\\mcp_server_git"
      ]
    },    
    "camera": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\sr996\\Projects\\libonvif\\packages\\mcp\\src",
        "run",
        "camera.py"
      ],
      "env": {
        "CAMERA_USERNAME": "admin",
        "CAMERA_PASSWORD": "admin123",
        "STREAM_SERVER_IP": "10.1.1.13"
      }
    }
  },

```
