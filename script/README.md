# Scripts

This directory groups helper scripts by purpose:

- `baseline/`: standalone Python baseline inference and its minimal runtime copy.
- `ai_servers/`: start/stop helpers for one or more AI engine servers.
- `build/`: rebuild/deployment/parser reset helpers.
- `setup/`: optional setup helpers for external components.
- `maintenance/`: one-off maintenance utilities and supporting text files.
- `experiment/`: experiment code and historical proof-of-concept scripts.

Common commands:

```bash
python script/baseline/baseline_inference.py
./script/ai_servers/start_ai_servers.sh 3
./script/ai_servers/stop_ai_servers.sh
```
