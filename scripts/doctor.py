#!/usr/bin/env python3
"""Check active local configuration without marking a missing device as success."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
from tools.detect_person import readiness as person_ready
from tools.detect_fire_smoke import readiness as fire_ready

result = {'person': person_ready(agent.CONFIG['person_detection']),
          'fire_smoke': fire_ready(agent.CONFIG['fire_smoke']),
          'llm_file': Path(agent.CONFIG['model_path']).is_file(),
          'llama_cli': Path(agent.CONFIG['llama_cli']).is_file(),
          'npu_device_present': Path('/dev/aipu').exists(),
          'active_backend': 'CPU (DAMO-YOLO + llama.cpp)'}
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result['person']['ready'] and result['fire_smoke']['ready'] and result['llm_file'] and result['llama_cli'] else 1)
