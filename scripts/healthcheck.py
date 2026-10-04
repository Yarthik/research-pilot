"""Docker HEALTHCHECK: the API exposes /health, Streamlit exposes /_stcore/health."""

import os
import sys
import urllib.request

port = os.environ.get("PORT", "7860")
path = "health" if os.environ.get("SERVICE") == "api" else "_stcore/health"
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
