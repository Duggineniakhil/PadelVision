"""Write the API's OpenAPI schema to web/openapi.json (the web app's types are generated from it).

python scripts/export_openapi.py && (cd web && npm run gen:api)
"""

import json
from pathlib import Path

from padelvision_api.main import app

out = Path(__file__).resolve().parents[1] / "web" / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=1) + "\n")
print(f"wrote {out}")
