"""Mark an externally stopped follow-up queue as paused without dropping results."""
import json
from datetime import datetime
from pathlib import Path

from run_followup import render, write_json

OUT = Path(__file__).resolve().parent
path = OUT / "status.json"
status = json.loads(path.read_text(encoding="utf-8"))
status["status"] = "paused_by_user"
status["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
status["pause_note"] = (
    "Stopped at user request after PEMS03 completed. Before termination took effect, "
    "PEMS04 sparse repeats 1-2 and full repeat 1 also completed and are retained. "
    "PEMS04 full repeat 2 was interrupted and is not counted."
)
write_json(path, status)
temp = OUT / "SRM_FOLLOWUP_VALIDATION_20261015.md.tmp"
temp.write_text(render(status), encoding="utf-8")
temp.replace(OUT / "SRM_FOLLOWUP_VALIDATION_20261015.md")
print(status["status"], len(status["jobs"]))
