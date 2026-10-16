"""Start MLflow UI with a JavaScript MIME type override for Windows Python."""

import mimetypes
import os
from pathlib import Path

# On some Windows Python installs, the system MIME registry reports .js as
# text/plain, which causes browsers to refuse to run the MLflow frontend.
mimetypes.init()
mimetypes.add_type("text/javascript", ".js")

# `mlflow ui` starts Uvicorn in a child Python process. Ensure that process
# also applies the MIME override before Flask serves the static bundle.
mime_fix_dir = Path(__file__).resolve().parent / "mlflow_mime_fix"
python_path = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    f"{mime_fix_dir}{os.pathsep}{python_path}" if python_path else str(mime_fix_dir)
)

from mlflow.cli import cli


project_dir = Path(__file__).resolve().parent
tracking_db = project_dir / "mlflow.db"
tracking_uri = f"sqlite:///{tracking_db.as_posix()}"

cli.main(
    args=[
        "ui",
        "--backend-store-uri",
        tracking_uri,
        "--host",
        "127.0.0.1",
        "--port",
        "5001",
    ],
    prog_name="mlflow",
)
