"""Execute the report notebook with this interpreter and save every output."""

import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = next((ROOT / "notebooks").glob("*.ipynb"))


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    manager = KernelManager(kernel_name="python3")
    spec = manager.kernel_spec
    if spec is None:
        raise RuntimeError("No Python kernel specification is available; install ipykernel.")
    spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        client = NotebookClient(
            notebook, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}}
        )
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {count} code cells: {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
