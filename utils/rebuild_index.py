import subprocess


def rebuild_index():

    print("Rebuilding FAISS index...")

    subprocess.run(
        ["python", "rag_ingest.py"]
    )

    print("✅ Index rebuilt.")