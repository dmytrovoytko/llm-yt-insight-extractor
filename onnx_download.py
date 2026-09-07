from core.hf_download import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKING_MODEL,
    download,
)

def model_download(model_name):
    path = f"models/{model_name}"
    try:
        from pathlib import Path
        dir_path = Path(path)
        # Check if it exists and is a directory
        if dir_path.is_dir():
            print(f"\nModel directory '{path}' already exists, downloading skipped.")
        else:
            download(model_name)
            print(f"\nModel '{model_name}' downloaded.")
    except Exception as e:
        print(f"\nModel '{model_name}' downloading failed:\n {e}")


if __name__ == "__main__":
    model_download(DEFAULT_EMBEDDING_MODEL)
    model_download(DEFAULT_RERANKING_MODEL)
    # exit()


