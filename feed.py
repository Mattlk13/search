from vespa.deployment import VespaDocker
from vespa.io import VespaResponse
from datasets import load_dataset
from tqdm import tqdm

from hybrid import package
from sentence_transformers import SentenceTransformer

ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="mps")


if __name__ == "__main__":
    vespa_docker = VespaDocker(container_memory=8 * (1024**3))
    app = vespa_docker.deploy(application_package=package)
    package.to_files(root="./vespa_app_hybrid")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        "CC-MAIN-2025-08",
        split="train",
        streaming=True,
    )

    vespa_feed = dataset.map(
        lambda x: {
            "id": x["id"],
            "fields": {
                "text": x["text"],
                "url": x["url"],
                "id": x["id"],
                "text_embedding": ST_MODEL.encode(
                    x["text"], convert_to_numpy=True
                ).tolist(),
            },
        }
    )

    pbar = tqdm(desc="Feeding documents", unit="docs")
    feed_count = {"success": 0, "error": 0}

    def callback(response: VespaResponse, id: str):
        if response.is_successful():
            feed_count["success"] += 1
        else:
            feed_count["error"] += 1
            pbar.write(f"Error when feeding document {id}: {response.get_json()}")
        pbar.update(1)

    app.feed_iterable(
        vespa_feed, schema="doc", callback=callback, max_workers=4, max_connections=4
    )
    pbar.close()
