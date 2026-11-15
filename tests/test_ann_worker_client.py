import json
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.ann_worker_client import read_ann_manifest_metadata  # noqa: E402


class ANNWorkerClientTests(unittest.TestCase):
    def test_manifest_metadata_reads_symbols_and_checkpoint_from_source_shards(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cnn.sqlite3"
            with sqlite3.connect(source) as connection:
                connection.execute(
                    """CREATE TABLE feature_versions (
                    version_id TEXT PRIMARY KEY, source_hash TEXT, config_json TEXT,
                    symbols_json TEXT, model TEXT, dimension INTEGER, status TEXT,
                    sample_count INTEGER, created_at_utc TEXT, updated_at_utc TEXT)"""
                )
                connection.execute(
                    "INSERT INTO feature_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "cnn-v1",
                        "source-v1",
                        json.dumps({"checkpoint_sha256": "checkpoint-v1"}),
                        json.dumps(["000001", "000002"]),
                        "cnn",
                        128,
                        "ready",
                        25,
                        "2026-01-01",
                        "2026-01-01",
                    ),
                )
            ann_dir = root / "ann"
            ann_dir.mkdir()
            (ann_dir / "manifest.json").write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "kline-ann-manifest-v1",
                        "model": "cnn",
                        "model_version": "cnn-test",
                        "version_id": "ann-v1",
                        "source_hash": "source-v1",
                        "dimension": 128,
                        "checkpoint_sha256": None,
                        "shards": [{"source": source.name, "ann": ann_dir.name}],
                    }
                ),
                encoding="utf-8",
            )

            metadata = read_ann_manifest_metadata(manifest)

        self.assertEqual(metadata.model, "cnn")
        self.assertEqual(metadata.checkpoint_sha256, "checkpoint-v1")
        self.assertEqual(metadata.symbols, ("000001", "000002"))
        self.assertEqual(metadata.sample_count, 25)


if __name__ == "__main__":
    unittest.main()
