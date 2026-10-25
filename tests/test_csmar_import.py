import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.csmar_import import import_csmar_csvs  # noqa: E402


class CsmarImportTests(unittest.TestCase):
    def test_imports_csmar_fields_and_writes_quality_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            csv_path = source / "TRD_Dalyr.csv"
            csv_path.write_text(
                "Stkcd,Trddt,Opnprc,Hiprc,Loprc,Clsprc,Dnshrtrd,Dnvaltrd,Dretnd,PreClosePrice\n"
                "000002,2024-01-02,10,11,9,10.5,100,1050,0.05,10\n"
                "000002,2024-01-03,10.5,12,10,11.5,110,1265,0.0952,10.5\n"
                "000004,2024-01-02,8,8.5,7.5,8.2,90,738,0.025,8\n",
                encoding="utf-8",
            )
            database = Path(directory) / "market.sqlite3"
            report_path = Path(directory) / "quality.json"

            report = import_csmar_csvs([source], database, quality_report_path=report_path)

            self.assertEqual(report["row_count"], 3)
            self.assertEqual(report["symbol_count"], 2)
            self.assertEqual(report["invalid_rows"], 0)
            self.assertEqual(report["data_start_date"], "2024-01-02")
            self.assertEqual(report["data_end_date"], "2024-01-03")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["provider"], "csmar")


if __name__ == "__main__":
    unittest.main()
