import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_classify_rule_on_sample_data():
    result = subprocess.run(
        [sys.executable, "-m", "wmft_motion.cli", "classify-rule", str(ROOT / "sample_data.txt")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "17,WMFT 17: Lift basket"
