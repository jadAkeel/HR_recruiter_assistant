from pathlib import Path
import re

import pdfplumber


CV_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "cvs_to_upload"
SYNTHETIC_LABEL = "synthetic test resume - not a real person"
RESERVED_PHONE = re.compile(r"\+1-202-555-01\d{2}")


def test_tracked_cv_fixtures_are_readable_and_unambiguously_synthetic():
    fixture_paths = sorted(CV_FIXTURE_DIR.glob("*.pdf"))

    assert len(fixture_paths) == 100
    assert not (CV_FIXTURE_DIR / "jad_akil_cv.pdf").exists()
    for path in fixture_paths:
        with pdfplumber.open(path) as document:
            assert document.pages, f"{path.name} has no pages"
            text = "\n".join(page.extract_text() or "" for page in document.pages)

        normalized = text.lower()
        assert SYNTHETIC_LABEL in normalized, path.name
        assert "@example.com" in normalized, path.name
        assert "gmail.com" not in normalized, path.name
        assert "+961" not in normalized, path.name
        assert RESERVED_PHONE.search(text), path.name
