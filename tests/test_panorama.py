import io
import json
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import main
from panorama.search import (
    make_search_url,
    extract_panoramas,
    Panorama,
)
from panorama.download import (
    get_width_and_height_from_zoom,
    make_download_url,
)
from unities import eulerAnglesToRotationMatrix


# ─── search.py ────────────────────────────────────────────────────────────────

class TestMakeSearchUrl:
    def test_url_contains_lat_lon(self):
        url = make_search_url(lat=25.0, lon=121.5)
        assert "25.0" in url
        assert "121.5" in url
        assert "callbackfunc" in url

    def test_url_format_valid(self):
        url = make_search_url(lat=40.7128, lon=-74.0060)
        assert url.startswith("https://maps.googleapis.com/maps/api/js/")


# ─── download.py ─────────────────────────────────────────────────────────────

class TestGetWidthAndHeightFromZoom:
    @pytest.mark.parametrize("zoom,expected_w,expected_h", [
        (1, 2, 1),
        (2, 4, 2),
        (3, 8, 4),
        (4, 16, 8),
        (5, 32, 16),
    ])
    def test_dimensions(self, zoom, expected_w, expected_h):
        w, h = get_width_and_height_from_zoom(zoom)
        assert w == expected_w
        assert h == expected_h


class TestMakeDownloadUrl:
    def test_url_contains_tile_params(self):
        url = make_download_url("test_pano_id", zoom=3, x=2, y=1)
        assert "test_pano_id" in url
        assert "zoom=3" in url
        assert "x=2" in url
        assert "y=1" in url

    def test_url_format(self):
        url = make_download_url("abc123", zoom=2, x=0, y=0)
        assert url.startswith("https://cbk0.google.com/cbk")


# ─── unities.py ──────────────────────────────────────────────────────────────

class TestEulerAnglesToRotationMatrix:
    def test_returns_3x3_matrix(self):
        R = eulerAnglesToRotationMatrix([0.0, 0.0, 0.0])
        assert R.shape == (3, 3)

    def test_identity_for_zero_angles(self):
        R = eulerAnglesToRotationMatrix([0.0, 0.0, 0.0])
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotation_matrix_is_orthogonal(self):
        R = eulerAnglesToRotationMatrix([0.5, 0.3, 0.1])
        # R^T R should be identity
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=5)

    def test_determinant_is_one(self):
        R = eulerAnglesToRotationMatrix([1.0, 0.5, 0.2])
        assert abs(np.linalg.det(R) - 1.0) < 1e-5


# ─── Panorama model ──────────────────────────────────────────────────────────

class TestPanoramaModel:
    def test_valid_panorama_creation(self):
        p = Panorama(
            pano_id="test123",
            lat=25.0,
            lon=121.5,
            heading=90.0,
            pitch=0.0,
            roll=0.0,
            date="2023-05",
            scale=[[512, 512]],
            tile=[256, 256],
        )
        assert p.pano_id == "test123"
        assert p.lat == 25.0
        assert p.date == "2023-05"

    def test_optional_fields(self):
        p = Panorama(
            pano_id="test456",
            lat=40.7,
            lon=-74.0,
            heading=0.0,
        )
        assert p.pitch is None
        assert p.roll is None
        assert p.date is None

    def test_save_file_format(self):
        p = Panorama(
            pano_id="test789",
            lat=25.0,
            lon=121.5,
            heading=180.0,
            pitch=10.0,
            roll=5.0,
            date="2022-03",
        )
        buf = io.StringIO()
        p.saveFile(buf)
        content = buf.getvalue()
        assert "test789" in content
        assert "25.0" in content
        assert "121.5" in content
        assert "2022-03" in content

    def test_save_file_csv_writer(self):
        """saveFile must also work with csv.writer (main.py use case)."""
        import csv
        p = Panorama(
            pano_id="csv_test",
            lat=35.0,
            lon=139.0,
            heading=90.0,
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        p.saveFile(writer)
        content = buf.getvalue()
        assert "csv_test" in content
        assert "35.0" in content
        assert "139.0" in content


class DummyImage:
    def save(self, path, fmt):
        Path(path).write_bytes(b"fake image bytes")


class TestFetchPanoramas:
    def test_upserts_records_and_downloads_only_missing_images(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        pano_dir.mkdir(parents=True)
        info_file = pano_dir / "info.csv"
        info_file.write_text(
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            "existing,1.0,2.0,90.0,0.0,0.0,2020-01\n",
            encoding="utf-8",
        )
        (pano_dir / "existing.png").write_bytes(b"already here")

        monkeypatch.setattr(main, "panoPath", pano_dir)
        monkeypatch.setattr(main, "infoPath", info_file)
        monkeypatch.setattr(main, "get_session", lambda: object())
        monkeypatch.setattr(main.random, "uniform", lambda _a, _b: 0.0)
        monkeypatch.setattr(main.time, "sleep", lambda _delay: None)

        panos = [
            Panorama(
                pano_id="existing",
                lat=1.1,
                lon=2.2,
                heading=180.0,
                pitch=1.0,
                roll=2.0,
                date="2022-05",
            ),
            Panorama(
                pano_id="new",
                lat=3.0,
                lon=4.0,
                heading=45.0,
                pitch=0.5,
                roll=0.2,
                date="2023-07",
            ),
        ]
        monkeypatch.setattr(main, "search_panoramas", lambda lat, lon: panos)

        downloaded = []

        def fake_get_panorama(*, pano, zoom, session):
            downloaded.append((pano.pano_id, zoom, session))
            return DummyImage()

        monkeypatch.setattr(main, "get_panorama", fake_get_panorama)

        main.fetch_panoramas((25.0, 121.5), isCurrent=False)

        rows = list(main.csv.DictReader(info_file.open(encoding="utf-8")))
        assert rows == [
            {
                "pano_id": "existing",
                "lat": "1.1",
                "lon": "2.2",
                "heading": "180.0",
                "pitch": "1.0",
                "roll": "2.0",
                "date": "2022-05",
            },
            {
                "pano_id": "new",
                "lat": "3.0",
                "lon": "4.0",
                "heading": "45.0",
                "pitch": "0.5",
                "roll": "0.2",
                "date": "2023-07",
            },
        ]
        assert len(downloaded) == 1
        assert downloaded[0][0] == "new"
        assert downloaded[0][1] == 3
        assert (pano_dir / "new.png").read_bytes() == b"fake image bytes"
        assert (pano_dir / "existing.png").read_bytes() == b"already here"

    def test_keeps_existing_file_when_write_fails_before_replace(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        pano_dir.mkdir(parents=True)
        info_file = pano_dir / "info.csv"
        original = (
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            "stable,1.0,2.0,3.0,4.0,5.0,2020-01\n"
        )
        info_file.write_text(original, encoding="utf-8")

        records = {
            "stable": {
                "pano_id": "stable",
                "lat": 9.0,
                "lon": 8.0,
                "heading": 7.0,
                "pitch": 6.0,
                "roll": 5.0,
                "date": "2024-02",
            }
        }

        monkeypatch.setattr(main, "panoPath", pano_dir)
        monkeypatch.setattr(main, "infoPath", info_file)

        def fail_replace(self, target):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(Path, "replace", fail_replace)

        with pytest.raises(OSError, match="simulated replace failure"):
            main.write_info_records(records)

        assert info_file.read_text(encoding="utf-8") == original
        assert not any(pano_dir.glob("info.*.tmp"))

    def test_current_mode_only_persists_records_without_date(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        pano_dir.mkdir(parents=True)
        info_file = pano_dir / "info.csv"
        info_file.write_text("pano_id,lat,lon,heading,pitch,roll,date\n", encoding="utf-8")

        monkeypatch.setattr(main, "panoPath", pano_dir)
        monkeypatch.setattr(main, "infoPath", info_file)
        monkeypatch.setattr(main, "get_session", lambda: object())
        monkeypatch.setattr(main.random, "uniform", lambda _a, _b: 0.0)
        monkeypatch.setattr(main.time, "sleep", lambda _delay: None)
        monkeypatch.setattr(
            main,
            "search_panoramas",
            lambda lat, lon: [
                Panorama(pano_id="current", lat=1.0, lon=2.0, heading=3.0, date=None),
                Panorama(pano_id="history", lat=4.0, lon=5.0, heading=6.0, date="2021-01"),
            ],
        )
        monkeypatch.setattr(main, "get_panorama", lambda **_kwargs: DummyImage())

        main.fetch_panoramas((25.0, 121.5), isCurrent=True)

        rows = list(main.csv.DictReader(info_file.open(encoding="utf-8")))
        assert [row["pano_id"] for row in rows] == ["current"]
        assert rows[0]["date"] == ""
