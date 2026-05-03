import io
import json
import pytest
import numpy as np
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
import main
import panorama.download as download_module
from panorama.search import (
    make_search_url,
    extract_panoramas,
    search_request,
    Panorama,
    PanoramaSearchError,
)
from panorama.download import (
    get_width_and_height_from_zoom,
    get_panorama,
    download_by_pano_id,
    PanoDownloadError,
)
from panorama.download import Tile
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


class TestSearchRequest:
    def test_uses_timeout_and_checks_status(self, monkeypatch):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                calls.append("raise_for_status")

        def fake_get(url, timeout):
            calls.append((url, timeout))
            return FakeResponse()

        monkeypatch.setattr("panorama.search.requests.get", fake_get)

        resp = search_request(25.0, 121.5)

        assert isinstance(resp, FakeResponse)
        assert calls[0][1] == 15
        assert calls[1] == "raise_for_status"

    def test_wraps_request_errors(self, monkeypatch):
        def fake_get(url, timeout):
            raise __import__("requests").RequestException("network down")

        monkeypatch.setattr("panorama.search.requests.get", fake_get)

        with pytest.raises(PanoramaSearchError, match="Search request failed"):
            search_request(25.0, 121.5)


class TestExtractPanoramas:
    def test_invalid_callback_raises_search_error(self):
        with pytest.raises(PanoramaSearchError, match="callbackfunc"):
            extract_panoramas("not a callback response")


# ─── download.py ─────────────────────────────────────────────────────────────

class TestGetWidthAndHeightFromZoom:
    @pytest.mark.parametrize("zoom,expected_w,expected_h", [
        (1, 4, 2),
        (2, 8, 4),
        (3, 16, 8),
        (4, 32, 16),
        (5, 64, 32),
    ])
    def test_dimensions(self, zoom, expected_w, expected_h):
        w, h = get_width_and_height_from_zoom(zoom)
        assert w == expected_w
        assert h == expected_h


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
        p.save_to_file(buf)
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
        p.save_to_file(writer)
        content = buf.getvalue()
        assert "csv_test" in content
        assert "35.0" in content
        assert "139.0" in content


class DummyImage:
    def save(self, path, fmt):
        Path(path).write_bytes(b"fake image bytes")


class TestFetchPanoramas:
    def test_default_info_path_is_not_inside_images_dir(self):
        assert main.infoPath == Path(main.__file__).parent / "images" / "info.csv"
        assert main.infoPath.parent != main.panoPath

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
        assert downloaded[0][1] == main._cfg["zoom"]
        assert (pano_dir / "new.png").read_bytes() == b"fake image bytes"
        assert (pano_dir / "existing.png").read_bytes() == b"already here"

    def test_keeps_existing_file_when_write_fails_before_replace(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        metadata_dir = tmp_path / "images"
        pano_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        info_file = metadata_dir / "info.csv"
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
        assert not any(metadata_dir.glob("info.*.tmp"))
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

    def test_removes_stale_record_when_missing_image_download_fails(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        pano_dir.mkdir(parents=True)
        info_file = pano_dir / "info.csv"
        info_file.write_text(
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            "stale,1.0,2.0,90.0,0.0,0.0,2020-01\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(main, "panoPath", pano_dir)
        monkeypatch.setattr(main, "infoPath", info_file)
        monkeypatch.setattr(main, "get_session", lambda: object())
        monkeypatch.setattr(main.random, "uniform", lambda _a, _b: 0.0)
        monkeypatch.setattr(main.time, "sleep", lambda _delay: None)
        monkeypatch.setattr(
            main,
            "search_panoramas",
            lambda lat, lon: [
                Panorama(pano_id="stale", lat=1.0, lon=2.0, heading=90.0, date="2020-01"),
            ],
        )
        monkeypatch.setattr(
            main,
            "get_panorama",
            lambda **_kwargs: (_ for _ in ()).throw(PanoDownloadError("simulated failure")),
        )

        main.fetch_panoramas((25.0, 121.5), isCurrent=False)

        rows = list(main.csv.DictReader(info_file.open(encoding="utf-8")))
        assert rows == []

    def test_processing_failure_keeps_final_image_and_csv_clean(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        pano_dir.mkdir(parents=True)
        info_file = pano_dir / "info.csv"
        info_file.write_text("pano_id,lat,lon,heading,pitch,roll,date\n", encoding="utf-8")

        monkeypatch.setattr(main, "panoPath", pano_dir)
        monkeypatch.setattr(main, "infoPath", info_file)
        monkeypatch.setattr(main, "get_session", lambda: object())
        monkeypatch.setattr(main.random, "uniform", lambda _a, _b: 0.0)
        monkeypatch.setattr(main.time, "sleep", lambda _delay: None)
        monkeypatch.setitem(main._cfg, "process_on_download", True)
        monkeypatch.setattr(
            main,
            "search_panoramas",
            lambda lat, lon: [
                Panorama(pano_id="postprocess_fail", lat=1.0, lon=2.0, heading=90.0, date="2020-01"),
            ],
        )
        monkeypatch.setattr(main, "get_panorama", lambda **_kwargs: DummyImage())
        monkeypatch.setattr(
            main,
            "detect_and_crop_black_edge",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad image")),
        )

        main.fetch_panoramas((25.0, 121.5), isCurrent=False)

        assert not (pano_dir / "postprocess_fail.png").exists()
        assert not any(pano_dir.glob("*.tmp.png"))
        rows = list(main.csv.DictReader(info_file.open(encoding="utf-8")))
        assert rows == []


# ─── download_by_pano_id ─────────────────────────────────────────────────────

class FakeDownloader:
    def __init__(self, tiles):
        self.tiles = tiles

    def iter_tiles(self, pano_id, zoom, session):
        yield from self.tiles

class TestDownloadByPanoId:
    def test_saves_to_default_images_dir(self, tmp_path, monkeypatch):
        """download_by_pano_id saves PNG to config images_dir when output_dir is None."""
        images_dir = tmp_path / "images" / "pano"
        img = Image.new("RGB", (512, 512), color=(255, 0, 0))

        monkeypatch.setitem(download_module._cfg, "images_dir", str(images_dir))
        monkeypatch.setattr("panorama.download.get_downloader", lambda: FakeDownloader([Tile(x=0, y=0, image=img)]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        result = download_by_pano_id("test_pano_001", zoom=1, output_dir=None)

        assert result is not None
        assert isinstance(result.size, tuple)
        assert (images_dir / "test_pano_001.png").exists()

    def test_saves_to_custom_output_dir(self, tmp_path, monkeypatch):
        """download_by_pano_id saves PNG to the specified output_dir."""
        out_dir = tmp_path / "custom_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (512, 512), color=(0, 255, 0))

        monkeypatch.setattr("panorama.download.get_downloader", lambda: FakeDownloader([Tile(x=0, y=0, image=img)]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        result = download_by_pano_id("custom_pano_999", zoom=1, output_dir=out_dir)

        assert result is not None
        expected_path = out_dir / "custom_pano_999.png"
        assert expected_path.exists(), f"Expected {expected_path} to exist"
        assert expected_path.read_bytes() != b""

    def test_zoom_1_produces_correct_tile_count(self, tmp_path, monkeypatch):
        """zoom=1 → 4 tiles wide × 2 tiles tall = 8 tiles total."""
        tiles = []

        class CountingDownloader:
            def iter_tiles(self, pano_id, zoom, session):
                nonlocal tiles
                w, h = get_width_and_height_from_zoom(zoom)
                for x in range(w):
                    for y in range(h):
                        tiles.append((x, y))
                        img = Image.new("RGB", (512, 512), color=(0, 0, 255))
                        yield Tile(x=x, y=y, image=img)

        monkeypatch.setattr("panorama.download.get_downloader", lambda: CountingDownloader())
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        download_by_pano_id("zoom_test", zoom=1, output_dir=tmp_path / "zoom1")

        assert len(tiles) == 8, f"Expected 8 tiles for zoom=1, got {len(tiles)}"
        assert tiles == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

    def test_zoom_3_produces_correct_tile_count(self, tmp_path, monkeypatch):
        """zoom=3 → 16 tiles wide × 8 tiles tall = 128 tiles total."""
        tiles_received = []

        class CountingDownloader:
            def iter_tiles(self, pano_id, zoom, session):
                nonlocal tiles_received
                w, h = get_width_and_height_from_zoom(zoom)
                for x in range(w):
                    for y in range(h):
                        tiles_received.append((x, y))
                        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
                        yield Tile(x=x, y=y, image=img)

        monkeypatch.setattr("panorama.download.get_downloader", lambda: CountingDownloader())
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        download_by_pano_id("zoom3_test", zoom=3, output_dir=tmp_path / "zoom3")

        assert len(tiles_received) == 128, f"Expected 128 tiles for zoom=3, got {len(tiles_received)}"

    def test_raises_pano_download_error_on_tile_failure(self, tmp_path, monkeypatch):
        """Downloader errors must propagate as PanoDownloadError."""
        class FailingDownloader:
            def iter_tiles(self, pano_id, zoom, session):
                raise PanoDownloadError("simulated tile failure")

        monkeypatch.setattr("panorama.download.get_downloader", lambda: FailingDownloader())
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        with pytest.raises(PanoDownloadError, match="simulated tile failure"):
            download_by_pano_id("fail_test", zoom=1, output_dir=tmp_path / "fail")

    def test_returns_pil_image(self, tmp_path, monkeypatch):
        """download_by_pano_id must return a PIL Image."""
        img = Image.new("RGB", (512, 512), color=(255, 128, 0))

        monkeypatch.setattr("panorama.download.get_downloader", lambda: FakeDownloader([Tile(x=0, y=0, image=img)]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        result = download_by_pano_id("pil_test", zoom=1, output_dir=tmp_path / "pil")

        assert isinstance(result, Image.Image)


class TestGetPanorama:
    def test_raises_when_no_tiles_downloaded(self, monkeypatch):
        pano = Panorama(
            pano_id="empty",
            lat=25.0,
            lon=121.5,
            heading=0.0,
            scale=[[[512, 1024]], [[512, 1024]]],
            tile=[512, 512],
        )

        monkeypatch.setattr("panorama.download.get_downloader", lambda api_key=None: FakeDownloader([]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())

        with pytest.raises(PanoDownloadError, match="No tiles downloaded"):
            get_panorama(pano=pano, zoom=1)
