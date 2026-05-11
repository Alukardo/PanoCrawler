import io
import json
import pytest
import requests
import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
import integration.build_quality_dataset as quality_dataset
import main
import build_training_pairs as training_pairs_module
import panorama.config as config_module
import panorama.download as download_module
import panorama.quota as quota_module
import panorama.search as search_module
from panorama.search import (
    make_search_url,
    extract_panoramas,
    search_request,
    Panorama,
    PanoramaSearchError,
)
from panorama.download import (
    choose_best_zoom_for_output,
    get_tile_grid_for_canvas,
    get_width_and_height_from_zoom,
    get_panorama,
    get_panorama_stages,
    download_by_pano_id,
    OUTPUT_SIZE,
    PanoDownloadError,
    PanoTileOutOfRangeError,
    MapsTileAPIDownloader,
)
from panorama.download import Tile
from panorama.process_images import crop_black_edge_from_image, detect_and_crop_black_edge
from panorama.quality import (
    apply_heading_adjustment,
    bottom_black_edge_ratio,
    build_quality_metrics,
    heading_shift_pixels,
    sharpness_score,
)
from panorama.quota import GoogleAPIQuotaExceededError, GoogleAPIUsageError
from panorama.geometry import eulerAnglesToRotationMatrix


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

    def test_url_uses_configured_search_base(self, monkeypatch):
        monkeypatch.setattr(search_module, "SEARCH_URL_BASE", "https://example.test/search")
        url = make_search_url(lat=25.0, lon=121.5)
        assert url.startswith("https://example.test/search")


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
        monkeypatch.setattr(search_module, "reserve_request", lambda category: calls.append(category))

        resp = search_request(25.0, 121.5)

        assert isinstance(resp, FakeResponse)
        assert calls[0] == "search_requests"
        assert calls[1][1] == 15
        assert calls[2] == "raise_for_status"

    def test_wraps_request_errors(self, monkeypatch):
        def fake_get(url, timeout):
            raise __import__("requests").RequestException("network down")

        monkeypatch.setattr("panorama.search.requests.get", fake_get)
        monkeypatch.setattr(search_module, "reserve_request", lambda category: None)
        monkeypatch.setattr(search_module, "record_failed_request", lambda category: None)

        with pytest.raises(PanoramaSearchError, match="Search request failed"):
            search_request(25.0, 121.5)

    def test_stops_before_request_when_quota_soft_limit_is_reached(self, monkeypatch):
        def fake_reserve_request(category):
            raise GoogleAPIQuotaExceededError("limit reached")

        monkeypatch.setattr(search_module, "reserve_request", fake_reserve_request)
        monkeypatch.setattr("panorama.search.requests.get", lambda *_args, **_kwargs: pytest.fail("request should not run"))

        with pytest.raises(PanoramaSearchError, match="limit reached"):
            search_request(25.0, 121.5)


class TestGoogleAPIQuota:
    def test_reserve_request_tracks_daily_total(self, tmp_path, monkeypatch):
        usage_path = tmp_path / "usage.json"
        monkeypatch.setattr(quota_module, "tracking_enabled", lambda: True)

        usage = quota_module.reserve_request("tile_requests", usage_path=usage_path, day="2026-05-09", soft_limit=10)

        assert usage["tile_requests"] == 1
        assert usage["estimated_total"] == 1
        data = json.loads(usage_path.read_text(encoding="utf-8"))
        assert data["2026-05-09"]["tile_requests"] == 1

    def test_reserve_request_raises_before_exceeding_soft_limit(self, tmp_path, monkeypatch):
        usage_path = tmp_path / "usage.json"
        usage_path.write_text(json.dumps({"2026-05-09": {"tile_requests": 10, "estimated_total": 10}}), encoding="utf-8")
        monkeypatch.setattr(quota_module, "tracking_enabled", lambda: True)

        with pytest.raises(GoogleAPIQuotaExceededError, match="daily soft limit"):
            quota_module.reserve_request("tile_requests", usage_path=usage_path, day="2026-05-09", soft_limit=10)

    def test_corrupted_usage_file_fails_closed(self, tmp_path):
        usage_path = tmp_path / "usage.json"
        usage_path.write_text("{broken", encoding="utf-8")

        with pytest.raises(GoogleAPIUsageError, match="corrupted"):
            quota_module.load_usage(usage_path)

    def test_reserve_request_uses_lock_file(self, tmp_path, monkeypatch):
        usage_path = tmp_path / "usage.json"
        monkeypatch.setattr(quota_module, "tracking_enabled", lambda: True)

        quota_module.reserve_request("search_requests", usage_path=usage_path, day="2026-05-09", soft_limit=10)

        assert usage_path.exists()
        assert usage_path.with_suffix(".json.lock").exists()


class TestConfigPaths:
    def test_resolve_images_path_uses_configured_images_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setitem(config_module.cfg, "images_root", "download_cache")

        assert config_module.get_images_root() == tmp_path / "download_cache"
        assert config_module.resolve_images_path("pano") == tmp_path / "download_cache" / "pano"
        assert config_module.resolve_images_path("info.csv") == tmp_path / "download_cache" / "info.csv"

    def test_resolve_images_path_preserves_structure_for_legacy_paths(self, tmp_path, monkeypatch):
        external_root = tmp_path / "external" / "images"
        monkeypatch.setitem(config_module.cfg, "images_root", str(external_root))

        assert config_module.resolve_images_path("images/pano") == external_root / "pano"
        assert config_module.resolve_images_path("images/info.csv") == external_root / "info.csv"

    def test_images_root_can_be_overridden_by_environment(self, tmp_path, monkeypatch):
        external_root = tmp_path / "scheduled_images"
        monkeypatch.setenv("PANOCRAWLER_IMAGES_ROOT", str(external_root))
        monkeypatch.setitem(config_module.cfg, "images_root", "images")

        assert config_module.get_images_root() == external_root
        assert config_module.resolve_images_path("pano") == external_root / "pano"

    def test_empty_images_root_environment_falls_back_to_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("PANOCRAWLER_IMAGES_ROOT", "")
        monkeypatch.setitem(config_module.cfg, "images_root", "images")

        assert config_module.get_images_root() == tmp_path / "images"

    def test_resolve_project_path_expands_user_home(self):
        assert config_module.resolve_project_path("~/panos") == Path("~/panos").expanduser()


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


class TestGetTileGridForCanvas:
    def test_exact_canvas_grid(self):
        assert get_tile_grid_for_canvas((1024, 512), (512, 512)) == (2, 1)

    def test_rounds_up_partial_tiles(self):
        assert get_tile_grid_for_canvas((1025, 513), (512, 512)) == (3, 2)

    def test_rejects_invalid_dimensions(self):
        with pytest.raises(ValueError, match="positive"):
            get_tile_grid_for_canvas((1024, 0), (512, 512))


class TestMapsTileAPIDownloader:
    def test_get_downloader_uses_environment_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")

        downloader = download_module.get_downloader()

        assert isinstance(downloader, MapsTileAPIDownloader)
        assert downloader.api_key == "env-key"

    def test_session_adapter_does_not_auto_retry_untracked_requests(self, monkeypatch):
        monkeypatch.setattr(download_module, "_session", None)

        session = download_module.get_session()
        adapter = session.get_adapter("https://example.test")

        assert adapter.max_retries.total == 0

    def test_create_session_wraps_http_errors(self, monkeypatch):
        downloader = MapsTileAPIDownloader(api_key="test")

        class FakeResponse:
            def raise_for_status(self):
                raise requests.HTTPError("forbidden")

        monkeypatch.setattr(download_module.requests, "post", lambda *_args, **_kwargs: FakeResponse())

        with pytest.raises(PanoDownloadError, match="Failed to create tile session"):
            downloader._create_session()

    def test_create_session_requires_session_token(self, monkeypatch):
        downloader = MapsTileAPIDownloader(api_key="test")

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {}

        monkeypatch.setattr(download_module.requests, "post", lambda *_args, **_kwargs: FakeResponse())

        with pytest.raises(PanoDownloadError, match="missing session token"):
            downloader._create_session()

    def test_iter_tiles_uses_custom_grid_and_skips_only_out_of_range(self, monkeypatch):
        downloader = MapsTileAPIDownloader(api_key="test")
        calls = []

        def fake_download_tile(pano_id, zoom, x, y, session):
            calls.append((x, y))
            if (x, y) == (1, 0):
                raise PanoTileOutOfRangeError("out")
            return Image.new("RGB", (1, 1), color=(x, y, 0))

        monkeypatch.setattr(downloader, "download_tile", fake_download_tile)

        tiles = list(downloader.iter_tiles("pano", zoom=1, session=object(), cols=2, rows=1))

        assert calls == [(0, 0), (1, 0)]
        assert [(tile.x, tile.y) for tile in tiles] == [(0, 0)]

    def test_iter_tiles_propagates_non_range_download_errors(self, monkeypatch):
        downloader = MapsTileAPIDownloader(api_key="test")

        def fake_download_tile(pano_id, zoom, x, y, session):
            raise PanoDownloadError("forbidden")

        monkeypatch.setattr(downloader, "download_tile", fake_download_tile)

        with pytest.raises(PanoDownloadError, match="forbidden"):
            list(downloader.iter_tiles("pano", zoom=1, session=object(), cols=1, rows=1))


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


class TestQualityMetrics:
    def test_bottom_black_edge_ratio_detects_bottom_bar(self):
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        img.paste(Image.new("RGB", (100, 10), color=(0, 0, 0)), (0, 40))

        assert bottom_black_edge_ratio(img, threshold=15) == pytest.approx(0.2)

    def test_apply_heading_adjustment_wraps_by_expected_pixels(self):
        img = Image.new("RGB", (4, 1))
        img.putpixel((0, 0), (10, 0, 0))
        img.putpixel((1, 0), (20, 0, 0))
        img.putpixel((2, 0), (30, 0, 0))
        img.putpixel((3, 0), (40, 0, 0))

        adjusted = apply_heading_adjustment(img, 180)

        assert heading_shift_pixels(4, 180) == 2
        assert [adjusted.getpixel((x, 0)) for x in range(4)] == [
            (30, 0, 0),
            (40, 0, 0),
            (10, 0, 0),
            (20, 0, 0),
        ]

    def test_quality_metrics_reports_perfect_heading_alignment(self):
        base = Image.new("RGB", (4, 2))
        for x in range(4):
            for y in range(2):
                base.putpixel((x, y), (x * 50, y * 50, 100))
        final = apply_heading_adjustment(base, 90)

        metrics = build_quality_metrics(base, final, heading=90)

        assert metrics.heading_shift_px == 1
        assert metrics.heading_mean_abs_diff == 0.0
        assert metrics.sharpness_ratio >= 0.0

    def test_sharpness_score_is_higher_for_checkerboard_than_flat_image(self):
        flat = Image.new("RGB", (8, 8), color=(128, 128, 128))
        checker = Image.new("RGB", (8, 8))
        for x in range(8):
            for y in range(8):
                color = 255 if (x + y) % 2 else 0
                checker.putpixel((x, y), (color, color, color))

        assert sharpness_score(checker) > sharpness_score(flat)


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


class TestPanoramaPipelineStages:
    @pytest.fixture
    def sample_panorama(self):
        return Panorama(
            pano_id="_-1_HSC1WAc2_lA2jmV6fw",
            lat=25.04568157378246,
            lon=121.5113594595623,
            heading=353.7138977050781,
            pitch=90.39588165283203,
            roll=359.048828125,
            date="2009-06",
            scale=[
                [[256, 512]],
                [[512, 1024]],
                [[1024, 2048]],
                [[2048, 4096]],
                [[4096, 8192]],
                [[8192, 16384]],
            ],
            zoom_resolutions=[
                (512, 256),
                (1024, 512),
                (2048, 1024),
                (4096, 2048),
                (8192, 4096),
                (16384, 8192),
            ],
            tile=[512, 512],
        )

    def test_get_canvas_size_and_zoom_resolutions(self, sample_panorama):
        assert sample_panorama.get_canvas_size(1) == (1024, 512)
        assert sample_panorama.get_zoom_resolutions()[1] == (1024, 512)

    def test_choose_best_zoom_for_output(self, sample_panorama):
        assert choose_best_zoom_for_output(sample_panorama, OUTPUT_SIZE) == 1

    def test_crop_black_edge_from_image_removes_bottom_bar(self):
        img = Image.new("RGB", (1024, 512), color=(255, 255, 255))
        black_bar = Image.new("RGB", (1024, 96), color=(0, 0, 0))
        img.paste(black_bar, (0, 512 - 96))

        cropped = crop_black_edge_from_image(img, threshold=15)

        assert cropped.size == (832, 416)

    def test_crop_black_edge_crops_redundant_horizontal_wrap(self):
        img = Image.new("RGB", (1024, 512), color=(0, 0, 0))
        content = Image.new("RGB", (1024, 416), color=(0, 255, 0))
        left = Image.new("RGB", (192, 416), color=(255, 0, 0))
        content.paste(left, (0, 0))
        content.paste(left, (832, 0))
        img.paste(content, (0, 0))

        cropped = crop_black_edge_from_image(img, threshold=15)

        assert cropped.size == (832, 416)
        assert cropped.getpixel((0, 0)) == (255, 0, 0)
        assert cropped.getpixel((831, 0)) == (0, 255, 0)

    def test_crop_black_edge_preserves_non_redundant_horizontal_content(self):
        img = Image.new("RGB", (1024, 512), color=(0, 0, 0))
        content = Image.new("RGB", (1024, 416), color=(0, 255, 0))
        content.paste(Image.new("RGB", (192, 416), color=(255, 0, 0)), (0, 0))
        content.paste(Image.new("RGB", (192, 416), color=(0, 0, 255)), (832, 0))
        img.paste(content, (0, 0))

        cropped = crop_black_edge_from_image(img, threshold=15)

        assert cropped.size == (1024, 512)
        assert cropped.getpixel((0, 0)) == (255, 0, 0)
        assert cropped.getpixel((1023, 0)) == (0, 0, 255)
        assert cropped.getpixel((1023, 511)) == (0, 0, 255)

    def test_detect_and_crop_black_edge_creates_output_dir_without_black_edge(self, tmp_path):
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "nested" / "output.png"
        Image.new("RGB", (32, 16), color=(255, 255, 255)).save(input_path)

        cropped = detect_and_crop_black_edge(input_path, output_path, threshold=15)

        assert cropped is False
        assert output_path.exists()

    def test_resize_preserves_target_output_size(self):
        img = Image.new("RGB", (1024, 416), color=(255, 255, 255))
        resized = img.resize(OUTPUT_SIZE, Image.BICUBIC)

        assert resized.size == OUTPUT_SIZE

    def test_heading_adjustment_wraps_pixels_correctly(self):
        img = Image.new("RGB", OUTPUT_SIZE)
        left = Image.new("RGB", (128, OUTPUT_SIZE[1]), color=(255, 0, 0))
        right = Image.new("RGB", (OUTPUT_SIZE[0] - 128, OUTPUT_SIZE[1]), color=(0, 255, 0))
        img.paste(left, (0, 0))
        img.paste(right, (128, 0))

        heading_norm = 353.7138977050781
        clip_x = int(heading_norm / 360.0 * OUTPUT_SIZE[0])

        clip1 = img.crop((0, 0, clip_x, OUTPUT_SIZE[1]))
        clip2 = img.crop((clip_x, 0, OUTPUT_SIZE[0], OUTPUT_SIZE[1]))
        heading_img = Image.new("RGB", OUTPUT_SIZE)
        heading_img.paste(clip2, (0, 0))
        heading_img.paste(clip1, (OUTPUT_SIZE[0] - clip_x, 0))

        assert heading_img.size == OUTPUT_SIZE
        assert heading_img.getpixel((0, 0)) == img.getpixel((clip_x, 0))

    def test_get_panorama_pipeline_with_fake_tiles(self, monkeypatch, sample_panorama):
        class PipelineDownloader:
            def iter_tiles(self, pano_id, zoom, session, cols=None, rows=None):
                for x in range(cols):
                    for y in range(rows):
                        img = Image.new("RGB", (512, 512), color=(x * 60, y * 120, 128))
                        yield Tile(x=x, y=y, image=img)

        monkeypatch.setattr("panorama.download.get_downloader", lambda: PipelineDownloader())
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        result = get_panorama(pano=sample_panorama, zoom=None, session=None)

        assert isinstance(result, Image.Image)
        assert result.size == OUTPUT_SIZE
        assert result.getpixel((0, 0)) == (60, 0, 128)

    def test_get_panorama_stages_exposes_raw_base_and_final_images(self, monkeypatch, sample_panorama):
        class PipelineDownloader:
            def iter_tiles(self, pano_id, zoom, session, cols=None, rows=None):
                for x in range(cols):
                    for y in range(rows):
                        img = Image.new("RGB", (512, 512), color=(x * 60, y * 120, 128))
                        yield Tile(x=x, y=y, image=img)

        monkeypatch.setattr("panorama.download.get_downloader", lambda: PipelineDownloader())
        monkeypatch.setattr("panorama.download.get_session", lambda: object())

        stages = get_panorama_stages(pano=sample_panorama, zoom=None, session=None)

        assert stages.raw.size == (1024, 512)
        assert stages.base.size == OUTPUT_SIZE
        assert stages.final.size == OUTPUT_SIZE
        assert stages.downloaded_tiles == 2
        assert stages.final.getpixel((0, 0)) == apply_heading_adjustment(stages.base, sample_panorama.heading).getpixel((0, 0))

    def test_get_panorama_preserves_pano_download_error(self, monkeypatch, sample_panorama):
        class FailingDownloader:
            def iter_tiles(self, pano_id, zoom, session, cols=None, rows=None):
                raise PanoDownloadError("original failure")

        monkeypatch.setattr("panorama.download.get_downloader", lambda: FailingDownloader())
        monkeypatch.setattr("panorama.download.get_session", lambda: object())

        with pytest.raises(PanoDownloadError) as exc_info:
            get_panorama(pano=sample_panorama, zoom=None, session=None)

        assert str(exc_info.value) == "original failure"


class DummyImage:
    def save(self, path, fmt):
        Path(path).write_bytes(b"fake image bytes")


class TestFetchPanoramas:
    def test_fieldnames_are_loaded_from_config(self):
        assert main.FIELDNAMES == main._cfg["fieldnames"]

    def test_default_info_path_is_not_inside_images_dir(self):
        assert main.infoPath == Path("/Users/alukardo/Documents/NTUST/研究資料/Researchs/[2026] NeRF/images/info.csv")
        assert main.panoPath == Path("/Users/alukardo/Documents/NTUST/研究資料/Researchs/[2026] NeRF/images/pano")
        assert main.infoPath.parent != main.panoPath

    def test_main_uses_random_incremental_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(main, "CRAWL_MODE", "random_incremental")
        monkeypatch.setattr(main, "init_info", lambda: calls.append("init"))
        monkeypatch.setattr(main, "fetch_random_incremental_panoramas", lambda: calls.append("random"))

        assert main.main() == 0
        assert calls == ["init", "random"]

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

        def fake_get_panorama(*, pano, zoom=None, session):
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
                "search_point": "[25.000000, 121.500000]",
            },
            {
                "pano_id": "new",
                "lat": "3.0",
                "lon": "4.0",
                "heading": "45.0",
                "pitch": "0.5",
                "roll": "0.2",
                "date": "2023-07",
                "search_point": "[25.000000, 121.500000]",
            },
        ]
        assert len(downloaded) == 1
        assert downloaded[0][0] == "new"
        assert downloaded[0][1] is None
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
        assert rows[0]["search_point"] == "[25.000000, 121.500000]"

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

    def test_process_on_download_does_not_run_second_black_edge_processing(self, tmp_path, monkeypatch):
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

        main.fetch_panoramas((25.0, 121.5), isCurrent=False)

        assert (pano_dir / "postprocess_fail.png").exists()
        assert not any(pano_dir.glob("*.tmp.png"))
        rows = list(main.csv.DictReader(info_file.open(encoding="utf-8")))
        assert [row["pano_id"] for row in rows] == ["postprocess_fail"]

    def test_random_incremental_crawl_adds_target_new_panoramas(self, tmp_path, monkeypatch):
        pano_dir = tmp_path / "images" / "pano"
        metadata_dir = tmp_path / "images"
        pano_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        info_file = metadata_dir / "info.csv"
        info_file.write_text(
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            "existing,1.0,2.0,90.0,0.0,0.0,2020-01\n",
            encoding="utf-8",
        )
        (pano_dir / "existing.png").write_bytes(b"already here")

        session = object()
        monkeypatch.setattr(main, "panoPath", pano_dir)
        monkeypatch.setattr(main, "infoPath", info_file)
        monkeypatch.setattr(main, "get_session", lambda: session)
        monkeypatch.setattr(main.time, "sleep", lambda _delay: None)
        monkeypatch.setattr(main.random, "shuffle", lambda _items: None)

        locations = iter([(10.0, 20.0), (30.0, 40.0)])
        monkeypatch.setattr(main, "random_location", lambda: next(locations))

        def fake_search(lat, lon):
            if lat == 10.0:
                return [
                    Panorama(pano_id="existing", lat=1.0, lon=2.0, heading=90.0, date="2020-01"),
                    Panorama(pano_id="new_current", lat=3.0, lon=4.0, heading=45.0, date=None),
                ]
            return [
                Panorama(pano_id="new_history", lat=5.0, lon=6.0, heading=135.0, date="2019-07"),
            ]

        downloaded = []

        def fake_get_panorama(*, pano, zoom=None, session):
            downloaded.append((pano.pano_id, zoom, session))
            return DummyImage()

        monkeypatch.setattr(main, "search_panoramas", fake_search)
        monkeypatch.setattr(main, "get_panorama", fake_get_panorama)

        added = main.fetch_random_incremental_panoramas(target_new=2, max_searches=3)

        rows = list(main.csv.DictReader(info_file.open(encoding="utf-8")))
        assert added == 2
        assert [row["pano_id"] for row in rows] == ["existing", "new_current", "new_history"]
        assert rows[1]["date"] == ""
        assert rows[2]["date"] == "2019-07"
        assert rows[0]["search_point"] == ""
        assert rows[1]["search_point"] == "[10.000000, 20.000000]"
        assert rows[2]["search_point"] == "[30.000000, 40.000000]"
        assert [item[0] for item in downloaded] == ["new_current", "new_history"]
        assert (pano_dir / "new_current.png").exists()
        assert (pano_dir / "new_history.png").exists()


class TestProcessData:
    def test_load_records_skips_invalid_rows_and_parses_optional_fields(self, tmp_path, caplog):
        metadata_path = tmp_path / "info.csv"
        metadata_path.write_text(
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            " good , 1.0 , 2.0 , 90.0 , , 0.5 , 2020-01 \n"
            "bad_lat,not-a-number,2.0,90.0,0.0,0.0,2020-02\n"
            ",3.0,4.0,90.0,0.0,0.0,2020-03\n",
            encoding="utf-8",
        )
        caplog.set_level("WARNING", logger=training_pairs_module.__name__)

        records = training_pairs_module.load_records(metadata_path)

        assert len(records) == 1
        assert records[0].pano_id == "good"
        assert records[0].heading == 90.0
        assert records[0].pitch is None
        assert records[0].roll == 0.5
        assert records[0].date == "2020-01"
        assert "Skipping invalid metadata row" in caplog.text

    def test_sample_writer_uses_start_index_for_bidirectional_outputs(self, tmp_path):
        source_dir = tmp_path / "images" / "pano"
        input_dir = tmp_path / "temp" / "train_A"
        output_dir = tmp_path / "temp" / "train_B"
        label_dir = tmp_path / "temp" / "train_cond"
        source_dir.mkdir(parents=True)
        (source_dir / "a.png").write_bytes(b"a")
        (source_dir / "b.png").write_bytes(b"b")
        writer = training_pairs_module.CopyBidirectionalSampleWriter(
            source_dir=source_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            label_dir=label_dir,
            prefix_int="int",
            prefix_out="out",
            prefix_ins="ins",
        )
        writer.prepare()

        written = writer.write_pair(
            training_pairs_module.PanoramaRecord("a", 1.0, 2.0),
            training_pairs_module.PanoramaRecord("b", 1.0, 2.0),
            start_index=7,
        )

        assert written == 2
        assert (input_dir / "int00007.png").read_bytes() == b"a"
        assert (output_dir / "out00007.png").read_bytes() == b"b"
        assert (input_dir / "int00008.png").read_bytes() == b"b"
        assert (output_dir / "out00008.png").read_bytes() == b"a"

    def test_sample_writer_skips_missing_images_without_partial_outputs(self, tmp_path, caplog):
        source_dir = tmp_path / "images" / "pano"
        input_dir = tmp_path / "temp" / "train_A"
        output_dir = tmp_path / "temp" / "train_B"
        label_dir = tmp_path / "temp" / "train_cond"
        source_dir.mkdir(parents=True)
        (source_dir / "a.png").write_bytes(b"a")
        writer = training_pairs_module.CopyBidirectionalSampleWriter(
            source_dir=source_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            label_dir=label_dir,
            prefix_int="int",
            prefix_out="out",
            prefix_ins="ins",
        )
        writer.prepare()
        caplog.set_level("WARNING", logger=training_pairs_module.__name__)

        written = writer.write_pair(
            training_pairs_module.PanoramaRecord("a", 1.0, 2.0),
            training_pairs_module.PanoramaRecord("missing", 1.0, 2.0),
            start_index=0,
        )

        assert written == 0
        assert not any(input_dir.iterdir())
        assert not any(output_dir.iterdir())
        assert not any(label_dir.iterdir())
        assert "Target file not found" in caplog.text

    def test_creates_output_dirs_when_called_directly(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "images" / "pano"
        input_dir = tmp_path / "temp" / "train_A"
        output_dir = tmp_path / "temp" / "train_B"
        label_dir = tmp_path / "temp" / "train_cond"
        source_dir.mkdir(parents=True)
        (source_dir / "a.png").write_bytes(b"a")
        (source_dir / "b.png").write_bytes(b"b")
        metadata_path = tmp_path / "info.csv"
        metadata_path.write_text(
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            "a,1.0,2.0,0.0,0.0,0.0,2020-01\n"
            "b,1.0,2.0,90.0,0.0,0.0,2020-02\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(training_pairs_module, "SOURCE_DIR", source_dir)
        monkeypatch.setattr(training_pairs_module, "INPUT_DIR", input_dir)
        monkeypatch.setattr(training_pairs_module, "OUTPUT_DIR", output_dir)
        monkeypatch.setattr(training_pairs_module, "LABEL_DIR", label_dir)

        count = training_pairs_module.process_data(metadata_path)

        assert count == 2
        assert (input_dir / "int00000.png").read_bytes() == b"a"
        assert (output_dir / "out00000.png").read_bytes() == b"b"
        assert (label_dir / "ins00000.txt").exists()

    def test_build_training_pairs_primary_entrypoint(self, tmp_path):
        source_dir = tmp_path / "images" / "pano"
        input_dir = tmp_path / "temp" / "train_A"
        output_dir = tmp_path / "temp" / "train_B"
        label_dir = tmp_path / "temp" / "train_cond"
        source_dir.mkdir(parents=True)
        (source_dir / "a.png").write_bytes(b"a")
        (source_dir / "b.png").write_bytes(b"b")
        metadata_path = tmp_path / "info.csv"
        metadata_path.write_text(
            "pano_id,lat,lon,heading,pitch,roll,date\n"
            "a,1.0,2.0,0.0,0.0,0.0,2020-01\n"
            "b,1.0,2.0,90.0,0.0,0.0,2020-02\n",
            encoding="utf-8",
        )
        writer = training_pairs_module.CopyBidirectionalSampleWriter(
            source_dir=source_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            label_dir=label_dir,
            prefix_int="int",
            prefix_out="out",
            prefix_ins="ins",
        )

        count = training_pairs_module.build_training_pairs(metadata_path, sample_writer=writer)

        assert count == 2
        assert (input_dir / "int00000.png").read_bytes() == b"a"
        assert (output_dir / "out00000.png").read_bytes() == b"b"


class TestQualityDatasetBuilder:
    def test_select_diverse_candidates_prefers_distinct_years(self):
        candidates = [
            quality_dataset.Candidate("a", 1.0, 2.0, Panorama(pano_id="p1", lat=1, lon=2, heading=0, date="2020-01")),
            quality_dataset.Candidate("a", 1.0, 2.0, Panorama(pano_id="p2", lat=1, lon=2, heading=0, date="2021-01")),
            quality_dataset.Candidate("a", 1.0, 2.0, Panorama(pano_id="p3", lat=1, lon=2, heading=0, date="2020-02")),
        ]

        selected = quality_dataset.select_diverse_candidates(candidates, sample_count=2)

        assert {candidate.year for candidate in selected} == {"2020", "2021"}

    def test_write_dataset_records_quality_metrics_without_network(self, tmp_path, monkeypatch):
        pano = Panorama(pano_id="p1", lat=1.0, lon=2.0, heading=180.0, date="2020-01")
        candidate = quality_dataset.Candidate("test_city", 1.1, 2.2, pano)
        args = SimpleNamespace(
            output_dir=str(tmp_path / "quality_dataset"),
            zoom=1,
            delay=0.0,
            max_bottom_black_ratio=0.01,
            min_sharpness_ratio=0.0,
            max_heading_diff=0.0,
        )
        raw = Image.new("RGB", (4, 2), color=(255, 255, 255))
        base = Image.new("RGB", (4, 2))
        for x in range(4):
            for y in range(2):
                base.putpixel((x, y), (x * 50, y * 50, 100))
        final = apply_heading_adjustment(base, pano.heading)
        stages = SimpleNamespace(raw=raw, base=base, final=final, zoom=1, downloaded_tiles=1)

        monkeypatch.setattr(quality_dataset, "get_session", lambda: object())
        monkeypatch.setattr(quality_dataset, "get_panorama_stages", lambda **_kwargs: stages)
        monkeypatch.setattr(quality_dataset.time, "sleep", lambda _delay: None)

        rows = quality_dataset.write_dataset(args, [candidate])

        assert rows[0]["status"] == "ok"
        assert rows[0]["quality_pass"] is True
        assert rows[0]["heading_mean_abs_diff"] == 0.0
        assert (tmp_path / "quality_dataset" / "raw" / "2020_p1.png").exists()
        assert (tmp_path / "quality_dataset" / "base" / "2020_p1.png").exists()
        assert (tmp_path / "quality_dataset" / "final" / "2020_p1.png").exists()
        assert (tmp_path / "quality_dataset" / "manifest.csv").exists()
        assert (tmp_path / "quality_dataset" / "metrics.jsonl").exists()


# ─── download_by_pano_id ─────────────────────────────────────────────────────

class FakeDownloader:
    def __init__(self, tiles):
        self.tiles = tiles

    def iter_tiles(self, pano_id, zoom, session, cols=None, rows=None):
        yield from self.tiles

class TestDownloadByPanoId:
    def test_saves_to_default_images_dir(self, tmp_path, monkeypatch):
        """download_by_pano_id saves PNG to config images_dir when output_dir is None."""
        images_dir = tmp_path / "images" / "pano"
        img = Image.new("RGB", (512, 512), color=(255, 0, 0))

        monkeypatch.setitem(download_module._cfg, "output_dir", str(images_dir))
        monkeypatch.setattr("panorama.download.get_downloader", lambda: FakeDownloader([Tile(x=0, y=0, image=img)]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        result = download_by_pano_id("test_pano_001", zoom=1, output_dir=None)

        assert result is not None
        assert isinstance(result.size, tuple)
        assert (images_dir / "test_pano_001.png").exists()

    def test_default_relative_output_dir_resolves_from_images_root(self, tmp_path, monkeypatch):
        img = Image.new("RGB", (512, 512), color=(255, 0, 0))

        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setitem(download_module._cfg, "images_root", "images")
        monkeypatch.setitem(download_module._cfg, "output_dir", "relative/pano")
        monkeypatch.setattr("panorama.download.get_downloader", lambda: FakeDownloader([Tile(x=0, y=0, image=img)]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        download_by_pano_id("relative_test", zoom=1, output_dir=None)

        assert (tmp_path / "images" / "relative" / "pano" / "relative_test.png").exists()

    def test_default_output_dir_can_use_custom_images_root(self, tmp_path, monkeypatch):
        img = Image.new("RGB", (512, 512), color=(255, 0, 0))
        external_root = tmp_path / "external_images"

        monkeypatch.setitem(download_module._cfg, "images_root", str(external_root))
        monkeypatch.setitem(download_module._cfg, "output_dir", "pano")
        monkeypatch.setattr("panorama.download.get_downloader", lambda: FakeDownloader([Tile(x=0, y=0, image=img)]))
        monkeypatch.setattr("panorama.download.get_session", lambda: object())
        monkeypatch.setattr("panorama.download.log", __import__("logging").getLogger())

        download_by_pano_id("custom_root_test", zoom=1, output_dir=None)

        assert (external_root / "pano" / "custom_root_test.png").exists()

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
