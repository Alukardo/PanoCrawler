import pytest
import numpy as np
from panorama.search import (
    make_search_url,
    extract_panoramas,
    Panorama,
)
from panorama.download import (
    get_width_and_height_from_zoom,
    make_download_url,
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
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
        import io
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
