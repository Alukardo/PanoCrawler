"""测试：通过 panoID 下载全景图并打开"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from panorama import search_panoramas, download_by_pano_id


def main() -> None:
    # 台北位置
    lat, lon = 25.045711097729114, 121.51134055812804
    panos = search_panoramas(lat=lat, lon=lon)

    if not panos:
        print("未找到全景图")
        sys.exit(1)

    # 尝试每个 pano，直到成功
    for pano in panos:
        print(f"尝试 pano_id: {pano.pano_id}  日期: {pano.date}")
        try:
            img = download_by_pano_id(pano.pano_id, zoom=3)
            print(f"下载成功，尺寸: {img.size}")
            img_path = Path("images/pano") / f"{pano.pano_id}.png"
            print(f"图片已保存: {img_path}")
            import subprocess
            subprocess.run(["open", str(img_path)], check=True)
            print("已用预览打开图片")
            break
        except Exception as e:
            print(f"  失败: {e}")
    else:
        print("所有 pano 都下载失败")


if __name__ == "__main__":
    main()
