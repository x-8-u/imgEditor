# -*- coding: utf-8 -*-
"""QuickCut 核心引擎：色彩分割、mask 合成、导出"""
import cv2
import numpy as np
from pathlib import Path


def imread_rgb(path):
    """兼容中文路径读取，统一转 8bit RGB"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_png(path, rgba):
    """兼容中文路径写 PNG（RGBA）"""
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if ok:
        buf.tofile(str(path))


def imwrite_jpg(path, rgb, quality=92):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))


def sample_background(source):
    """四角 + 四边中点共 8 点取中位数背景色（RGB）"""
    h, w = source.shape[:2]
    pts = [(2, 2), (w // 2, 2), (w - 3, 2),
           (2, h // 2), (w - 3, h // 2),
           (2, h - 3), (w // 2, h - 3), (w - 3, h - 3)]  # (x, y)
    cols = np.array([source[y, x] for x, y in pts], dtype=np.float32)
    return np.median(cols, axis=0)


def compute_base_mask(source, bg_color, tolerance, connected=True):
    """Lab 色距阈值分割；连通模式只抠与边角相连的背景区域"""
    lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = np.array([[[int(c) for c in bg_color]]], dtype=np.uint8)  # (1,1,3) RGB
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    dist = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=2))
    thr = float(tolerance)  # 容差 0-100 直接作为 ΔE 阈值
    bg_like = (dist <= thr).astype(np.uint8)

    if connected:
        num, labels = cv2.connectedComponents(bg_like, connectivity=4)
        border = set(labels[0, :]) | set(labels[-1, :]) | \
                 set(labels[:, 0]) | set(labels[:, -1])
        border.discard(0)
        if border:
            bgm = np.isin(labels, np.array(sorted(border), dtype=labels.dtype))
            fg = ~bgm
        else:  # 边角完全不像背景，退化：全图同色模式
            fg = ~bg_like.astype(bool)
    else:
        fg = ~bg_like.astype(bool)

    m = fg.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def composite_mask(base, keep, remove, erode=1, feather=1):
    """基础 mask + 边缘管线 + 涂抹约束（涂抹硬值优先）"""
    m = base
    if erode > 0:
        k = erode * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.erode(m, kernel)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    m = m.copy()
    if keep is not None and keep.any():
        m[keep > 128] = 255
    if remove is not None and remove.any():
        m[remove > 128] = 0
    return m


def alpha_bbox(mask):
    ys, xs = np.where(mask > 8)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), \
        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def clamp_rect(rect, w, h):
    x, y, rw, rh = rect
    rw, rh = max(1, min(rw, w)), max(1, min(rh, h))
    x = max(0, min(int(x), w - rw))
    y = max(0, min(int(y), h - rh))
    return x, y, rw, rh


def export_image(item, out_dir, mode="png", target_size=None,
                 bg_color=(255, 255, 255), suffix="_cutout"):
    """导出单张。mode: png=异形 / png_rect / jpg_rect。返回输出路径。"""
    out_dir = Path(out_dir)
    if item.composite is None:
        item.rebuild_composite()
    src, mask = item.source, item.composite
    h, w = mask.shape[:2]

    if mode == "png":  # 异形：无裁剪框时自动取主体包围盒
        rect = item.crop_rect or alpha_bbox(mask) or (0, 0, w, h)
    else:
        rect = item.crop_rect or (0, 0, w, h)
    x, y, rw, rh = clamp_rect(rect, w, h)

    if mode == "png":
        rgba = np.dstack([src[y:y + rh, x:x + rw],
                          mask[y:y + rh, x:x + rw]])
        out = rgba
    else:
        a = (mask[y:y + rh, x:x + rw].astype(np.float32) / 255.0)[..., None]
        fg = src[y:y + rh, x:x + rw].astype(np.float32)
        comp = fg * a + np.array(bg_color, np.float32) * (1 - a)
        out = comp.clip(0, 255).astype(np.uint8)

    if target_size:
        tw, th = target_size
        interp = cv2.INTER_LANCZOS4
        out = cv2.resize(out, (tw, th), interpolation=interp)

    stem = Path(item.path).stem
    ext = ".jpg" if mode == "jpg_rect" else ".png"
    name = f"{stem}{suffix}{ext}"
    i = 2
    while (out_dir / name).exists():
        name = f"{stem}{suffix}({i}){ext}"
        i += 1
    path = out_dir / name
    if ext == ".jpg":
        imwrite_jpg(path, out)
    else:
        imwrite_png(path, out)
    return path


class ImageItem:
    """会话内单张图状态"""

    def __init__(self, path):
        self.path = Path(path)
        self.source = None          # RGB uint8
        self.base_mask = None
        self.keep = None            # 涂抹"保留"层
        self.remove = None          # 涂抹"移除"层
        self.composite = None
        self.bg_color = None        # RGB tuple
        self.params = {"tolerance": 25, "connected": True,
                       "erode": 1, "feather": 1}
        self.crop_rect = None       # (x, y, w, h) 或 None
        self.status = "未处理"       # 未处理/自动完成/已人工调整/失败
        self.error = ""
        self._thumb = None

    # ---- 处理 ----
    def load(self):
        if self.source is None:
            self.source = imread_rgb(self.path)
            self.keep = np.zeros(self.source.shape[:2], np.uint8)
            self.remove = np.zeros(self.source.shape[:2], np.uint8)
            self.bg_color = tuple(int(c) for c in sample_background(self.source))

    def process(self):
        """重算基础 mask（自动预处理 / 调参触发）"""
        self.load()
        self.base_mask = compute_base_mask(
            self.source, self.bg_color,
            self.params["tolerance"], self.params["connected"])
        self.rebuild_composite()

    def rebuild_composite(self):
        self.composite = composite_mask(
            self.base_mask, self.keep, self.remove,
            self.params["erode"], self.params["feather"])

    def reset_strokes(self):
        if self.keep is not None:
            self.keep[:] = 0
        if self.remove is not None:
            self.remove[:] = 0
        self.rebuild_composite()

    # ---- 展示 ----
    def thumbnail(self, size=160):
        if self._thumb is None:
            if self.source is None:
                self.load()
            src = self.source
            if src is None:
                raise ValueError(f"图片加载失败: {self.path}")
            scale = min(size / src.shape[1], size / src.shape[0], 1.0)
            tw = max(1, int(src.shape[1] * scale))
            th = max(1, int(src.shape[0] * scale))
            self._thumb = cv2.resize(src, (tw, th))
        return self._thumb
