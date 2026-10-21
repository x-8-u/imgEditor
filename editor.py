# -*- coding: utf-8 -*-
"""单张编辑器：画布（棋盘格/缩放/涂抹/裁剪）+ 参数面板 + 底部导航"""
import numpy as np
import cv2
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor,
                           QKeySequence, QShortcut, QCursor, QBrush)
from PySide6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsLineItem, QGraphicsRectItem, QLabel, QPushButton, QSlider,
    QCheckBox, QSpinBox, QVBoxLayout, QHBoxLayout, QGroupBox, QButtonGroup,
    QFrame, QMessageBox)


# ---------------- 画布 ----------------
class Canvas(QGraphicsView):
    strokeCommitted = Signal()      # 一笔完成（撤销入栈 + 状态更新）
    bgPicked = Signal(tuple)        # 取色（RGB）
    cropChanged = Signal()

    TOOL_KEEP, TOOL_REMOVE, TOOL_ERASE, TOOL_CROP, TOOL_PAN = range(5)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        # 纯色深灰底：不显示棋盘格，避免干扰看清图案
        self.setBackgroundBrush(QBrush(QColor(58, 58, 58)))
        self.item = None
        self.preview_mode = "result"   # original / result / overlay
        self.tool = self.TOOL_KEEP
        self.brush_size = 40
        self.brush_soft = True
        self.pick_mode = False
        self.push_undo_needed = False
        self._painting = False
        self._erase_mode = None        # 正在涂的层
        self._last_pt = None
        self._crop_start = None
        self._space_down = False

        self.pix_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pix_item)

        pen = QPen(QColor(80, 160, 255, 160), 0, Qt.DashLine)
        self.v_guide = QGraphicsLineItem()
        self.h_guide = QGraphicsLineItem()
        for g in (self.v_guide, self.h_guide):
            g.setPen(pen)
            g.setVisible(False)
            self.scene().addItem(g)

        self.crop_item = QGraphicsRectItem()
        self.crop_item.setPen(QPen(QColor(0, 170, 255), 0, Qt.DashLine))
        self.crop_item.setVisible(False)
        self.scene().addItem(self.crop_item)

        # 涂抹重算节流
        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(80)
        self._recompute_timer.timeout.connect(self._do_recompute)

    # ---- 数据绑定 ----
    def set_item(self, item):
        self.item = item
        self._cancel_crop_drag()
        self.scene().setSceneRect(0, 0, item.source.shape[1],
                                  item.source.shape[0])
        self.update_crop_item()
        self.refresh_preview()

    def refresh_preview(self):
        item = self.item
        if item is None:
            return
        if self.preview_mode == "original":
            rgba = np.dstack([item.source, np.full_like(item.source[..., 0], 255)])
        else:
            if item.composite is None:
                item.rebuild_composite()
            if self.preview_mode == "result":
                # 被移除区域不全透明：原内容隐约可见 + 半透明紫色遮罩（仅预览，
                # 导出仍按 composite 完全透明）
                src = item.source.astype(np.float32)
                a = item.composite.astype(np.float32)
                faint = a < 128
                src[faint] = src[faint] * 0.75 + \
                    np.array([140, 40, 220], np.float32) * 0.25
                a[faint] = 110.0
                rgba = np.dstack([src.clip(0, 255).astype(np.uint8),
                                  a.clip(0, 255).astype(np.uint8)])
            else:  # overlay：原内容可见，被移除区域盖半透明紫色遮罩（PPT 式）
                mask = item.composite
                src = item.source.astype(np.float32)
                out = src.copy()
                removed = mask < 128
                out[removed] = src[removed] * 0.72 + \
                    np.array([140, 40, 220], np.float32) * 0.28
                if item.keep is not None and item.keep.any():
                    k = item.keep > 128
                    out[k] = src[k] * 0.72 + np.array([0, 210, 0], np.float32) * 0.28
                if item.remove is not None and item.remove.any():
                    r = (item.remove > 128) & (mask >= 128)
                    out[r] = src[r] * 0.72 + np.array([230, 30, 30], np.float32) * 0.28
                rgba = np.dstack([out.clip(0, 255).astype(np.uint8),
                                  np.full_like(item.source[..., 0], 255)])
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        self.pix_item.setPixmap(QPixmap.fromImage(img))
        self._update_brush_cursor()

    # ---- 撤销：由 EditorView 在笔画提交时入栈 ----

    # ---- 缩放 ----
    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        cur = self.transform().m11()
        if 0.1 <= cur * factor <= 32:
            self.scale(factor, factor)
        self._update_brush_cursor()

    def fit_view(self):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._update_brush_cursor()

    def set_zoom_100(self):
        self.resetTransform()
        self._update_brush_cursor()

    def _update_brush_cursor(self):
        if self.tool in (self.TOOL_KEEP, self.TOOL_REMOVE, self.TOOL_ERASE) \
                and not self.pick_mode:
            d = max(6, int(self.brush_size * self.transform().m11()))
            d = min(d, 240)
            pix = QPixmap(d + 4, d + 4)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            cx = (d + 4) / 2
            r = d / 2
            # 工具对应颜色：保留=绿 / 移除=红 / 橡皮=橙
            color = {self.TOOL_KEEP: QColor(0, 230, 80),
                     self.TOOL_REMOVE: QColor(255, 60, 60),
                     self.TOOL_ERASE: QColor(255, 170, 0)}[self.tool]
            # 外圈黑色描边 + 内圈高亮色粗线，再叠一圈白色虚线，任何底色都显眼
            ring = QRectF(cx - r, cx - r, d, d)
            p.setPen(QPen(QColor(0, 0, 0, 180), 4))
            p.drawEllipse(ring)
            p.setPen(QPen(color, 2))
            p.drawEllipse(ring)
            pen = QPen(QColor(255, 255, 255, 230), 1, Qt.DashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.drawEllipse(ring)
            # 中心十字
            p.setPen(QPen(QColor(255, 255, 255, 220), 1))
            p.drawLine(int(cx) - 5, int(cx), int(cx) + 5, int(cx))
            p.drawLine(int(cx), int(cx) - 5, int(cx), int(cx) + 5)
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.drawLine(int(cx) - 5, int(cx) + 1, int(cx) + 5, int(cx) + 1)
            p.end()
            self.setCursor(QCursor(pix, int(cx), int(cx)))
        elif self.pick_mode:
            self.setCursor(Qt.CrossCursor)
        elif self.tool == self.TOOL_CROP:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    # ---- 鼠标交互 ----
    def mousePressEvent(self, e):
        if self.item is None:
            return
        if self.pick_mode and e.button() == Qt.LeftButton:
            pt = self._img_pos(e)
            if pt is not None:
                c = tuple(int(v) for v in
                          self.item.source[pt[1], pt[0]])
                self.pick_mode = False
                self._update_brush_cursor()
                self.bgPicked.emit(c)
            return
        if e.button() == Qt.MiddleButton or self._space_down or \
                (self.tool == self.TOOL_PAN and e.button() == Qt.LeftButton):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = e.clone()
            super().mousePressEvent(fake)
            return
        if e.button() == Qt.LeftButton:
            if self.tool in (self.TOOL_KEEP, self.TOOL_REMOVE, self.TOOL_ERASE):
                self._start_stroke(e)
                return
            if self.tool == self.TOOL_CROP:
                self._crop_start = self._img_pos(e)
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._painting:
            self._paint_to(self._img_pos(e))
            return
        if self._crop_start is not None:
            pt = self._img_pos(e)
            if pt is not None:
                x0, y0 = self._crop_start
                rect = QRectF(min(x0, pt[0]), min(y0, pt[1]),
                              abs(pt[0] - x0), abs(pt[1] - y0))
                self.crop_item.setRect(rect)
                self.crop_item.setVisible(True)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._painting:
            self._painting = False
            self._last_pt = None
            if self._recompute_timer.isActive():
                self._recompute_timer.stop()
            self._do_recompute()
            self.strokeCommitted.emit()
            return
        if self._crop_start is not None:
            it = self.item
            if it is None:
                return
            pt = self._img_pos(e) or self._crop_start
            x0, y0 = self._crop_start
            w, h = abs(pt[0] - x0), abs(pt[1] - y0)
            if w < 3 or h < 3:
                self.update_crop_item()
            else:
                it.crop_rect = (int(min(x0, pt[0])),
                                int(min(y0, pt[1])),
                                int(w), int(h))
                self.update_crop_item()
                self.cropChanged.emit()
            self._crop_start = None
            return
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space_down = True
            self.setCursor(Qt.ClosedHandCursor)
        elif e.key() == Qt.Key_Escape and self._crop_start is not None:
            self._cancel_crop_drag()
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space_down = False
            self._update_brush_cursor()
        super().keyReleaseEvent(e)

    # ---- 涂抹 ----
    def _img_pos(self, e):
        it = self.item
        if it is None:
            return (0, 0)
        sp = self.mapToScene(e.position().toPoint())
        h, w = it.source.shape[:2]
        x, y = int(sp.x()), int(sp.y())
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return (max(0, min(x, w - 1)), max(0, min(y, h - 1)))

    def _active_layer(self):
        it = self.item
        if it is None:
            return None
        if self.tool == self.TOOL_KEEP:
            return it.keep
        if self.tool == self.TOOL_REMOVE:
            return it.remove
        return None

    def _start_stroke(self, e):
        self.push_undo_needed = True
        self._painting = True
        pt = self._img_pos(e)
        self._last_pt = pt
        self._paint_segment(pt, pt)

    def _paint_to(self, pt):
        if pt is None or self._last_pt is None:
            self._last_pt = pt
            return
        self._paint_segment(self._last_pt, pt)
        self._last_pt = pt
        self._recompute_timer.start()

    def _target_layer(self):
        it = self.item
        if it is None or self.tool == self.TOOL_ERASE:
            return None
        return it.keep if self.tool == self.TOOL_KEEP else it.remove

    def _paint_segment(self, p0, p1):
        """在笔迹层画一段（含端点圆），软边时对局部区域羽化"""
        it = self.item
        if it is None:
            return
        if self.tool == self.TOOL_ERASE:
            r = max(1, int(self.brush_size)) // 2
            for (x, y) in {p0, p1}:
                h, w = it.keep.shape[:2]
                x0, x1 = max(0, x - r), min(w, x + r + 1)
                y0, y1 = max(0, y - r), min(h, y + r + 1)
                it.keep[y0:y1, x0:x1] = 0
                it.remove[y0:y1, x0:x1] = 0
        else:
            layer = self._target_layer()
            if layer is None:
                return
            layer = np.asarray(layer)
            r = max(1, int(self.brush_size))
            cv2.line(layer, p0, p1, 255, r, cv2.LINE_AA)
            cv2.circle(layer, p0, r // 2, 255, -1, cv2.LINE_AA)
            if self.brush_soft:
                pad = r + 4
                x0 = max(0, min(p0[0], p1[0]) - pad)
                y0 = max(0, min(p0[1], p1[1]) - pad)
                x1 = min(layer.shape[1], max(p0[0], p1[0]) + pad)
                y1 = min(layer.shape[0], max(p0[1], p1[1]) + pad)
                if x1 > x0 and y1 > y0:
                    sub = layer[y0:y1, x0:x1]
                    layer[y0:y1, x0:x1] = cv2.GaussianBlur(
                        sub, (0, 0), max(1.0, self.brush_size / 8))
        self._recompute_timer.start()

    def _do_recompute(self):
        if self.item is None:
            return
        self.item.rebuild_composite()
        self.refresh_preview()

    # ---- 裁剪 ----
    def update_crop_item(self):
        if self.item and self.item.crop_rect:
            x, y, w, h = self.item.crop_rect
            self.crop_item.setRect(QRectF(x, y, w, h))
            self.crop_item.setVisible(True)
        else:
            self.crop_item.setVisible(False)

    def _cancel_crop_drag(self):
        self._crop_start = None
        self.update_crop_item()

    def set_guides(self, v_on, h_on):
        it = self.item
        if it is None:
            return
        h, w = it.source.shape[:2]
        if v_on:
            self.v_guide.setLine(w / 2, 0, w / 2, h)
        if h_on:
            self.h_guide.setLine(0, h / 2, w, h / 2)
        self.v_guide.setVisible(v_on)
        self.h_guide.setVisible(h_on)


class UndoStack(list):
    """(undo list, redo list)"""

    def __init__(self):
        super().__init__()
        self.redo = []


# ---------------- 编辑器整页 ----------------
class EditorView(QWidget):
    saveNextRequested = Signal()
    exportRequested = Signal()
    navigated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.index = -1
        self.undo = UndoStack()

        self.canvas = Canvas()

        # ---- 左侧工具 ----
        tool_box = QGroupBox("工具")
        tl = QVBoxLayout(tool_box)
        self.tool_group = QButtonGroup(self)
        self.btn_tools = {}
        for name, key, tip in (("保留笔 (B)", "keep", "涂抹必须保留的区域"),
                               ("移除笔 (E)", "remove", "涂抹必须移除的区域"),
                               ("橡皮 (R)", "erase", "擦除涂抹标记"),
                               ("裁剪 (C)", "crop", "拖出裁剪框"),
                               ("抓手 (H)", "pan", "拖拽平移，空格+拖拽随时可用")):
            b = QPushButton(name)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _, k=key: self.set_tool(k))
            self.tool_group.addButton(b)
            self.btn_tools[key] = b
            tl.addWidget(b)
        self.btn_tools["keep"].setChecked(True)

        tl.addWidget(QLabel("笔刷大小"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 300)
        self.brush_slider.setValue(40)
        self.brush_slider.valueChanged.connect(self._brush_changed)
        tl.addWidget(self.brush_slider)
        self.brush_soft_cb = QCheckBox("软边笔刷")
        self.brush_soft_cb.setChecked(True)
        self.brush_soft_cb.toggled.connect(
            lambda v: setattr(self.canvas, "brush_soft", v))
        tl.addWidget(self.brush_soft_cb)
        tl.addStretch(1)

        # ---- 右侧参数 ----
        right = QVBoxLayout()
        p1 = QGroupBox("抠图参数")
        pl = QVBoxLayout(p1)
        pl.addWidget(QLabel("容差"))
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(0, 100)
        self.tol_slider.valueChanged.connect(self._param_changed)
        pl.addWidget(self.tol_slider)
        self.connected_cb = QCheckBox("仅连通背景")
        self.connected_cb.setChecked(True)
        self.connected_cb.toggled.connect(self._param_changed)
        pl.addWidget(self.connected_cb)
        self.pick_btn = QPushButton("取背景色（点击画布）")
        self.pick_btn.setCheckable(True)
        self.pick_btn.toggled.connect(self._toggle_pick)
        pl.addWidget(self.pick_btn)

        row = QHBoxLayout()
        row.addWidget(QLabel("收缩"))
        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 10)
        self.erode_spin.valueChanged.connect(self._param_changed)
        row.addWidget(self.erode_spin)
        row.addWidget(QLabel("羽化"))
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 10)
        self.feather_spin.valueChanged.connect(self._param_changed)
        row.addWidget(self.feather_spin)
        pl.addLayout(row)
        right.addWidget(p1)

        p2 = QGroupBox("裁剪导出尺寸")
        p2l = QVBoxLayout(p2)
        wh = QHBoxLayout()
        wh.addWidget(QLabel("宽"))
        self.w_edit = QSpinBox()
        self.w_edit.setRange(1, 99999)
        wh.addWidget(self.w_edit)
        wh.addWidget(QLabel("高"))
        self.h_edit = QSpinBox()
        self.h_edit.setRange(1, 99999)
        wh.addWidget(self.h_edit)
        self.w_edit.valueChanged.connect(self._apply_wh)
        self.h_edit.valueChanged.connect(self._apply_wh)
        p2l.addLayout(wh)
        for text, cb in (("水平居中", self._crop_center_h),
                         ("垂直居中", self._crop_center_v),
                         ("居中贴合主体", self._crop_center_subject),
                         ("清除裁剪框", self._crop_clear)):
            p2l.addWidget(QPushButton(text, clicked=cb))
        right.addWidget(p2)

        self.guide_v = QCheckBox("垂直辅助线")
        self.guide_h = QCheckBox("水平辅助线")
        self.guide_v.toggled.connect(self._guides_changed)
        self.guide_h.toggled.connect(self._guides_changed)
        right.addWidget(self.guide_v)
        right.addWidget(self.guide_h)
        right.addWidget(QPushButton("重置涂抹标记", clicked=self._reset_strokes))
        right.addStretch(1)

        # ---- 底部 ----
        bottom = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        self.btn_views = {}
        for i, name in enumerate(("原图", "结果", "叠加")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 1)
            b.clicked.connect(lambda _, m=("original", "result", "overlay")[i]:
                              self._set_preview(m))
            self.view_group.addButton(b)
            self.btn_views[name] = b
            bottom.addWidget(b)
        bottom.addStretch(1)
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(lambda: self.navigate(-1))
        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(lambda: self.navigate(1))
        self.btn_savenext = QPushButton("保存并下一张 (Ctrl+Enter)")
        self.btn_savenext.clicked.connect(self.saveNextRequested.emit)
        self.btn_export = QPushButton("导出此图")
        self.btn_export.clicked.connect(self.exportRequested.emit)
        bottom.addWidget(self.btn_prev)
        bottom.addWidget(self.btn_savenext)
        bottom.addWidget(self.btn_export)
        bottom.addWidget(self.btn_next)

        mid = QHBoxLayout()
        mid.addWidget(self._tool_wrap(tool_box))
        mid.addWidget(self.canvas, 1)
        mid.addWidget(self._right_wrap(right))

        layout = QVBoxLayout(self)
        layout.addLayout(mid, 1)
        layout.addLayout(bottom)

        self._bind_shortcuts()
        self.canvas.strokeCommitted.connect(self._on_stroke)
        self.canvas.bgPicked.connect(self._on_bg_picked)
        self.canvas.cropChanged.connect(self._sync_crop_wh)

    # ---- 工具函数 ----
    @staticmethod
    def _tool_wrap(w):
        box = QWidget()
        v = QVBoxLayout(box)
        v.addWidget(w)
        v.addStretch(1)
        return box

    @staticmethod
    def _right_wrap(layout):
        box = QWidget()
        box.setFixedWidth(230)
        box.setLayout(layout)
        return box

    def _bind_shortcuts(self):
        for seq, fn in (
            ("B", lambda: self.set_tool("keep")),
            ("E", lambda: self.set_tool("remove")),
            ("R", lambda: self.set_tool("erase")),
            ("C", lambda: self.set_tool("crop")),
            ("H", lambda: self.set_tool("pan")),
            ("[", lambda: self.brush_slider.setValue(
                max(1, self.brush_slider.value() - 6))),
            ("]", lambda: self.brush_slider.setValue(
                min(300, self.brush_slider.value() + 6))),
            ("P", lambda: self._cycle_preview()),
            ("Ctrl+Z", self._undo),
            ("Ctrl+Shift+Z", self._redo),
            ("Ctrl+Return", self.saveNextRequested.emit),
            ("Ctrl+0", self.canvas.fit_view),
            ("Ctrl+1", self.canvas.set_zoom_100),
        ):
            s = QShortcut(QKeySequence(seq), self)
            s.activated.connect(fn)

    # ---- 打开/导航 ----
    def open_items(self, items, index):
        self.items = items
        self.index = index
        self._load_current()

    def _load_current(self):
        item = self.items[self.index]
        item.load()
        if item.base_mask is None:
            item.process()
        self.undo.clear()
        self.undo.redo.clear()
        p = item.params
        self.tol_slider.blockSignals(True)
        self.tol_slider.setValue(p["tolerance"])
        self.tol_slider.blockSignals(False)
        self.connected_cb.setChecked(p["connected"])
        self.erode_spin.setValue(p["erode"])
        self.feather_spin.setValue(p["feather"])
        for sb in (self.w_edit, self.h_edit):
            sb.blockSignals(True)
        self.w_edit.setValue(item.crop_rect[2] if item.crop_rect else 1)
        self.h_edit.setValue(item.crop_rect[3] if item.crop_rect else 1)
        for sb in (self.w_edit, self.h_edit):
            sb.blockSignals(False)
        self.canvas.set_item(item)
        self.canvas.fit_view()
        self.navigated.emit()

    def navigate(self, step):
        ni = self.index + step
        if 0 <= ni < len(self.items):
            self.index = ni
            self._load_current()

    # ---- 工具/预览 ----
    def set_tool(self, key):
        self.canvas.pick_mode = False
        self.pick_btn.setChecked(False)
        mapping = {"keep": Canvas.TOOL_KEEP, "remove": Canvas.TOOL_REMOVE,
                   "erase": Canvas.TOOL_ERASE, "crop": Canvas.TOOL_CROP,
                   "pan": Canvas.TOOL_PAN}
        self.canvas.tool = mapping[key]
        for k, b in self.btn_tools.items():
            b.setChecked(k == key)
        self.canvas._update_brush_cursor()

    def _set_preview(self, mode):
        self.canvas.preview_mode = mode
        for name, b in self.btn_views.items():
            b.setChecked({"原图": mode == "original",
                          "结果": mode == "result",
                          "叠加": mode == "overlay"}[name])
        self.canvas.refresh_preview()

    def _cycle_preview(self):
        order = ["original", "result", "overlay"]
        m = order[(order.index(self.canvas.preview_mode) + 1) % 3]
        self._set_preview(m)

    def _brush_changed(self, v):
        self.canvas.brush_size = v
        self.canvas._update_brush_cursor()

    # ---- 参数 ----
    def _param_changed(self, _=None):
        if self.index < 0:
            return
        item = self.items[self.index]
        item.params["tolerance"] = self.tol_slider.value()
        item.params["connected"] = self.connected_cb.isChecked()
        item.params["erode"] = self.erode_spin.value()
        item.params["feather"] = self.feather_spin.value()
        item.process()   # 笔迹保留，只重算基础 mask + 合成
        self.canvas.refresh_preview()

    def _toggle_pick(self, on):
        self.canvas.pick_mode = on
        self.canvas._update_brush_cursor()

    def _on_bg_picked(self, color):
        self.pick_btn.setChecked(False)
        item = self.items[self.index]
        item.bg_color = color
        item.process()
        self.canvas.refresh_preview()

    def _on_stroke(self):
        item = self.items[self.index]
        if self.canvas.push_undo_needed:
            self.undo.append((item.keep.copy(), item.remove.copy()))
            self.canvas.push_undo_needed = False
        item.status = "已人工调整"

    def _undo(self):
        if not self.undo:
            return
        item = self.items[self.index]
        self.undo.redo.append((item.keep.copy(), item.remove.copy()))
        k, r = self.undo.pop()
        item.keep[:] = k
        item.remove[:] = r
        item.rebuild_composite()
        self.canvas.refresh_preview()

    def _redo(self):
        if not self.undo.redo:
            return
        item = self.items[self.index]
        self.undo.append((item.keep.copy(), item.remove.copy()))
        k, r = self.undo.redo.pop()
        item.keep[:] = k
        item.remove[:] = r
        item.rebuild_composite()
        self.canvas.refresh_preview()

    def _reset_strokes(self):
        if QMessageBox.question(self, "重置", "清除该图所有涂抹标记？") == \
                QMessageBox.Yes:
            self.undo.append((self.items[self.index].keep.copy(),
                              self.items[self.index].remove.copy()))
            self.items[self.index].reset_strokes()
            self.canvas.refresh_preview()

    # ---- 裁剪 ----
    def _sync_crop_wh(self):
        r = self.items[self.index].crop_rect
        if r:
            self.w_edit.blockSignals(True)
            self.h_edit.blockSignals(True)
            self.w_edit.setValue(r[2])
            self.h_edit.setValue(r[3])
            self.w_edit.blockSignals(False)
            self.h_edit.blockSignals(False)

    def _apply_wh(self):
        w, h = self.w_edit.value(), self.h_edit.value()
        if w <= 0 or h <= 0:
            return
        item = self.items[self.index]
        ih, iw = item.source.shape[:2]
        if item.crop_rect:
            x, y, rw, rh = item.crop_rect
        else:
            rw, rh = iw * 0.8, iw * 0.8 * h / w
            if rh > ih * 0.8:
                rh = ih * 0.8
                rw = rh * w / h
            x, y = (iw - rw) / 2, (ih - rh) / 2
        cx, cy = x + rw / 2, y + rh / 2
        # 以输入宽高比例为准，保持中心重建裁剪框（限制在图内）
        rw2 = min(iw, max(rw, rh * w / h))
        rh2 = rw2 * h / w
        if rh2 > ih:
            rh2 = ih
            rw2 = rh2 * w / h
        self.items[self.index].crop_rect = (int(cx - rw2 / 2),
                                            int(cy - rh2 / 2),
                                            int(rw2), int(rh2))
        self.canvas.update_crop_item()
        self._sync_crop_wh()

    def _crop_center_h(self):
        item = self.items[self.index]
        if not item.crop_rect:
            return
        x, y, w, h = item.crop_rect
        iw = item.source.shape[1]
        item.crop_rect = ((iw - w) // 2, y, w, h)
        self.canvas.update_crop_item()

    def _crop_center_v(self):
        item = self.items[self.index]
        if not item.crop_rect:
            return
        x, y, w, h = item.crop_rect
        ih = item.source.shape[0]
        item.crop_rect = (x, (ih - h) // 2, w, h)
        self.canvas.update_crop_item()

    def _crop_center_subject(self):
        from engine import alpha_bbox
        item = self.items[self.index]
        if item.composite is None:
            item.rebuild_composite()
        bbox = alpha_bbox(item.composite)
        if bbox is None:
            return
        bx, by, bw, bh = bbox
        if not item.crop_rect:
            ih, iw = item.source.shape[:2]
            item.crop_rect = (0, 0, iw, ih)
        x, y, w, h = item.crop_rect
        item.crop_rect = (int(bx + bw / 2 - w / 2),
                          int(by + bh / 2 - h / 2), w, h)
        from engine import clamp_rect
        ih, iw = item.source.shape[:2]
        item.crop_rect = clamp_rect(item.crop_rect, iw, ih)
        self.canvas.update_crop_item()

    def _crop_clear(self):
        self.items[self.index].crop_rect = None
        self.canvas.update_crop_item()

    def _guides_changed(self):
        self.canvas.set_guides(self.guide_v.isChecked(),
                               self.guide_h.isChecked())
