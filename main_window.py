# -*- coding: utf-8 -*-
"""主窗口：图库墙 + 批量预处理 + 批量导出"""
import traceback
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget, QListWidgetItem, QStackedWidget,
    QToolBar, QLabel, QFileDialog, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QComboBox, QCheckBox, QProgressDialog,
    QMessageBox, QButtonGroup, QSpinBox)

from engine import ImageItem, export_image, clamp_rect
from editor import EditorView

STATUS_ICON = {"未处理": "⬜", "自动完成": "✅", "已人工调整": "✏️", "失败": "❌"}


class BatchWorker(QThread):
    progress = Signal(int, str)
    itemDone = Signal(int)
    finishedAll = Signal()

    def __init__(self, items, jobs, parent=None):
        super().__init__(parent)
        self.items = items
        self.jobs = jobs          # [(index, fn)]
        self.cancelled = False

    def run(self):
        for i, (idx, fn) in enumerate(self.jobs):
            if self.cancelled:
                break
            self.progress.emit(i, Path(self.items[idx].path).name)
            try:
                fn()
            except Exception:
                traceback.print_exc()
            self.itemDone.emit(idx)
        self.finishedAll.emit()


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量导出")
        self.resize(420, 240)
        v = QVBoxLayout(self)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("导出目录"))
        self.dir_edit = QLineEdit()
        row1.addWidget(self.dir_edit)
        b = QPushButton("浏览…")
        b.clicked.connect(self._pick_dir)
        row1.addWidget(b)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("导出模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["异形 PNG（透明背景）",
                                  "矩形 PNG（填充底色）",
                                  "矩形 JPG（填充底色）"])
        row2.addWidget(self.mode_combo, 1)
        v.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("底色"))
        self.bg_edit = QLineEdit("#FFFFFF")
        row3.addWidget(self.bg_edit)
        row3.addWidget(QLabel("目标宽(留空=原尺寸)"))
        self.w_edit = QLineEdit()
        self.w_edit.setFixedWidth(70)
        row3.addWidget(self.w_edit)
        row3.addWidget(QLabel("高"))
        self.h_edit = QLineEdit()
        self.h_edit.setFixedWidth(70)
        row3.addWidget(self.h_edit)
        row3.addStretch(1)
        v.addLayout(row3)

        self.name_edit = QLineEdit("_cutout")
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("文件名后缀"))
        name_row.addWidget(self.name_edit, 1)
        v.addLayout(name_row)

        v.addStretch(1)
        ok = QPushButton("开始导出")
        ok.clicked.connect(self.accept)
        v.addWidget(ok)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if d:
            self.dir_edit.setText(d)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickCut - 异形抠图批量处理")
        self.resize(1200, 780)
        self.items = []
        self.worker = None
        self.settings = QSettings("QuickCut", "QuickCut")

        # 图库页
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(self._icon_size())
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.gallery.setWordWrap(True)
        self.gallery.itemDoubleClicked.connect(self._open_editor)
        self.gallery.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.editor = EditorView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.gallery)
        self.stack.addWidget(self.editor)
        self.setCentralWidget(self.stack)
        self.setAcceptDrops(True)

        self._build_toolbar()
        self.statusBar().showMessage("导入图片开始（支持拖入文件夹/多选/粘贴）")
        self.editor.saveNextRequested.connect(self._save_and_next)
        self.editor.exportRequested.connect(self._export_current)
        self.editor.navigated.connect(
            lambda: self.statusBar().showMessage(
                f"编辑第 {self.editor.index + 1}/{len(self.items)} 张："
                f"{Path(self.items[self.editor.index].path).name}"))

    @staticmethod
    def _icon_size():
        from PySide6.QtCore import QSize
        return QSize(160, 160)

    def _build_toolbar(self):
        tb = QToolBar("main")
        tb.setMovable(False)
        self.addToolBar(tb)
        act_import = QAction("导入图片", self)
        act_import.triggered.connect(self._import_files)
        act_folder = QAction("导入文件夹", self)
        act_folder.triggered.connect(self._import_folder)
        act_batch = QAction("批量自动处理", self)
        act_batch.triggered.connect(self._batch_process)
        act_export = QAction("批量导出", self)
        act_export.triggered.connect(self._batch_export)
        act_remove = QAction("移除选中", self)
        act_remove.triggered.connect(self._remove_selected)
        for a in (act_import, act_folder, act_batch, act_export, act_remove):
            tb.addAction(a)

    # ---- 导入 ----
    def _import_files(self):
        last = self.settings.value("lastImportDir", "", str)
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", last, "图片 (*.jpg *.jpeg *.png *.webp *.bmp)")
        if files:
            self.settings.setValue("lastImportDir",
                                   str(Path(files[0]).parent))
            self._add_paths(files)

    def _import_folder(self):
        last = self.settings.value("lastImportDir", "", str)
        d = QFileDialog.getExistingDirectory(self, "选择文件夹", last)
        if not d:
            return
        self.settings.setValue("lastImportDir", d)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self._add_paths([str(p) for p in Path(d).iterdir()
                         if p.suffix.lower() in exts])

    def _add_paths(self, paths):
        start = len(self.items)
        for p in paths:
            self.items.append(ImageItem(p))
            it = QListWidgetItem(f"{STATUS_ICON['未处理']}\n{Path(p).name}")
            it.setData(Qt.ItemDataRole.UserRole, len(self.items) - 1)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gallery.addItem(it)
        self._refresh_thumbs(range(start, len(self.items)))
        self.statusBar().showMessage(f"已导入 {len(paths)} 张，共 {len(self.items)} 张")

    def _refresh_thumbs(self, indexes):
        for i in indexes:
            item = self.items[i]
            try:
                thumb = item.thumbnail()
            except Exception:
                continue
            h, w = thumb.shape[:2]
            img = QImage(thumb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            li = self.gallery.item(i)
            if li is not None:
                li.setIcon(QPixmap.fromImage(img))

    def _update_status_label(self, idx):
        li = self.gallery.item(idx)
        if li is not None:
            item = self.items[idx]
            li.setText(f"{STATUS_ICON.get(item.status, '⬜')}\n"
                       f"{Path(item.path).name}")
            tip = item.error if item.status == "失败" else item.status
            li.setToolTip(tip)

    # ---- 批量处理 ----
    def _selected_indexes(self):
        return sorted({i.data(Qt.ItemDataRole.UserRole) for i in self.gallery.selectedItems()})

    def _batch_process(self):
        sel = self._selected_indexes()
        targets = sel if sel else list(range(len(self.items)))
        if not targets:
            QMessageBox.information(self, "提示", "请先导入图片")
            return
        self._run_worker([(i, self._process_item(i)) for i in targets])

    def _process_item(self, idx):
        def fn():
            item = self.items[idx]
            item.status = "自动完成"
            item.error = ""
            item.process()
        return fn

    def _run_worker(self, jobs):
        self.worker = BatchWorker(self.items, jobs, self)
        total = len(jobs)
        self.progress = QProgressDialog("处理中…", "取消", 0, total, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.canceled.connect(self._cancel_worker)

        def on_progress(i, name):
            self.progress.setLabelText(f"({i + 1}/{total}) {name}")
            self.progress.setValue(i)

        def on_done(idx):
            self._update_status_label(idx)

        def on_all():
            self.progress.setValue(total)
            cancelled = bool(self.worker and self.worker.cancelled)
            self.statusBar().showMessage(
                "已取消（已完成项保留）" if cancelled else "处理完成")

        self.worker.progress.connect(on_progress)
        self.worker.itemDone.connect(on_done)
        self.worker.finishedAll.connect(on_all)
        self.worker.start()

    def _cancel_worker(self):
        if self.worker:
            self.worker.cancelled = True

    # ---- 编辑 ----
    def _open_editor(self, li=None):
        idx = li.data(Qt.UserRole) if li is not None else \
            self.gallery.currentRow()
        if not (0 <= idx < len(self.items)):
            return
        self.stack.setCurrentWidget(self.editor)
        self.editor.open_items(self.items, idx)

    def _save_and_next(self):
        idx = self.editor.index
        item = self.items[idx]
        item.status = "已人工调整"
        self._update_status_label(idx)
        # 找下一张
        nxt = idx + 1
        if nxt < len(self.items):
            self.editor.index = nxt
            self.editor._load_current()
        else:
            self.stack.setCurrentWidget(self.gallery)

    # ---- 导出 ----
    def _show_export_dialog(self):
        """弹出导出设置对话框，确认后返回 (dlg, out_dir, mode, target, bg, suffix)，取消返回 None"""
        dlg = ExportDialog(self)
        dlg.dir_edit.setText(self.settings.value("lastExportDir", "", str))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        out_dir = dlg.dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请选择导出目录")
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("lastExportDir", str(out_dir))
        mode = {0: "png", 1: "png_rect", 2: "jpg_rect"}[dlg.mode_combo.currentIndex()]
        bg = (255, 255, 255)
        try:
            hexs = dlg.bg_edit.text().strip().lstrip("#")
            bg = tuple(int(hexs[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            pass
        target = None
        if dlg.w_edit.text().strip() and dlg.h_edit.text().strip():
            target = (int(dlg.w_edit.text()), int(dlg.h_edit.text()))
        suffix = dlg.name_edit.text() or ""
        return dlg, out_dir, mode, target, bg, suffix

    def _export_worker_fn(self, idx, out_dir, mode, target, bg, suffix):
        def fn():
            item = self.items[idx]
            if item.source is None:
                item.load()
                item.process()
            elif item.composite is None:
                item.rebuild_composite()
            export_image(item, out_dir, mode, target, bg, suffix)
        return fn

    def _batch_export(self):
        if not self.items:
            QMessageBox.information(self, "提示", "请先导入图片")
            return
        conf = self._show_export_dialog()
        if conf is None:
            return
        _, out_dir, mode, target, bg, suffix = conf
        sel = self._selected_indexes()
        targets = sel if sel else list(range(len(self.items)))
        self._run_worker(
            [(i, self._export_worker_fn(i, out_dir, mode, target, bg, suffix))
             for i in targets])

    def _export_current(self):
        """编辑器内导出当前图"""
        if not self.items or self.editor.index < 0:
            return
        conf = self._show_export_dialog()
        if conf is None:
            return
        _, out_dir, mode, target, bg, suffix = conf
        idx = self.editor.index
        self._run_worker(
            [(idx, self._export_worker_fn(idx, out_dir, mode, target, bg, suffix))])

    # ---- 其他 ----
    def _remove_selected(self):
        for i in sorted(self._selected_indexes(), reverse=True):
            del self.items[i]
            self.gallery.takeItem(i)
        for row in range(self.gallery.count()):
            self.gallery.item(row).setData(Qt.ItemDataRole.UserRole, row)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths += [str(f) for f in p.iterdir() if f.suffix.lower() in exts]
            elif p.suffix.lower() in exts:
                paths.append(str(p))
        self._add_paths(paths)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancelled = True
            self.worker.wait(2000)
        super().closeEvent(e)
