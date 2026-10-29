"""GSplat Studio - one panel for turning a video or an object into Gaussian splats.

Two ways in, one way out:
  * a 360 video  -> frames -> camera solve -> training
  * an object in the scene -> 360 render with an HDRI -> training

Everything heavy runs in a child process, so Houdini stays usable. While the
trainer runs, the panel keeps the newest splat checkpoint loaded in the
viewport, so the artist watches the result form in real 3D.
"""
import os
import time

import hou
from PySide6 import QtCore, QtWidgets

from . import config as cfg
from . import preview
from .runner import JobQueue, Step, read_training_progress

PREVIEW_SECONDS = 5
TRAIN_LABEL = "Hoc splat"


def _row(form: QtWidgets.QFormLayout, label: str, widget: QtWidgets.QWidget) -> None:
    form.addRow(label, widget)


class PickerRow(QtWidgets.QWidget):
    """A text field with a Choose button next to it."""

    def __init__(self, placeholder: str, on_pick):
        super().__init__()
        self.field = QtWidgets.QLineEdit()
        self.field.setPlaceholderText(placeholder)
        button = QtWidgets.QPushButton("Chon...")
        button.setFixedWidth(70)
        button.clicked.connect(lambda: self._pick(on_pick))
        box = QtWidgets.QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.field)
        box.addWidget(button)

    def _pick(self, on_pick) -> None:
        value = on_pick()
        if value:
            self.field.setText(value)

    def text(self) -> str:
        return self.field.text().strip()


class GSplatStudio(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GSplat Studio")
        self.queue = JobQueue(self)
        self.queue.line.connect(self._log)
        self.queue.progress.connect(self._on_progress)
        self.queue.finished.connect(self._on_finished)

        self.run_dir = ""
        self.shown_ply = ""
        self.started_at = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(PREVIEW_SECONDS * 1000)
        self.timer.timeout.connect(self._tick)

        self._build_ui()

    # -- ui ----------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._video_tab(), "Tu video 360")
        self.tabs.addTab(self._object_tab(), "Tu object trong canh")
        layout.addWidget(self.tabs)

        common = QtWidgets.QFormLayout()
        common.setContentsMargins(0, 0, 0, 0)
        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(2000, 100000)
        self.steps.setSingleStep(1000)
        self.steps.setValue(cfg.DEFAULT_STEPS)
        self.live = QtWidgets.QCheckBox("Hien splat trong viewport khi dang tinh")
        self.live.setChecked(True)
        _row(common, "So buoc hoc", self.steps)
        common.addRow("", self.live)
        layout.addLayout(common)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Bat dau")
        self.start_button.setMinimumHeight(32)
        self.start_button.clicked.connect(self._start)
        self.stop_button = QtWidgets.QPushButton("Dung")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.queue.stop)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFormat("San sang")
        layout.addWidget(self.bar)

        self.log_toggle = QtWidgets.QPushButton("Xem nhat ky")
        self.log_toggle.setCheckable(True)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setVisible(False)
        self.log_toggle.toggled.connect(self.log.setVisible)
        layout.addWidget(self.log_toggle)
        layout.addWidget(self.log, 1)

        if cfg.is_apprentice():
            note = QtWidgets.QLabel(
                "Ban Apprentice: anh render toi da 1280x720 va bi dong dau. "
                "Vung dong dau duoc xoa tu dong truoc khi hoc.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #c89b3c;")
            layout.addWidget(note)

    def _video_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.video = PickerRow("video quay quanh nhan vat", self._pick_video)
        self.video_name = QtWidgets.QLineEdit()
        self.video_name.setPlaceholderText("ten du an, vi du: woman_tpose")
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(40, 600)
        self.frames.setSingleStep(10)
        self.frames.setValue(cfg.DEFAULT_FRAMES)
        self.height_m = QtWidgets.QDoubleSpinBox()
        self.height_m.setRange(0.1, 50.0)
        self.height_m.setSingleStep(0.05)
        self.height_m.setValue(cfg.DEFAULT_HEIGHT_M)
        self.height_m.setSuffix(" m")
        _row(form, "Video", self.video)
        _row(form, "Ten", self.video_name)
        _row(form, "So anh", self.frames)
        _row(form, "Chieu cao that", self.height_m)
        return page

    def _object_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.obj = PickerRow("object can chup", self._pick_object)
        self.hdri = PickerRow("anh HDRI chieu sang", self._pick_hdri)
        self.obj_name = QtWidgets.QLineEdit()
        self.obj_name.setPlaceholderText("ten du an")
        self.views = QtWidgets.QSpinBox()
        self.views.setRange(40, 600)
        self.views.setSingleStep(10)
        self.views.setValue(cfg.DEFAULT_VIEWS)
        self.quality = QtWidgets.QComboBox()
        self.quality.addItems(list(cfg.QUALITY))
        self.quality.setCurrentText(cfg.DEFAULT_QUALITY)
        _row(form, "Object", self.obj)
        _row(form, "HDRI", self.hdri)
        _row(form, "Ten", self.obj_name)
        _row(form, "So goc nhin", self.views)
        _row(form, "Chat luong", self.quality)
        return page

    # -- pickers -----------------------------------------------------------

    def _pick_video(self) -> str:
        return hou.ui.selectFile(title="Chon video", pattern="*.mp4 *.mov *.avi *.mkv",
                                 chooser_mode=hou.fileChooserMode.Read) or ""

    def _pick_hdri(self) -> str:
        return hou.ui.selectFile(title="Chon HDRI", pattern="*.exr *.hdr *.rat *.png *.jpg",
                                 chooser_mode=hou.fileChooserMode.Read) or ""

    def _pick_object(self) -> str:
        return hou.ui.selectNode(title="Chon object", node_type_filter=hou.nodeTypeFilter.Obj) or ""

    # -- running -----------------------------------------------------------

    def _start(self) -> None:
        if self.queue.busy:
            return
        try:
            steps, name = self._build_steps()
        except ValueError as err:
            hou.ui.displayMessage(str(err), severity=hou.severityType.Warning)
            return

        self.run_dir = cfg.run_dir(name, "default")  # the Object route trains with the default strategy
        self.shown_ply = ""
        self.started_at = time.time()
        self.log.clear()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start()   # drives both the preview and the training progress
        self.queue.start(steps)

    def _build_steps(self) -> tuple[list[Step], str]:
        if self.tabs.currentIndex() == 0:
            return self._video_steps()
        return self._object_steps()

    def _video_steps(self) -> tuple[list[Step], str]:
        video, name = self.video.text(), self.video_name.text().strip()
        if not video or not os.path.isfile(video):
            raise ValueError("Chua chon video.")
        if not name:
            raise ValueError("Chua dat ten du an.")
        steps = [
            Step("Tach anh va giai camera",
                 [cfg.VENV_PYTHON, os.path.join(cfg.SCRIPTS, "video_to_dataset.py"),
                  video, name, "--frames", self.frames.value(),
                  "--height", self.height_m.value()],
                 weight=3.0),
            self._train_step(os.path.join(cfg.DATA_DIR, name)),
        ]
        return steps, name

    def _object_steps(self) -> tuple[list[Step], str]:
        obj, name = self.obj.text(), self.obj_name.text().strip()
        if not obj or hou.node(obj) is None:
            raise ValueError("Chua chon object.")
        if not name:
            raise ValueError("Chua dat ten du an.")

        work = os.path.join(cfg.DATA_DIR, "_work", name)
        os.makedirs(work, exist_ok=True)
        scene = os.path.join(work, "scene" + (".hipnc" if cfg.is_apprentice() else ".hip"))
        hou.hipFile.save(scene, save_to_recent_files=False)

        samples, close = cfg.QUALITY[self.quality.currentText()]
        width, height = cfg.render_resolution()
        args = [cfg.HYTHON, os.path.join(cfg.SCRIPTS, "gs_render_360.py"),
                "--hip", scene, "--object", obj, "--name", name,
                "--views", self.views.value(), "--close-fraction", close,
                "--samples", samples, "--res", f"{width}x{height}",
                "--data-dir", cfg.DATA_DIR]
        if self.hdri.text():
            args += ["--hdri", self.hdri.text()]
        if cfg.is_apprentice():
            args += ["--mask-watermark", ",".join(str(v) for v in cfg.WATERMARK_BOX)]

        colmap = os.path.join(cfg.DATA_DIR, f"{name}_colmap")
        steps = [
            Step("Render 360 do", args, weight=6.0),
            # pycolmap lives in tools/venv, not in Houdini's Python, so this
            # conversion runs on its own interpreter.
            Step("Chuyen sang dinh dang COLMAP",
                 [cfg.VENV_PYTHON, os.path.join(cfg.SCRIPTS, "nerfstudio_to_colmap.py"),
                  os.path.join(cfg.DATA_DIR, name), colmap],
                 weight=0.3),
            self._train_step(colmap),
        ]
        return steps, name

    def _train_step(self, dataset: str) -> Step:
        name = os.path.basename(dataset).replace("_colmap", "")
        return Step(TRAIN_LABEL,
                    [cfg.HYTHON, os.path.join(cfg.SCRIPTS, "h22_train_gsplats.py"),
                     dataset, name, "--keep-world",
                     "--steps", self.steps.value(),
                     "--ckpt-interval", cfg.CKPT_INTERVAL],
                    weight=7.0)

    # -- feedback ----------------------------------------------------------

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _on_progress(self, percent: float, message: str) -> None:
        self.bar.setValue(int(percent))
        self.bar.setFormat(f"{message} - %p%")

    def _on_finished(self, ok: bool, message: str) -> None:
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.bar.setFormat(message)
        if ok:
            self._refresh_preview()
            preview.frame_in_viewport()
        else:
            self.log_toggle.setChecked(True)

    def _tick(self) -> None:
        self._refresh_training_progress()
        if self.live.isChecked():
            self._refresh_preview()

    def _refresh_training_progress(self) -> None:
        """The trainer logs to its own file, so poll that for the progress bar."""
        if not self.run_dir or self.queue.current_label != TRAIN_LABEL:
            return
        found = read_training_progress(self.run_dir, self.started_at)
        if found is None:
            return
        fraction, step, total = found
        self.queue.report(fraction, f"{TRAIN_LABEL} - buoc {step}/{total}")

    def _refresh_preview(self) -> None:
        if not self.run_dir:
            return
        ply = preview.latest_ply(self.run_dir)
        if not ply or ply == self.shown_ply:
            return
        self.shown_ply = ply
        count = preview.show(ply)
        self._log(f"xem truoc: {os.path.basename(ply)} ({count} splat)")
