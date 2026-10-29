"""GSplat Studio - a 360 video becomes a Gaussian splat in four steps.

  1 Nhap video      pick a video (or a folder of frames): frame count shown at once,
                    choose the step (like a ROP's Increment) -> how many images
  2 Tach nen        ffmpeg lossless frames + background cut   -> data/<name>
  3 COLMAP          cameras measured on the orbit             -> colmap/<name>
                    (a Claude check exists under Tuy chon, off: it missed errors in testing)
  4 Train GSplats   Houdini ML Train GSplats                   -> GSplats/<name>

Each step has its own progress bar and its own play button; "Chay het" runs
from the first unfinished step to the end. Everything heavy runs in a child
process, so Houdini stays usable. The preview box shows the newest cut-out or
the camera check; the solved cameras and the growing splat appear in the
Houdini viewport (/obj/GS_COLMAP, /obj/GS_PREVIEW).
"""
import glob
import json
import os
import subprocess
import time

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import config as cfg
from . import preview
from .runner import JobQueue, Step, read_training_progress

TICK_MS = 1500                  # preview refresh while a step runs
SPLAT_EVERY_TICKS = 4           # splat checkpoints change slowly; reload less often
VIDEO_PATTERN = "*.mp4 *.mov *.avi *.mkv *.webm"
STEPS = (("source", "1  Nhap video"), ("frames", "2  Tach nen"),
         ("colmap", "3  COLMAP"), ("train", "4  Train GSplats"))
WEIGHTS = {"frames": 2.0, "colmap": 3.0, "train": 6.0}


def _folders(name: str) -> dict:
    return {
        "data": os.path.join(cfg.DATA_DIR, name),
        "colmap": os.path.join(cfg.COLMAP_DIR, name),
        "work": os.path.join(cfg.COLMAP_DIR, "_work", name),
        "run": cfg.run_dir(name, base=cfg.GSPLATS_DIR),
    }


class StepRow(QtWidgets.QWidget):
    """Title, progress bar (with the status text inside) and a play button for one step."""

    def __init__(self, title: str, on_play):
        super().__init__()
        self.title = QtWidgets.QLabel(f"<b>{title}</b>")
        self.title.setFixedWidth(130)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        self.bar.setFormat("")
        self.play = QtWidgets.QToolButton()
        self.play.setText("▶")
        self.play.setToolTip("Chay rieng buoc nay")
        self.play.clicked.connect(on_play)
        box = QtWidgets.QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.title)
        box.addWidget(self.bar, 1)
        box.addWidget(self.play)

    def set(self, fraction: float, text: str) -> None:
        self.bar.setValue(int(1000 * max(0.0, min(1.0, fraction))))
        self.bar.setFormat(text)


class GSplatStudio(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GSplat Studio")
        self.queue = JobQueue(self)
        self.queue.line.connect(self._on_line)
        self.queue.step_progress.connect(self._on_step_progress)
        self.queue.step_done.connect(self._on_step_done)
        self.queue.finished.connect(self._on_finished)

        self.info: dict = {}
        self.auto_name = ""
        self.folders: dict = {}
        self.running: list[str] = []
        self.shown_image = ""
        self.shown_ply = ""
        self.started_at = 0.0
        self.ticks = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self._build_ui()

    # -- ui ----------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        src_row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QLineEdit()
        self.source.setPlaceholderText("Video quay 360 (hoac thu muc anh)")
        self.source.editingFinished.connect(lambda: self._load_source(self.source.text().strip()))
        pick_video = QtWidgets.QPushButton("Video...")
        pick_video.clicked.connect(lambda: self._pick(False))
        pick_dir = QtWidgets.QPushButton("Thu muc...")
        pick_dir.clicked.connect(lambda: self._pick(True))
        src_row.addWidget(self.source, 1)
        src_row.addWidget(pick_video)
        src_row.addWidget(pick_dir)
        layout.addLayout(src_row)

        opts = QtWidgets.QHBoxLayout()
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("ten nhan vat")
        self.name.editingFinished.connect(self._refresh_status)
        self.step = QtWidgets.QSpinBox()
        self.step.setRange(1, 50)
        self.step.setToolTip("Lay 1 anh moi N khung (1 = lay het), nhu Increment cua ROP")
        self.step.valueChanged.connect(self._update_count)
        self.count = QtWidgets.QLabel("")
        opts.addWidget(QtWidgets.QLabel("Ten"))
        opts.addWidget(self.name, 1)
        opts.addWidget(QtWidgets.QLabel("Buoc nhay"))
        opts.addWidget(self.step)
        opts.addWidget(self.count)
        layout.addLayout(opts)

        self.rows = {key: StepRow(title, lambda _=False, k=key: self._run([k])) for key, title in STEPS}
        self.rows["source"].play.hide()  # step 1 happens as soon as a source is picked
        for row in self.rows.values():
            layout.addWidget(row)

        buttons = QtWidgets.QHBoxLayout()
        self.run_all = QtWidgets.QPushButton("Chay het")
        self.run_all.setMinimumHeight(30)
        self.run_all.clicked.connect(self._run_remaining)
        self.stop = QtWidgets.QPushButton("Dung")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.queue.stop)
        buttons.addWidget(self.run_all, 1)
        buttons.addWidget(self.stop)
        layout.addLayout(buttons)

        self.view = QtWidgets.QLabel("Chon video de bat dau")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumHeight(240)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.view.setStyleSheet("background:#1d1d1d; color:#888; border:1px solid #333;")
        layout.addWidget(self.view, 1)

        self._build_advanced(layout)

    def _build_advanced(self, layout: QtWidgets.QVBoxLayout) -> None:
        toggles = QtWidgets.QHBoxLayout()
        adv_toggle = QtWidgets.QToolButton()
        adv_toggle.setText("Tuy chon")
        adv_toggle.setCheckable(True)
        self.log_toggle = QtWidgets.QToolButton()
        self.log_toggle.setText("Nhat ky")
        self.log_toggle.setCheckable(True)
        toggles.addWidget(adv_toggle)
        toggles.addWidget(self.log_toggle)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        self.advanced = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.advanced)
        form.setContentsMargins(0, 0, 0, 0)
        self.use_ai = QtWidgets.QCheckBox("Claude kiem tra goc camera (thu nghiem)")
        self.use_ai.setChecked(False)
        self.height_m = QtWidgets.QDoubleSpinBox()
        self.height_m.setRange(0.1, 50.0)
        self.height_m.setValue(cfg.DEFAULT_HEIGHT_M)
        self.height_m.setSuffix(" m")
        self.train_steps = QtWidgets.QSpinBox()
        self.train_steps.setRange(2000, 100000)
        self.train_steps.setSingleStep(5000)
        self.train_steps.setValue(cfg.DEFAULT_STEPS)
        form.addRow("", self.use_ai)
        form.addRow("Chieu cao nhan vat", self.height_m)
        form.addRow("So buoc train", self.train_steps)
        self.advanced.hide()
        adv_toggle.toggled.connect(self.advanced.setVisible)
        layout.addWidget(self.advanced)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible)
        layout.addWidget(self.log, 1)

    # -- step 1: source ----------------------------------------------------

    def _pick(self, folder: bool) -> None:
        if folder:
            path = hou.ui.selectFile(title="Thu muc anh", file_type=hou.fileType.Directory)
        else:
            path = hou.ui.selectFile(title="Video 360", pattern=VIDEO_PATTERN)
        if path:
            path = hou.text.expandString(path).rstrip("/")
            self.source.setText(path)
            self._load_source(path)

    def _load_source(self, path: str) -> None:
        if not path or self.info.get("path") == path:
            return
        self.rows["source"].set(0.3, "Dang doc...")
        QtWidgets.QApplication.processEvents()
        try:
            out = subprocess.run([cfg.VENV_PYTHON, os.path.join(cfg.SCRIPTS, "gs_probe.py"), path],
                                 capture_output=True, text=True, timeout=60,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.info = dict(json.loads(out.stdout.strip().splitlines()[-1]), path=path)
        except (subprocess.SubprocessError, ValueError, IndexError):
            self.info = {}
            self.rows["source"].set(0.0, "Khong doc duoc nguon nay")
            return
        # follow the new source unless the artist typed a name of their own
        if self.name.text().strip() in ("", self.auto_name):
            base = os.path.splitext(os.path.basename(path))[0]
            self.auto_name = "".join(c if c.isalnum() else "_" for c in base).strip("_").lower()
            self.name.setText(self.auto_name)
        fps = f" - {self.info['fps']:g} fps" if self.info.get("fps") else ""
        self.rows["source"].set(1.0, f"{self.info['frames']} khung - {self.info['width']}x{self.info['height']}{fps}")
        self._update_count()
        self._refresh_status()

    def _update_count(self) -> None:
        frames = self.info.get("frames", 0)
        n = -(-frames // self.step.value()) if frames else 0
        self.count.setText(f"-> {n} anh" if n else "")

    # -- running -------------------------------------------------------------

    def _build_step(self, key: str, name: str) -> Step | None:
        f = _folders(name)
        py, scripts = cfg.VENV_PYTHON, cfg.SCRIPTS
        if key == "frames":
            if not self.info:
                return None
            args = [py, os.path.join(scripts, "gs_frames.py"), self.info["path"], name, "--step", self.step.value()]
            if cfg.KEEP_BG:
                args.append("--keep-bg")
            return Step("Tach nen", args, WEIGHTS[key], key)
        if key == "colmap":
            args = [py, os.path.join(scripts, "gs_colmap_ai.py"), name, "--height", self.height_m.value(),
                    "--rounds", cfg.AI_ROUNDS]
            if cfg.KEEP_BG:
                args.append("--keep-bg")
            if self.use_ai.isChecked():
                args.append("--ai")
            return Step("COLMAP", args, WEIGHTS[key], key)
        if key == "train":
            return Step("Train GSplats", [cfg.HYTHON, os.path.join(scripts, "h22_train_gsplats.py"), f["colmap"],
                                          name, "--keep-world", "--strategy", cfg.DEFAULT_STRATEGY,
                                          "--steps", self.train_steps.value(),
                                          "--ckpt-interval", cfg.CKPT_INTERVAL, "--out-dir", cfg.GSPLATS_DIR],
                        WEIGHTS[key], key)
        return None

    def _run(self, keys: list[str]) -> None:
        name = self.name.text().strip()
        if self.queue.busy:
            return
        if not name:
            hou.ui.displayMessage("Dat ten nhan vat truoc.")
            return
        if "frames" in keys and "frames" in self._done_steps(name):
            choice = hou.ui.displayMessage(f"'{name}' da co anh. Lam lai buoc 2 se xoa anh cu.",
                                           buttons=("Lam lai", "Huy"), default_choice=1, close_choice=1)
            if choice != 0:
                return
        steps = [s for s in (self._build_step(k, name) for k in keys) if s]
        if not steps:
            hou.ui.displayMessage("Chon video truoc.")
            return
        self.running = [s.key for s in steps]
        for key in self.running:
            self.rows[key].set(0.0, "Cho...")
        self.folders = _folders(name)
        self.shown_ply = ""
        self.started_at = time.time()
        self.run_all.setEnabled(False)
        self.stop.setEnabled(True)
        self.timer.start()
        self.queue.start(steps)

    def _run_remaining(self) -> None:
        done = self._done_steps(self.name.text().strip())
        todo = [k for k, _ in STEPS[1:] if k not in done]
        self._run(todo or [k for k, _ in STEPS[1:]])

    # -- queue events ----------------------------------------------------------

    def _on_line(self, line: str) -> None:
        self.log.appendPlainText(line)
        if line.startswith("GSS_PREVIEW "):
            self.shown_image = ""  # same file name every round, new picture
            self._show_image(line[len("GSS_PREVIEW "):].strip())
        elif line.startswith("[Claude] SUMMARY:"):
            self.rows["colmap"].bar.setToolTip(line[len("[Claude] SUMMARY:"):].strip())

    def _on_step_progress(self, key: str, fraction: float, message: str) -> None:
        if key in self.rows:
            self.rows[key].set(fraction, f"{message} - {int(fraction * 100)}%")

    def _on_step_done(self, key: str) -> None:
        self.running = [k for k in self.running if k != key]
        self.rows[key].set(1.0, "Xong")
        if key == "colmap":
            self._show_colmap()
        elif key == "train":
            self._refresh_splat(force=True)
            preview.frame_in_viewport()

    def _on_finished(self, ok: bool, message: str) -> None:
        self.timer.stop()
        self.run_all.setEnabled(True)
        self.stop.setEnabled(False)
        if not ok:
            for key in self.running:  # the failed step and the ones that never started
                row = self.rows[key]
                row.set(row.bar.value() / 1000, f"Dung: {message}")
            self.log_toggle.setChecked(True)
        self.running = []

    # -- live preview ------------------------------------------------------------

    def _tick(self) -> None:
        self.ticks += 1
        key = self.running[0] if self.running else ""
        if key == "frames":
            self._show_newest_frame()
        elif key == "train":
            self._refresh_training_progress()
            if self.ticks % SPLAT_EVERY_TICKS == 0:
                self._refresh_splat()

    def _show_newest_frame(self) -> None:
        for sub in ("images", "frames"):
            files = sorted(glob.glob(os.path.join(self.folders["data"], sub, "*.png")))
            if files:
                # the newest finished file, not the one still being written
                self._show_image(files[-2] if len(files) > 1 else files[-1])
                return

    def _show_image(self, path: str) -> None:
        if not os.path.isfile(path) or path == self.shown_image and self.view.pixmap():
            return
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            return
        self.shown_image = path
        self.view.setPixmap(pix.scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self.shown_image:
            path, self.shown_image = self.shown_image, ""
            self._show_image(path)

    def _show_colmap(self) -> None:
        work = self.folders["work"]
        points = preview.show_colmap(os.path.join(work, "preview_points.ply"), os.path.join(work, "preview_cams.obj"))
        self.log.appendPlainText(f"COLMAP: {points} diem 3D, xem trong viewport (/obj/{preview.COLMAP_NODE})")
        preview.frame_in_viewport(preview.COLMAP_NODE)

    def _refresh_training_progress(self) -> None:
        status = read_training_progress(self.folders["run"], self.started_at)
        if status:
            fraction, step, total = status
            self.queue.report(fraction, f"Train - buoc {step}/{total}")

    def _refresh_splat(self, force: bool = False) -> None:
        ply = preview.latest_ply(self.folders["run"])
        if ply and (force or ply != self.shown_ply):
            self.shown_ply = ply
            count = preview.show(ply)
            self.log.appendPlainText(f"Splat: {os.path.basename(ply)} - {count} diem")

    # -- what a name already has ---------------------------------------------------

    def _done_steps(self, name: str) -> set[str]:
        if not name:
            return set()
        f = _folders(name)
        done = set()
        if glob.glob(os.path.join(f["data"], "images", "*.png")):
            done.add("frames")
        if os.path.isfile(os.path.join(f["colmap"], "sparse", "0", "cameras.bin")):
            done.add("colmap")
        if glob.glob(os.path.join(f["run"], "gsplats", "*.ply")):
            done.add("train")
        return done

    def _refresh_status(self) -> None:
        if self.queue.busy:
            return
        done = self._done_steps(self.name.text().strip())
        for key, _ in STEPS[1:]:
            self.rows[key].set(1.0 if key in done else 0.0, "Da co ket qua" if key in done else "")
