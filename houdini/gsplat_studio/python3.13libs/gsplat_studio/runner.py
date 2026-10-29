"""Run the pipeline steps as child processes and report their progress.

Heavy work (frame extraction, COLMAP, Karma, training) runs outside Houdini so
the UI stays usable and a crash in a worker cannot take the scene down with it.
Workers report progress on stdout; the patterns below turn that into a number.
"""
import os
import re
import subprocess

from PySide6 import QtCore

TAIL_BYTES = 8192

# Our own workers print this; it is the most reliable signal.
GSS = re.compile(r"^GSS_PROGRESS\s+([\d.]+)\s*(.*)$")
# Houdini's GSplat trainer: "[29899/29999] (99.7%) loss = 0.005 | sh degree = 3"
TRAIN = re.compile(r"^\[(\d+)/(\d+)\]\s+\(([\d.]+)%\)(.*)$")
# Karma / husk render progress.
ALF = re.compile(r"ALF_PROGRESS\s+([\d.]+)%")
# ffmpeg while pulling stills out of a video.
FFMPEG = re.compile(r"^frame=\s*(\d+)")


def read_training_progress(run_dir: str, newer_than: float = 0.0) -> tuple[float, int, int] | None:
    """How far the trainer has got, read from the run's own log file.

    The trainer runs inside a PDG work item, so its progress lines never reach
    our stdout - they go to <run>/logs/log.txt. Returns (fraction, step, total)
    or None when there is nothing fresh to report.
    """
    log = os.path.join(run_dir, "logs", "log.txt")
    if not os.path.isfile(log) or os.path.getmtime(log) < newer_than:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - TAIL_BYTES))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.split("\n")):
        m = TRAIN.match(line.strip())
        if m:
            return float(m.group(3)) / 100.0, int(m.group(1)), int(m.group(2))
    return None


def _kill_tree(proc: QtCore.QProcess) -> None:
    """Stop a step and everything it started (claude, ffmpeg, the PDG trainer):
    QProcess.kill alone leaves grandchildren running on Windows."""
    pid = proc.processId()
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    proc.kill()


class Step:
    """One child process: a command line plus the label shown to the user."""

    def __init__(self, label: str, args: list[str], weight: float = 1.0, key: str = ""):
        self.label = label
        self.key = key or label
        self.args = [str(a) for a in args]
        self.weight = weight


class JobQueue(QtCore.QObject):
    """Runs steps one after another, stopping at the first failure."""

    line = QtCore.Signal(str)
    progress = QtCore.Signal(float, str)   # 0..100 over the whole queue, status text
    finished = QtCore.Signal(bool, str)    # success, message
    step_progress = QtCore.Signal(str, float, str)  # step key, 0..1 within the step, status text
    step_done = QtCore.Signal(str)                  # step key, finished without error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: list[Step] = []
        self._index = 0
        self._done_weight = 0.0
        self._total_weight = 0.0
        self._process: QtCore.QProcess | None = None
        self._tail = ""

    @property
    def busy(self) -> bool:
        return self._process is not None

    @property
    def current_label(self) -> str:
        step = self._current()
        return step.label if step else ""

    def report(self, step_fraction: float, message: str) -> None:
        """Progress pushed in from outside, for work a child process stays quiet about."""
        if self._current() is not None:
            self._emit(step_fraction, message)

    def start(self, steps: list[Step]) -> None:
        if self.busy:
            raise RuntimeError("a job is already running")
        self._steps = steps
        self._index = 0
        self._done_weight = 0.0
        self._total_weight = sum(s.weight for s in steps) or 1.0
        self._next()

    def stop(self) -> None:
        self._steps = []
        if self._process is not None:
            _kill_tree(self._process)

    # -- internals ---------------------------------------------------------

    def _next(self) -> None:
        if self._index >= len(self._steps):
            self._process = None
            self.progress.emit(100.0, "Xong")
            self.finished.emit(True, "Xong")
            return

        step = self._steps[self._index]
        self.line.emit(f"\n=== {step.label} ===\n{' '.join(step.args)}")
        self._emit(0.0, step.label)

        proc = QtCore.QProcess(self)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.setProgram(step.args[0])
        proc.setArguments(step.args[1:])
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._step_finished)
        proc.errorOccurred.connect(self._step_error)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")  # child Pythons print Vietnamese; a pipe is cp1252 otherwise
        proc.setProcessEnvironment(env)
        self._process = proc
        self._tail = ""
        proc.start()
        proc.closeWriteChannel()  # no stdin: tools that wait for EOF (claude -p, ffmpeg) must not hang

    def _current(self) -> Step | None:
        """The running step, or None once Stop has cleared the queue."""
        if 0 <= self._index < len(self._steps):
            return self._steps[self._index]
        return None

    def _emit(self, step_fraction: float, message: str) -> None:
        step = self._current()
        if step is not None:
            self.step_progress.emit(step.key, max(0.0, min(1.0, step_fraction)), message)
        self.progress.emit(self._overall(step_fraction), message)

    def _overall(self, step_fraction: float) -> float:
        step = self._current()
        weight = step.weight if step else 0.0
        done = self._done_weight + weight * max(0.0, min(1.0, step_fraction))
        return 100.0 * done / self._total_weight

    def _read(self) -> None:
        if self._process is None:
            return
        text = self._tail + bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = text.split("\n")
        self._tail = lines.pop()  # keep the unfinished line for the next chunk
        for raw in lines:
            line = raw.rstrip("\r")
            if line:
                self.line.emit(line)
                self._parse(line)

    def _parse(self, line: str) -> None:
        step = self._current()
        if step is None:
            return
        label = step.label
        m = GSS.match(line)
        if m:
            self._emit(float(m.group(1)) / 100.0, m.group(2) or label)
            return
        m = TRAIN.match(line)
        if m:
            self._emit(float(m.group(3)) / 100.0,
                       f"{label} - buoc {m.group(1)}/{m.group(2)}")
            return
        m = ALF.search(line)
        if m:
            self._emit(float(m.group(1)) / 100.0, label)
            return
        m = FFMPEG.match(line)
        if m:
            self._emit(0.1, f"{label} - khung {m.group(1)}")

    def _step_error(self, error) -> None:
        step = self._current()
        if self._process is None or step is None:
            return
        self._fail(f"{step.label}: {self._process.errorString()}")

    def _step_finished(self, code: int, status) -> None:
        if self._process is None:
            return
        self._read()
        step = self._current()
        self._process = None
        if step is None:        # Stop cleared the queue while this was running
            self.finished.emit(False, "Da dung")
            return
        if code != 0:
            self._fail(f"{step.label} loi (ma {code})")
            return
        self.step_done.emit(step.key)
        self._done_weight += step.weight
        self._index += 1
        self._next()

    def _fail(self, message: str) -> None:
        self._steps = []
        if self._process is not None:
            _kill_tree(self._process)
            self._process = None
        self.line.emit(message)
        self.finished.emit(False, message)
