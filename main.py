from __future__ import annotations
import sys, os, warnings, re
from io import BytesIO

# --- приглушаем конкретные варнинги зависимостей ---
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", message="A new version of Albumentations is available", module="albumentations")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", module="pydantic")

from PySide6.QtCore import Qt, QRect, QPoint, QSize, QBuffer, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QPixmap, QImage, QPainter, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSplitter, QPlainTextEdit, QLineEdit, QFrame, QMessageBox, QSizePolicy, QRubberBand,
    QCheckBox
)

from sympy import latex as sympy_latex, sympify
from sympy.core.sympify import SympifyError

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rcParams

from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw
from pix2tex.cli import LatexOCR

# --- настройки ---
PREVIEW_DPI = 140
SNIP_HIDE_DELAY_MS = 200   # задержка перед запуском оверлея после hide()

# torch (опционально)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# --- Latex helpers ---
LATEX_ENV_RE = re.compile(r'\\begin\{.*?\}|\\\[[\s\S]*?\\\]|\\\(.+?\\\)', re.DOTALL)
MATH_TEXT_CMD_MAP = [
    (re.compile(r"\\arg(?![A-Za-z])"), r"\\operatorname{arg}"),
    (re.compile(r"\\Arg(?![A-Za-z])"), r"\\operatorname{Arg}"),
]

def sanitize_latex(lx: str) -> str:
    s = lx.strip()
    if s.startswith('$$') and s.endswith('$$'):
        s = s[2:-2].strip()
    if s.startswith(r'\[') and s.endswith(r'\]'):
        s = s[2:-2].strip()
    if s.startswith(r'\(') and s.endswith(r'\)'):
        s = s[2:-2].strip()
    return s

def build_tex_document(latex_body: str) -> str:
    body = sanitize_latex(latex_body)
    is_env = LATEX_ENV_RE.search(body) is not None or r'\begin{' in body
    math_block = body if is_env else f"\\[\n{body}\n\\]"
    return (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{margin=1in}\n"
        "\\begin{document}\n"
        f"{math_block}\n"
        "\\end{document}\n"
    )

def _render_mathtext(latex_str: str, dpi: int) -> QImage:
    body = sanitize_latex(latex_str)
    for rx, repl in MATH_TEXT_CMD_MAP:
        body = rx.sub(repl, body)
    use_wrap = not LATEX_ENV_RE.search(body)
    text_expr = f"${body}$" if use_wrap else body
    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    fig.patch.set_alpha(0.0)
    text = fig.text(0.5, 0.5, text_expr, ha="center", va="center")
    canvas = FigureCanvas(fig); canvas.draw()
    bbox = text.get_window_extent(renderer=canvas.get_renderer()).expanded(1.2, 1.4)
    w, h = int(bbox.width), int(bbox.height)
    fig2 = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig2.patch.set_alpha(0.0)
    fig2.text(0, 0, text_expr, ha="left", va="baseline")
    canvas2 = FigureCanvas(fig2); canvas2.draw()
    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); plt.close(fig2)
    return QImage.fromData(buf.getvalue(), "PNG")

def _render_with_usetex(latex_str: str, dpi: int) -> QImage:
    old_usetex = rcParams.get('text.usetex', False)
    old_preamble = rcParams.get('text.latex.preamble', '')
    try:
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
        body = sanitize_latex(latex_str)
        is_env = LATEX_ENV_RE.search(body) is not None or r'\begin{' in body
        text_expr = body if is_env else f"${body}$"
        fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
        fig.patch.set_alpha(0.0)
        text = fig.text(0.5, 0.5, text_expr, ha="center", va="center")
        canvas = FigureCanvas(fig); canvas.draw()
        bbox = text.get_window_extent(renderer=canvas.get_renderer()).expanded(1.2, 1.4)
        w, h = int(bbox.width), int(bbox.height)
        fig2 = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig2.patch.set_alpha(0.0)
        fig2.text(0, 0, text_expr, ha="left", va="baseline")
        canvas2 = FigureCanvas(fig2); canvas2.draw()
        buf = BytesIO()
        fig2.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig); plt.close(fig2)
        return QImage.fromData(buf.getvalue(), "PNG")
    finally:
        rcParams['text.usetex'] = old_usetex
        rcParams['text.latex.preamble'] = old_preamble

def render_latex_to_qimage(latex_str: str, dpi: int = PREVIEW_DPI) -> QImage:
    need_full = LATEX_ENV_RE.search(latex_str) is not None or latex_str.strip().startswith(('$$', r'\[', r'\('))
    try:
        if not need_full:
            return _render_mathtext(latex_str, dpi)
        return _render_with_usetex(latex_str, dpi)
    except Exception:
        try:
            return _render_with_usetex(latex_str, dpi)
        except Exception:
            return QImage()

# === постобработка распознанного LaTeX: исправление штрихов (расширено, БЕЗ строк-замен) ===
UNICODE_PRIMES = [
    ("\u2032", r"^{\prime}"),                  # ′
    ("\u2033", r"^{\prime\prime}"),            # ″
    ("\u2034", r"^{\prime\prime\prime}"),      # ‴
    ("\u2057", r"^{\prime\prime\prime\prime}") # ⁗
]
PRIME_AFTER_TOKEN = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])('{1,4})")
# последовательности штрихов внутри скобок степеней/индексов
PRIME_APOSTROPHE_RUN = re.compile(r"(?<!\\)(?<![A-Za-z])'{1,4}(?![A-Za-z])")
# паттерны, которые нужно «склеивать» в двойной штрих
PRIME_GLUE_PATTERNS = [
    re.compile(r"(\^\{\s*\\prime\s*\})\s*'"),
    re.compile(r"'\s*(\^\{\s*\\prime\s*\})"),
    re.compile(r"\^\{\s*\\prime\s*\}\s*\{\s*\\prime\s*\}"),
    re.compile(r"\^\{\s*\\prime\s*\}\s*\\prime"),
]
# шаблон для выделения скобок вида ^{m(1)} → ^{\prime\prime}(1)
PAREN_BLOCK = r"(?:\\left\s*)?\([^{}]*?(?:\\right\s*)?\)"
DOUBLE_PRIME_PREFIX_BRACED = re.compile(
    rf"\^\{{\s*(?P<token>m|rn)\s*(?P<rest>{PAREN_BLOCK})\s*\}}"
)
DOUBLE_PRIME_PREFIX = re.compile(
    rf"\^\s*(?P<token>m|rn)\s*(?P<rest>{PAREN_BLOCK})"
)
FAKE_ONE_AS_PRIME = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])\s*\^\{\s*[1l]\s*\}(\s*(?=[\(\[]))?")
SUPERSCRIPT_CHAIN = re.compile(r"\^\{\s*([^{}]*?)\s*\}\s*\{\s*\}\s*\^\{\s*([^{}]*?)\s*\}")

def _replace_apostrophe_runs(segment: str) -> str:
    return PRIME_APOSTROPHE_RUN.sub(lambda m: r"\\prime" * len(m.group(0)), segment)

def _normalize_braced_primes(text: str) -> str:
    def _walk(s: str) -> str:
        res: list[str] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in "^_":
                res.append(ch)
                i += 1
                if i < len(s) and s[i] == '{':
                    depth = 1
                    j = i + 1
                    while j < len(s) and depth:
                        if s[j] == '{':
                            depth += 1
                        elif s[j] == '}':
                            depth -= 1
                        j += 1
                    inner = s[i + 1:j - 1] if j - 1 > i else ''
                    processed_inner = _walk(inner)
                    processed_inner = _replace_apostrophe_runs(processed_inner)
                    res.append('{'); res.append(processed_inner); res.append('}')
                    i = j
                    continue
            res.append(ch)
            i += 1
        combined = ''.join(res)
        return _replace_apostrophe_runs(combined)
    return _walk(text)

def collapse_superscript_chain(text: str) -> str:
    def _repl(m: re.Match) -> str:
        left = m.group(1).strip()
        right = m.group(2).strip()
        if not left:
            return f"^{{{right}}}"
        if not right:
            return f"^{{{left}}}"
        return f"^{{{left}{right}}}"

    prev = None
    out = text
    while prev != out:
        prev = out
        out = SUPERSCRIPT_CHAIN.sub(_repl, out)
    return out

def _apply_double_prime_prefixes(text: str) -> str:
    def _rest_or_empty(rest: str, following: str) -> str:
        if not rest:
            return ''
        after = following.lstrip()
        if after.startswith(rest):
            return ''
        return rest

    def _repl_braced(m: re.Match) -> str:
        rest = m.group('rest') or ''
        tail = text[m.end():]
        keep = _rest_or_empty(rest, tail)
        return f"^{{\\prime\\prime}}{keep}"

    text = DOUBLE_PRIME_PREFIX_BRACED.sub(_repl_braced, text)

    def _repl_plain(m: re.Match) -> str:
        rest = m.group('rest') or ''
        tail = text[m.end():]
        keep = _rest_or_empty(rest, tail)
        return f"^{{\\prime\\prime}}{keep}"

    text = DOUBLE_PRIME_PREFIX.sub(_repl_plain, text)
    return text

def _replace_caret_shortcuts(text: str) -> str:
    def _prime_for_sequence(seq: str) -> str | None:
        core = seq.replace(' ', '')
        if not core:
            return None
        if all(ch in 'nN' for ch in core):
            count = len(core)
            if count <= 4:
                return '{' + r"\\prime" * count + '}'
        return None

    res: list[str] = []
    i = 0
    ln = len(text)
    while i < ln:
        ch = text[i]
        if ch != '^':
            res.append(ch)
            i += 1
            continue

        j = i + 1
        # пропускаем пробелы после ^
        while j < ln and text[j].isspace():
            j += 1

        if j < ln and text[j] == '{':
            depth = 1
            k = j + 1
            while k < ln and depth:
                if text[k] == '{':
                    depth += 1
                elif text[k] == '}':
                    depth -= 1
                k += 1
            if depth != 0:
                res.append(ch)
                i += 1
                continue
            inner = text[j + 1:k - 1]
            repl = _prime_for_sequence(inner)
            if repl:
                res.append('^' + repl)
            else:
                res.append(text[i:k])
            i = k
            continue

        k = j
        while k < ln and text[k] in ('n', 'N', ' '):
            k += 1
        seq = text[j:k]
        repl = _prime_for_sequence(seq)
        if repl and not (k < ln and text[k] in '([\'_"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'):  # avoid touching real exponents
            res.append('^' + repl)
            i = k
            continue

        res.append('^')
        i += 1

    return ''.join(res)

def fix_primes_heuristic(s: str, allow_caret_n: bool = True, allow_one_as_prime: bool = True) -> str:
    out = s
    # 1) юникодные штрихи → \prime
    for uni, repl in UNICODE_PRIMES:
        out = out.replace(uni, repl)
    # 2) ASCII штрихи сразу после токена: x' x'' )' }'
    def _rep(m: re.Match) -> str:
        base = m.group("base")
        primes = len(m.group(0)) - len(base)
        return base + '^{' + r"\\prime" * primes + '}'
    out = PRIME_AFTER_TOKEN.sub(_rep, out)
    # 3) склейка разбитых двойных штрихов (через lambda, чтобы не было bad escape)
    for rx in PRIME_GLUE_PATTERNS:
        out = rx.sub(lambda _m: r"^{\prime\prime}", out)
    # 4) ^n → ^{\prime} (если разрешено)
    if allow_caret_n:
        out = _replace_caret_shortcuts(out)
    # 4b) двойные штрихи, распознанные как m/rn перед (⋅)
    out = _apply_double_prime_prefixes(out)
    # 5) ^{1} / ^{l} как штрих перед скобкой: f^{1}(x) → f^{\prime}(x)
    if allow_one_as_prime:
        out = FAKE_ONE_AS_PRIME.sub(lambda m: f"{m.group('base')}^{{\\prime}}", out)
    # 6) вложенные фигурные скобки в степенях/индексах
    out = _normalize_braced_primes(out)
    return out

# --- Оверлей выделения области ---
class SnipOverlay(QWidget):
    captured = Signal(QImage)
    canceled = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)
        self.virtual_geo = QGuiApplication.primaryScreen().virtualGeometry()
        self.setGeometry(self.virtual_geo)

        self.origin = QPoint()
        self.dragging = False
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)

        self.backing = QPixmap(self.virtual_geo.size())
        self.backing.fill(Qt.transparent)
        painter = QPainter(self.backing)
        for screen in QGuiApplication.screens():
            g = screen.geometry()
            pix = screen.grabWindow(0)
            painter.drawPixmap(g.topLeft() - self.virtual_geo.topLeft(), pix)
        painter.end()

        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def showEvent(self, e):
        super().showEvent(e)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    def _cancel(self):
        self.rubber.hide()
        self.canceled.emit()
        self.close()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._cancel(); return
        self.origin = e.globalPosition().toPoint()
        self.rubber.setGeometry(QRect(self.mapFromGlobal(self.origin), QSize()))
        self.rubber.show()
        self.dragging = True

    def mouseMoveEvent(self, e):
        if not self.dragging:
            return
        current = e.globalPosition().toPoint()
        rect = QRect(self.mapFromGlobal(self.origin), self.mapFromGlobal(current)).normalized()
        self.rubber.setGeometry(rect)
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.dragging:
            return
        self.dragging = False
        rect_local = self.rubber.geometry()
        self.rubber.hide()
        if rect_local.width() < 3 or rect_local.height() < 3:
            self._cancel(); return
        rect_global = QRect(rect_local)
        rect_global.translate(self.virtual_geo.topLeft())
        self._capture_rect(rect_global)
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._cancel()

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(0, 0, self.backing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if self.rubber.isVisible():
            sel = self.rubber.geometry()
            sub = self.backing.copy(QRect(sel.topLeft(), sel.size()))
            p.drawPixmap(sel.topLeft(), sub)
            pen = p.pen(); pen.setWidth(2); p.setPen(pen)
            p.drawRect(sel)

    def closeEvent(self, e):
        if self.dragging:
            self.canceled.emit()
        super().closeEvent(e)

    def _capture_rect(self, rect_global: QRect):
        if rect_global.isEmpty():
            self._cancel(); return
        local = QRect(rect_global)
        local.translate(-self.virtual_geo.topLeft())
        cropped = self.backing.copy(local)
        self.captured.emit(cropped.toImage())

# --- Главное окно ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen & LaTeX Helper")
        self.setMinimumSize(1200, 700)

        # левая панель
        self.preview = QLabel("Скрин-предпросмотр")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFrameShape(QFrame.StyledPanel)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        btn_region = QPushButton("Скриншот области")
        btn_copy = QPushButton("Копировать изображение")
        btn_save = QPushButton("Сохранить PNG")
        btn_ocr_pix = QPushButton("Распознать (pix2tex, офлайн)")

        btn_region.clicked.connect(self.capture_region)
        btn_copy.clicked.connect(self.copy_image)
        btn_save.clicked.connect(self.save_image)
        btn_ocr_pix.clicked.connect(self.ocr_from_last_image_pix2tex)

        btn_region.setShortcut(QKeySequence("Ctrl+Shift+S"))
        btn_ocr_pix.setShortcut(QKeySequence("Ctrl+Shift+R"))

        # чекбоксы качества/правок
        self.cb_high_acc = QCheckBox("Высокая точность (медленнее)")
        self.cb_fix_primes = QCheckBox("Исправлять штрихи (')")
        self.cb_fix_primes.setChecked(True)
        self.cb_keep_caret_n = QCheckBox("Не трогать ^n (степени)")
        self.cb_keep_caret_n.setChecked(False)

        shot_buttons = QHBoxLayout()
        shot_buttons.addWidget(btn_region)
        shot_buttons.addStretch()
        shot_buttons.addWidget(btn_copy)
        shot_buttons.addWidget(btn_save)

        left = QVBoxLayout()
        left.addWidget(self.preview, 1)
        left.addLayout(shot_buttons)
        left.addWidget(self.cb_high_acc)
        left.addWidget(self.cb_fix_primes)
        left.addWidget(self.cb_keep_caret_n)
        left.addWidget(btn_ocr_pix)
        left_wrap = QWidget(); left_wrap.setLayout(left)

        # правая панель
        self.expr_input = QPlainTextEdit()
        self.expr_input.setPlaceholderText(
            "SymPy-выражение: sin(x)^2 + cos(x)^2,\nIntegral(x^2, (x,0,1)), Matrix([[1,x],[y,2]])…"
        )
        self.var_hint = QLineEdit(); self.var_hint.setPlaceholderText("Переменные: x y z t")
        btn_to_latex = QPushButton("Преобразовать в LaTeX")
        btn_to_latex.clicked.connect(self.convert_to_latex)
        btn_to_latex.setShortcut(QKeySequence("Ctrl+Enter"))

        self.latex_output = QPlainTextEdit(); self.latex_output.setReadOnly(True)
        self.tex_output = QPlainTextEdit(); self.tex_output.setReadOnly(True)

        self.latex_preview = QLabel("Превью LaTeX")
        self.latex_preview.setAlignment(Qt.AlignCenter)
        self.latex_preview.setFrameShape(QFrame.StyledPanel)
        self.latex_preview.setMinimumHeight(180)

        # кнопки копирования/сохранения
        btn_copy_latex = QPushButton("Копировать LaTeX")
        btn_copy_tex   = QPushButton("Копировать TeX")
        btn_save_tex   = QPushButton("Сохранить .tex")

        btn_copy_latex.clicked.connect(self.copy_latex_to_clipboard)
        btn_copy_tex.clicked.connect(self.copy_tex_to_clipboard)
        btn_save_tex.clicked.connect(self.save_tex_file)

        right = QVBoxLayout()
        right.addWidget(QLabel("Переменные (через пробел):"))
        right.addWidget(self.var_hint)
        right.addWidget(QLabel("Выражение:"))
        right.addWidget(self.expr_input, 1)
        right.addWidget(btn_to_latex)

        right.addWidget(QLabel("LaTeX код (фрагмент):"))
        right.addWidget(self.latex_output)

        hl = QHBoxLayout()
        hl.addWidget(btn_copy_latex)
        hl.addWidget(btn_copy_tex)
        hl.addWidget(btn_save_tex)
        right.addLayout(hl)

        right.addWidget(QLabel("TeX документ (.tex):"))
        right.addWidget(self.tex_output)

        right.addWidget(QLabel("Превью LaTeX:"))
        right.addWidget(self.latex_preview, 0)
        right_wrap = QWidget(); right_wrap.setLayout(right)

        splitter = QSplitter()
        splitter.addWidget(left_wrap)
        splitter.addWidget(right_wrap)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        # меню
        file_menu = self.menuBar().addMenu("&Файл")
        act_save = QAction("Сохранить скрин как PNG…", self); act_save.triggered.connect(self.save_image)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("Выход", self); act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # состояние
        self._last_image: QImage | None = None
        self._ocr_model: LatexOCR | None = None
        self._last_latex: str | None = None
        self._overlay = None

        if HAS_TORCH:
            try:
                torch.set_num_threads(max(1, os.cpu_count() // 2))
                if torch.cuda.is_available():
                    torch.set_float32_matmul_precision('medium')
            except Exception:
                pass

    # --- скрин области с автоскрытием окна ---
    def capture_region(self):
        self.hide()
        def start_overlay():
            overlay = SnipOverlay()
            overlay.captured.connect(self._on_snip_captured)
            overlay.canceled.connect(self._on_snip_canceled)
            overlay.show()
            self._overlay = overlay
        QTimer.singleShot(SNIP_HIDE_DELAY_MS, start_overlay)

    def _on_snip_captured(self, img: QImage):
        self.set_image(img)
        self.show(); self.activateWindow(); self.raise_()
        self._overlay = None

    def _on_snip_canceled(self):
        self.show(); self.activateWindow(); self.raise_()
        self._overlay = None

    # --- утилиты ---
    def set_image(self, img: QImage):
        self._last_image = img
        pm = QPixmap.fromImage(img)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_image is not None:
            pm = QPixmap.fromImage(self._last_image)
            scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(scaled)

    def copy_image(self):
        if not self._last_image: return
        QGuiApplication.clipboard().setImage(self._last_image)
        QMessageBox.information(self, "Готово", "Изображение скопировано в буфер обмена.")

    def save_image(self):
        if not self._last_image: return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить PNG", "screenshot.png", "PNG images (*.png)")
        if not path: return
        self._last_image.save(path, "PNG")
        QMessageBox.information(self, "Сохранено", f"Сохранено: {path}")

    # --- текст → LaTeX/TeX ---
    def convert_to_latex(self):
        expr_text = self.expr_input.toPlainText().strip()
        symbols_text = self.var_hint.text().strip()
        if not expr_text: return
        local_ns = {}
        if symbols_text:
            try:
                from sympy import symbols
                syms = symbols(symbols_text)
                if isinstance(syms, tuple):
                    for s in syms: local_ns[str(s)] = s
                else:
                    local_ns[str(syms)] = syms
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось объявить переменные: {e}")
                return
        try:
            expr = sympify(expr_text, locals=local_ns)
            latex_str = sympy_latex(expr)
            self._show_latex_and_tex(latex_str)
        except SympifyError as e:
            QMessageBox.warning(self, "Ошибка разбора", f"SymPy не понял выражение.\nИспользуйте синтаксис SymPy.\n\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # --- картинка → LaTeX (pix2tex) с улучшенной предобработкой ---
    def _qimage_to_pil(self, qimg: QImage) -> Image.Image:
        buf = QBuffer(); buf.open(QBuffer.ReadWrite)
        qimg.save(buf, "PNG")
        pil = Image.open(BytesIO(bytes(buf.data()))).convert("RGB")
        return pil

    def _preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        max_side = 2200 if self.cb_high_acc.isChecked() else 1600
        img = ImageOps.exif_transpose(img)
        img = ImageEnhance.Contrast(img).enhance(1.12 if self.cb_high_acc.isChecked() else 1.08)
        img = ImageEnhance.Brightness(img).enhance(1.03)
        w, h = img.size
        scale = min(1.0 * max_side / max(w, h), 2.5 if self.cb_high_acc.isChecked() else 2.0)
        if scale > 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=1)
        blur_r = 0.7 if self.cb_high_acc.isChecked() else 0.5
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
        w, h = gray.size
        radius = max(8, min(w, h) // (50 if self.cb_high_acc.isChecked() else 60))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(2, 2), (w-3, h-3)], radius=radius, fill=255)
        bg = Image.new("RGB", (w, h), "white")
        bg.paste(gray.convert("RGB"), (0, 0), mask)
        us_r = 1.4 if self.cb_high_acc.isChecked() else 1.2
        us_p = 90 if self.cb_high_acc.isChecked() else 80
        bg = bg.filter(ImageFilter.UnsharpMask(radius=us_r, percent=us_p, threshold=3))
        return bg

    def _init_ocr_model(self):
        if self._ocr_model is not None:
            return
        if HAS_TORCH and torch.cuda.is_available():
            self._ocr_model = LatexOCR(device="cuda")
        else:
            self._ocr_model = LatexOCR()  # cpu

    def ocr_from_last_image_pix2tex(self):
        if self._last_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала сделайте скриншот области.")
            return
        try:
            self._init_ocr_model()
            pil_img = self._qimage_to_pil(self._last_image)
            pil_img = self._preprocess_for_ocr(pil_img)

            if HAS_TORCH and torch.cuda.is_available():
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    latex = self._ocr_model(pil_img)
            elif HAS_TORCH:
                with torch.inference_mode():
                    latex = self._ocr_model(pil_img)
            else:
                latex = self._ocr_model(pil_img)

            latex = (latex or "").strip()

            # постобработка: штрихи
            if self.cb_fix_primes.isChecked():
                allow_caret = not self.cb_keep_caret_n.isChecked()
                latex = fix_primes_heuristic(latex, allow_caret_n=allow_caret, allow_one_as_prime=True)

            latex = collapse_superscript_chain(latex)

            if not latex:
                QMessageBox.information(self, "Пусто", "pix2tex не распознал формулу на скрине.")
                return
            self._show_latex_and_tex(latex)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка pix2tex", str(e))

    # --- вывод LaTeX / TeX / превью ---
    def _show_latex_and_tex(self, latex_str: str):
        self._last_latex = latex_str
        self.latex_output.setPlainText(latex_str)
        self.tex_output.setPlainText(build_tex_document(latex_str))
        img = render_latex_to_qimage(latex_str)
        if not img.isNull():
            pm = QPixmap.fromImage(img)
            scaled = pm.scaled(self.latex_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.latex_preview.setPixmap(scaled)
        else:
            self.latex_preview.setText(
                "Не удалось отрисовать превью (для \\begin{...} нужен установленный TeX). "
                "LaTeX/TeX сгенерированы выше."
            )

    def copy_latex_to_clipboard(self):
        text = self.latex_output.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)
            QMessageBox.information(self, "Скопировано", "LaTeX вставлен в буфер обмена.")

    def copy_tex_to_clipboard(self):
        text = self.tex_output.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)
            QMessageBox.information(self, "Скопировано", "TeX-документ вставлен в буфер обмена.")

    def save_tex_file(self):
        text = self.tex_output.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Пусто", "Пока нечего сохранять.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить TeX", "formula.tex", "TeX files (*.tex)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        QMessageBox.information(self, "Сохранено", f"Файл сохранён: {path}")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
