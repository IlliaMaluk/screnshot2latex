from __future__ import annotations
import sys, os, warnings, re, json
from dataclasses import dataclass
from enum import Enum
from io import BytesIO

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", message="A new version of Albumentations is available", module="albumentations")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", module="pydantic")

from PySide6.QtCore import Qt, QRect, QPoint, QSize, QBuffer, QTimer, Signal, QUrl
from PySide6.QtGui import QAction, QGuiApplication, QPixmap, QImage, QPainter, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSplitter, QPlainTextEdit, QLineEdit, QFrame, QMessageBox, QSizePolicy, QRubberBand,
    QCheckBox, QStackedWidget, QComboBox, QDialog, QDialogButtonBox
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from sympy import latex as sympy_latex, sympify
from sympy.core.sympify import SympifyError

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PIL import Image, ImageFilter, ImageOps
from pix2tex.cli import LatexOCR

PREVIEW_DPI = 140
SNIP_HIDE_DELAY_MS = 200

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

LATEX_ENV_RE = re.compile(r"\\begin\{.*?\}|\\\[[\s\S]*?\\\]|\\\(.+?\\\)", re.DOTALL)


def _strip_math_wrappers(s: str) -> str:
    body = s.strip()
    if body.startswith('$$') and body.endswith('$$'):
        return body[2:-2].strip()
    if body.startswith(r'\[') and body.endswith(r'\]'):
        return body[2:-2].strip()
    if body.startswith(r'\(') and body.endswith(r'\)'):
        return body[2:-2].strip()
    return body


class LatexRenderer:
    def __init__(self, dpi: int = PREVIEW_DPI):
        self.dpi = dpi

    def math_fragment(self, latex_body: str) -> str:
        content = _strip_math_wrappers(latex_body)
        if LATEX_ENV_RE.search(content) or r'\begin{' in content:
            return content
        return f"\\[\n{content}\n\\]"

    def document(self, latex_body: str) -> str:
        math_block = self.math_fragment(latex_body)
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

    def render_qimage(self, latex_body: str) -> QImage:
        content = _strip_math_wrappers(latex_body)
        wrap = not LATEX_ENV_RE.search(content)
        expr = f"${content}$" if wrap else content
        fig = plt.figure(figsize=(0.01, 0.01), dpi=self.dpi)
        fig.patch.set_alpha(0.0)
        text = fig.text(0.5, 0.5, expr, ha="center", va="center")
        canvas = FigureCanvas(fig)
        canvas.draw()
        bbox = text.get_window_extent(renderer=canvas.get_renderer()).expanded(1.2, 1.4)
        width, height = int(bbox.width), int(bbox.height)
        fig2 = plt.figure(figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi)
        fig2.patch.set_alpha(0.0)
        fig2.text(0, 0, expr, ha="left", va="baseline")
        canvas2 = FigureCanvas(fig2)
        canvas2.draw()
        buf = BytesIO()
        fig2.savefig(buf, format="png", dpi=self.dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        plt.close(fig2)
        return QImage.fromData(buf.getvalue(), "PNG")

    def katex_html(self, latex_body: str) -> str:
        body = _strip_math_wrappers(latex_body)
        payload = json.dumps(body)
        return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\"/>
<link rel=\"preconnect\" href=\"https://cdn.jsdelivr.net\"/>
<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css\">
<script src=\"https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js\"></script>
<style>
html,body,#root{{height:100%;margin:0;background:#1f1f1f;color:#eaeaea}}
#wrap{{display:flex;align-items:center;justify-content:center;height:100%;}}
.katex-display{{font-size:1.6rem;}}
</style>
</head>
<body>
<div id=\"wrap\"><div id=\"root\"></div></div>
<script>
(function(){{
  function renderNow(){{
    try{{
      var el = document.getElementById('root');
      var TEX = {payload};
      katex.render(TEX, el, {{displayMode:true, throwOnError:false}});
    }}catch(e){{
      document.getElementById('root').textContent = {payload};
    }}
  }}
  if (window.katex) renderNow(); else window.addEventListener('load', renderNow);
}})();
</script>
</body>
</html>"""


class PreprocessMode(Enum):
    STANDARD = "Стандартная"
    HIGH_ACC = "Мягкая Hi-Acc"
    RAW = "Без предобработки"
    ADAPTIVE = "Адаптивная бинаризация"

    @classmethod
    def from_label(cls, label: str) -> "PreprocessMode":
        for mode in cls:
            if mode.value == label:
                return mode
        return cls.STANDARD


def _resize_for_quality(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    scale = min(max_side / max(width, height), 2.5)
    if scale <= 1:
        return img
    return img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)


class ImageProcessor:
    def prepare(self, image: Image.Image, mode: PreprocessMode, boost_quality: bool) -> Image.Image:
        base = ImageOps.exif_transpose(image.convert("RGB"))
        if mode is PreprocessMode.RAW:
            return base
        max_side = 2600 if mode is PreprocessMode.HIGH_ACC else 2200
        if mode is PreprocessMode.ADAPTIVE:
            max_side = 2400
        if boost_quality:
            max_side += 240
        scaled = _resize_for_quality(base, max_side)
        if mode is PreprocessMode.HIGH_ACC:
            gray = scaled.convert("L")
            sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.4, percent=90, threshold=2))
            return Image.merge("RGB", (sharp, sharp, sharp))
        if mode is PreprocessMode.STANDARD:
            gray = scaled.convert("L")
            norm = ImageOps.autocontrast(gray, cutoff=1)
            blur = norm.filter(ImageFilter.GaussianBlur(radius=0.6))
            sharp = blur.filter(ImageFilter.UnsharpMask(radius=1.1, percent=78, threshold=4))
            return Image.merge("RGB", (sharp, sharp, sharp))
        gray = scaled.convert("L")
        norm = ImageOps.autocontrast(gray, cutoff=1)
        blur = norm.filter(ImageFilter.GaussianBlur(radius=1.2))
        mixed = Image.blend(norm, blur, alpha=0.55)
        bw = mixed.point(lambda p: 255 if p > 185 else 0, mode='1').convert("L")
        filt = bw.filter(ImageFilter.MedianFilter(size=3))
        return Image.merge("RGB", (filt, filt, filt))


UNICODE_PRIMES = [
    ("\u2032", r"^{\prime}"),
    ("\u2033", r"^{\prime\prime}"),
    ("\u2034", r"^{\prime\prime\prime}"),
    ("\u2057", r"^{\prime\prime\prime\prime}"),
]


@dataclass
class CleanOptions:
    normalize_primes: bool
    keep_caret_n: bool
    confusion_table: bool


class LatexCleaner:
    def __init__(self):
        self._prime_after_token = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])('{1,4})")
        self._prime_glue = [
            re.compile(r"(\^\{\s*\\prime\s*\})\s*'"),
            re.compile(r"'\s*(\^\{\s*\\prime\s*\})"),
            re.compile(r"\^\{\s*\\prime\s*\}\s*\{\s*\\prime\s*\}"),
            re.compile(r"\^\{\s*\\prime\s*\}\s*\\prime"),
        ]
        self._prime_eta_wrap = re.compile(r"\^\{\s*(?:\\,|\{\\,\})?\s*\\prime\s*\(\s*1\s*\)\s*\}", re.IGNORECASE)
        self._fake_double_prime = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])\s*\^\{\s*\{?\s*m\s*\}?\s*\(\s*1\s*\)\s*\}")
        self._fake_one = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])\s*\^\{\s*[1l]\s*\}(\s*(?=[\(\[]))?")
        self._caret_map = {
            re.compile(r"\^\{?\s*n\s*\}?"): r"^{\prime}",
            re.compile(r"\^\{?\s*nn\s*\}?"): r"^{\prime\prime}",
            re.compile(r"\^\{?\s*nnn\s*\}?"): r"^{\prime\prime\prime}",
        }
        self._gamma_swap = re.compile(r'(?<![\\a-zA-Z])\\gamma(?![a-zA-Z])')
        self._theta_swap = re.compile(r'\\Theta(?=\s*(?:[\(\[]|\\bigl|\\left))')
        self._arrow_swap = re.compile(r'\\longrightarrow')
        self._prime_super = re.compile(r'(\^\{\s*\\prime\s*\})(?:\s*\{\}\s*)?(\^\{[^}]+\})')
        self._eta_exp = re.compile(r'\^\{\s*\\eta\s*(\([^{}]*\))\s*\}')
        self._eta_sub = re.compile(r'_\{\s*\\eta\s*(\([^{}]*\))\s*\}')
        self._eta_inline = re.compile(r'\\eta(\s*\(\s*\d+\s*\))')
        self._cal_macro = re.compile(r'\\cal\s*\{')
        self._cal_token = re.compile(r'\\(?:math)?cal\s*\{([A-Z])\}')
        self._brace_r = re.compile(r'\{(r(?:_\{[^{}]*\}|_[^{}\s]+)?(?:\^\{[^}]+\})?)\}')
        self._double_hyphen = re.compile(r'\s*--\s*')
        self._back_plus = re.compile(r'\\\s*\+')
        self._back_minus = re.compile(r'\\\s*-')
        self._back_hyphen = re.compile(r'\\\s*--\s*')
        self._bigl = re.compile(r'\\bigl\(')
        self._bigr = re.compile(r'\\bigr\)')

    def clean(self, latex: str, options: CleanOptions) -> str:
        text = latex.strip()
        if not text:
            return text
        if options.normalize_primes:
            text = self._normalize_primes(text, options.keep_caret_n)
        if options.confusion_table:
            text = self._resolve_confusions(text)
        return text

    def _normalize_primes(self, text: str, keep_caret_n: bool) -> str:
        out = text.replace(r"\,", "")
        for uni, repl in UNICODE_PRIMES:
            out = out.replace(uni, repl)
        def repl(m: re.Match) -> str:
            base = m.group("base")
            primes = len(m.group(0)) - len(base)
            return f"{base}^{{" + r"\prime" * primes + "}}"
        out = self._prime_after_token.sub(repl, out)
        for pattern in self._prime_glue:
            out = pattern.sub(lambda _m: r"^{\prime\prime}", out)
        out = self._prime_eta_wrap.sub(lambda _m: r"^{\prime\prime}", out)
        out = self._fake_double_prime.sub(lambda m: f"{m.group('base')}^{{\\prime\\prime}}(1)", out)
        out = self._fake_one.sub(lambda m: f"{m.group('base')}^{{\\prime}}", out)
        if not keep_caret_n:
            for rx, repl in self._caret_map.items():
                out = rx.sub(repl, out)
        return out

    def _resolve_confusions(self, text: str) -> str:
        out = text
        out = self._gamma_swap.sub('r', out)
        out = self._theta_swap.sub('6', out)
        out = self._arrow_swap.sub('=', out)
        out = self._prime_super.sub(lambda m: m.group(2), out)
        out = self._eta_exp.sub(lambda m: f"^{{n{m.group(1)}}}", out)
        out = self._eta_sub.sub(lambda m: f"_{{n{m.group(1)}}}", out)
        out = self._eta_inline.sub(lambda m: f"n{m.group(1)}", out)
        out = self._cal_macro.sub(r'\\mathcal{', out)
        out = self._cal_token.sub(lambda m: m.group(1).lower(), out)
        out = self._brace_r.sub(lambda m: m.group(1), out)
        out = self._back_hyphen.sub('-', out)
        out = self._double_hyphen.sub('-', out)
        out = self._back_plus.sub('+', out)
        out = self._back_minus.sub('-', out)
        out = self._bigl.sub('(', out)
        out = self._bigr.sub(')', out)
        return out


@dataclass
class OCRRequest:
    image: QImage
    preprocess_mode: PreprocessMode
    boost_quality: bool
    clean: CleanOptions


def _qimage_to_pil(image: QImage) -> Image.Image:
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    pil = Image.open(BytesIO(bytes(buffer.data()))).convert("RGB")
    return pil


class Pix2TexController:
    def __init__(self, processor: ImageProcessor, cleaner: LatexCleaner):
        self.processor = processor
        self.cleaner = cleaner
        self._model: LatexOCR | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if HAS_TORCH and torch.cuda.is_available():
            self._model = LatexOCR(device="cuda")
        else:
            self._model = LatexOCR()

    def infer(self, request: OCRRequest) -> str:
        self._ensure_model()
        pil = _qimage_to_pil(request.image)
        prepared = self.processor.prepare(pil, request.preprocess_mode, request.boost_quality)
        if HAS_TORCH and torch.cuda.is_available():
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                raw = self._model(prepared)
        elif HAS_TORCH:
            with torch.inference_mode():
                raw = self._model(prepared)
        else:
            raw = self._model(prepared)
        cleaned = self.cleaner.clean((raw or "").strip(), request.clean)
        return cleaned

    def preview_input(self, request: OCRRequest) -> QImage:
        pil = _qimage_to_pil(request.image)
        prepared = self.processor.prepare(pil, request.preprocess_mode, request.boost_quality)
        buf = BytesIO()
        prepared.save(buf, format="PNG")
        return QImage.fromData(buf.getvalue(), "PNG")


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
            geometry = screen.geometry()
            pix = screen.grabWindow(0)
            painter.drawPixmap(geometry.topLeft() - self.virtual_geo.topLeft(), pix)
        painter.end()
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    def _cancel(self):
        self.rubber.hide()
        self.canceled.emit()
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._cancel()
            return
        self.origin = event.globalPosition().toPoint()
        self.rubber.setGeometry(QRect(self.mapFromGlobal(self.origin), QSize()))
        self.rubber.show()
        self.dragging = True

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        current = event.globalPosition().toPoint()
        rect = QRect(self.mapFromGlobal(self.origin), self.mapFromGlobal(current)).normalized()
        self.rubber.setGeometry(rect)
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return
        self.dragging = False
        rect_local = self.rubber.geometry()
        self.rubber.hide()
        if rect_local.width() < 3 or rect_local.height() < 3:
            self._cancel()
            return
        rect_global = QRect(rect_local)
        rect_global.translate(self.virtual_geo.topLeft())
        self._capture_rect(rect_global)
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._cancel()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.backing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if self.rubber.isVisible():
            selection = self.rubber.geometry()
            sub = self.backing.copy(QRect(selection.topLeft(), selection.size()))
            painter.drawPixmap(selection.topLeft(), sub)
            pen = painter.pen()
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(selection)

    def closeEvent(self, event):
        if self.dragging:
            self.canceled.emit()
        super().closeEvent(event)

    def _capture_rect(self, rect_global: QRect):
        if rect_global.isEmpty():
            self._cancel()
            return
        local = QRect(rect_global)
        local.translate(-self.virtual_geo.topLeft())
        cropped = self.backing.copy(local)
        self.captured.emit(cropped.toImage())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen & LaTeX Helper")
        self.setMinimumSize(1220, 720)
        self.renderer = LatexRenderer()
        self.processor = ImageProcessor()
        self.cleaner = LatexCleaner()
        self.ocr_controller = Pix2TexController(self.processor, self.cleaner)
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
        self.cb_high_acc = QCheckBox("Доп. улучшение качества")
        self.cb_fix_primes = QCheckBox("Исправлять штрихи (')")
        self.cb_fix_primes.setChecked(True)
        self.cb_keep_caret_n = QCheckBox("Не трогать ^n (степени)")
        self.cb_keep_caret_n.setChecked(False)
        self.cb_anti_confuse = QCheckBox("Анти-подмена символов (γ→r, Θ→6, η→n, →=)")
        self.cb_anti_confuse.setChecked(False)
        self.pre_combo = QComboBox()
        self.pre_combo.addItems([mode.value for mode in PreprocessMode])
        self.pre_combo.setCurrentText(PreprocessMode.HIGH_ACC.value)
        btn_show_ocr_in = QPushButton("Показать вход OCR")
        btn_show_ocr_in.clicked.connect(self._show_ocr_input)
        shot_buttons = QHBoxLayout()
        shot_buttons.addWidget(btn_region)
        shot_buttons.addStretch()
        shot_buttons.addWidget(btn_copy)
        shot_buttons.addWidget(btn_save)
        left = QVBoxLayout()
        left.addWidget(self.preview, 1)
        left.addLayout(shot_buttons)
        left.addWidget(self.pre_combo)
        left.addWidget(self.cb_high_acc)
        left.addWidget(self.cb_fix_primes)
        left.addWidget(self.cb_keep_caret_n)
        left.addWidget(self.cb_anti_confuse)
        left.addWidget(btn_show_ocr_in)
        left.addWidget(btn_ocr_pix)
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        self.expr_input = QPlainTextEdit()
        self.expr_input.setPlaceholderText("SymPy-выражение: sin(x)^2 + cos(x)^2,\nIntegral(x^2, (x,0,1)), Matrix([[1,x],[y,2]])…")
        self.var_hint = QLineEdit()
        self.var_hint.setPlaceholderText("Переменные: x y z t")
        btn_to_latex = QPushButton("Преобразовать в LaTeX")
        btn_to_latex.clicked.connect(self.convert_to_latex)
        btn_to_latex.setShortcut(QKeySequence("Ctrl+Enter"))
        self.latex_output = QPlainTextEdit()
        self.latex_output.setReadOnly(True)
        self.tex_output = QPlainTextEdit()
        self.tex_output.setReadOnly(True)
        self.preview_stack = QStackedWidget()
        self.latex_preview_img = QLabel("Превью LaTeX")
        self.latex_preview_img.setAlignment(Qt.AlignCenter)
        self.latex_preview_img.setFrameShape(QFrame.StyledPanel)
        self.latex_preview_img.setMinimumHeight(180)
        self.latex_preview_web = QWebEngineView()
        self.latex_preview_web.setMinimumHeight(180)
        self.preview_stack.addWidget(self.latex_preview_img)
        self.preview_stack.addWidget(self.latex_preview_web)
        self.preview_stack.setCurrentIndex(0)
        btn_copy_latex = QPushButton("Копировать LaTeX")
        btn_copy_tex = QPushButton("Копировать TeX")
        btn_save_tex = QPushButton("Сохранить .tex")
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
        buttons_line = QHBoxLayout()
        buttons_line.addWidget(btn_copy_latex)
        buttons_line.addWidget(btn_copy_tex)
        buttons_line.addWidget(btn_save_tex)
        right.addLayout(buttons_line)
        right.addWidget(QLabel("TeX документ (.tex):"))
        right.addWidget(self.tex_output)
        right.addWidget(QLabel("Превью LaTeX:"))
        right.addWidget(self.preview_stack, 0)
        right_wrap = QWidget()
        right_wrap.setLayout(right)
        splitter = QSplitter()
        splitter.addWidget(left_wrap)
        splitter.addWidget(right_wrap)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        file_menu = self.menuBar().addMenu("&Файл")
        act_save = QAction("Сохранить скрин как PNG…", self)
        act_save.triggered.connect(self.save_image)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("Выход", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        self._last_image: QImage | None = None
        self._last_latex: str | None = None
        self._overlay: SnipOverlay | None = None
        if HAS_TORCH:
            try:
                torch.set_num_threads(max(1, os.cpu_count() // 2))
                if torch.cuda.is_available():
                    torch.set_float32_matmul_precision('medium')
            except Exception:
                pass

    def capture_region(self):
        self.hide()
        def start_overlay():
            overlay = SnipOverlay()
            overlay.captured.connect(self._on_snip_captured)
            overlay.canceled.connect(self._on_snip_canceled)
            overlay.show()
            self._overlay = overlay
        QTimer.singleShot(SNIP_HIDE_DELAY_MS, start_overlay)

    def _on_snip_captured(self, image: QImage):
        self.set_image(image)
        self.show()
        self.activateWindow()
        self.raise_()
        self._overlay = None

    def _on_snip_canceled(self):
        self.show()
        self.activateWindow()
        self.raise_()
        self._overlay = None

    def set_image(self, image: QImage):
        self._last_image = image
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_image is not None:
            pixmap = QPixmap.fromImage(self._last_image)
            scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(scaled)

    def copy_image(self):
        if not self._last_image:
            return
        QGuiApplication.clipboard().setImage(self._last_image)
        QMessageBox.information(self, "Готово", "Изображение скопировано в буфер обмена.")

    def save_image(self):
        if not self._last_image:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить PNG", "screenshot.png", "PNG images (*.png)")
        if not path:
            return
        self._last_image.save(path, "PNG")
        QMessageBox.information(self, "Сохранено", f"Сохранено: {path}")

    def convert_to_latex(self):
        expr_text = self.expr_input.toPlainText().strip()
        symbols_text = self.var_hint.text().strip()
        if not expr_text:
            return
        local_ns: dict[str, object] = {}
        if symbols_text:
            try:
                from sympy import symbols
                declared = symbols(symbols_text)
                if isinstance(declared, tuple):
                    for sym in declared:
                        local_ns[str(sym)] = sym
                else:
                    local_ns[str(declared)] = declared
            except Exception as exc:
                QMessageBox.warning(self, "Ошибка", f"Не удалось объявить переменные: {exc}")
                return
        try:
            expr = sympify(expr_text, locals=local_ns)
            latex_str = sympy_latex(expr)
            self._show_latex_and_tex(latex_str)
        except SympifyError as exc:
            QMessageBox.warning(self, "Ошибка разбора", f"SymPy не понял выражение.\nИспользуйте синтаксис SymPy.\n\n{exc}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def ocr_from_last_image_pix2tex(self):
        if self._last_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала сделайте скриншот области.")
            return
        request = OCRRequest(
            image=self._last_image,
            preprocess_mode=PreprocessMode.from_label(self.pre_combo.currentText()),
            boost_quality=self.cb_high_acc.isChecked(),
            clean=CleanOptions(
                normalize_primes=self.cb_fix_primes.isChecked(),
                keep_caret_n=self.cb_keep_caret_n.isChecked(),
                confusion_table=self.cb_anti_confuse.isChecked(),
            ),
        )
        try:
            latex = self.ocr_controller.infer(request)
            if not latex:
                QMessageBox.information(self, "Пусто", "pix2tex не распознал формулу на скрине.")
                return
            self._show_latex_and_tex(latex)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка pix2tex", str(exc))

    def _show_ocr_input(self):
        if self._last_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала сделайте скриншот области.")
            return
        request = OCRRequest(
            image=self._last_image,
            preprocess_mode=PreprocessMode.from_label(self.pre_combo.currentText()),
            boost_quality=self.cb_high_acc.isChecked(),
            clean=CleanOptions(False, True, False),
        )
        preview = self.ocr_controller.preview_input(request)
        dialog = QDialog(self)
        dialog.setWindowTitle("Вход в pix2tex")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap.fromImage(preview)
        label.setPixmap(pixmap.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(label)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.resize(940, 680)
        dialog.exec()

    def _show_latex_and_tex(self, latex_str: str):
        self._last_latex = latex_str
        self.latex_output.setPlainText(latex_str)
        self.tex_output.setPlainText(self.renderer.document(latex_str))
        image = self.renderer.render_qimage(latex_str)
        if not image.isNull() and not LATEX_ENV_RE.search(latex_str):
            pixmap = QPixmap.fromImage(image)
            scaled = pixmap.scaled(self.latex_preview_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.latex_preview_img.setPixmap(scaled)
            self.preview_stack.setCurrentIndex(0)
            return
        html = self.renderer.katex_html(latex_str)
        self.latex_preview_web.setHtml(html, QUrl("https://cdn.jsdelivr.net/"))
        self.preview_stack.setCurrentIndex(1)

    def copy_latex_to_clipboard(self):
        text = self.latex_output.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)
            QMessageBox.information(self, "Скопировано", "LaTeX скопирован.")

    def copy_tex_to_clipboard(self):
        text = self.tex_output.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)
            QMessageBox.information(self, "Скопировано", "TeX скопирован.")

    def save_tex_file(self):
        text = self.tex_output.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Пусто", "Пока нечего сохранять.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить TeX", "formula.tex", "TeX files (*.tex)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        QMessageBox.information(self, "Сохранено", f"Файл сохранён: {path}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
