from __future__ import annotations
import sys, os, warnings, re, json
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

LATEX_ENV_RE = re.compile(r'\\begin\{.*?\}|\\\[[\s\S]*?\\\]|\\\(.+?\\\)', re.DOTALL)


def sanitize_latex(lx: str) -> str:
    s = lx.strip()
    if s.startswith('$$') and s.endswith('$$'): s = s[2:-2].strip()
    if s.startswith(r'\[') and s.endswith(r'\]'): s = s[2:-2].strip()
    if s.startswith(r'\(') and s.endswith(r'\)'): s = s[2:-2].strip()
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


def render_latex_to_qimage_fast(latex_str: str, dpi: int = PREVIEW_DPI) -> QImage:
    try:
        return _render_mathtext(latex_str, dpi)
    except Exception:
        return QImage()


                                                                     
def make_katex_html(latex_str: str) -> str:
    body = sanitize_latex(latex_str)
    tex_json = json.dumps(body)                    

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="preconnect" href="https://cdn.jsdelivr.net"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<style>
  html,body,#root{{height:100%;margin:0;background:#1f1f1f;color:#eaeaea}}
  #wrap{{display:flex;align-items:center;justify-content:center;height:100%;}}
  .katex-display{{font-size:1.6rem;}}
</style>
</head>
<body>
<div id="wrap"><div id="root"></div></div>
<script>
(function(){{
  function renderNow(){{
    try{{
      var el = document.getElementById('root');
      var TEX = {tex_json};
      katex.render(TEX, el, {{displayMode:true, throwOnError:false}});
    }}catch(e){{
      document.getElementById('root').textContent = {tex_json};
    }}
  }}
  if (window.katex) renderNow(); else window.addEventListener('load', renderNow);
}})();
</script>
</body>
</html>"""


                                                                           
UNICODE_PRIMES = [
    ("\u2032", r"^{\prime}"),
    ("\u2033", r"^{\prime\prime}"),
    ("\u2034", r"^{\prime\prime\prime}"),
    ("\u2057", r"^{\prime\prime\prime\prime}"),
]
PRIME_AFTER_TOKEN = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])('{1,4})")
PRIME_GLUE_PATTERNS = [
    re.compile(r"(\^\{\s*\\prime\s*\})\s*'"),
    re.compile(r"'\s*(\^\{\s*\\prime\s*\})"),
    re.compile(r"\^\{\s*\\prime\s*\}\s*\{\s*\\prime\s*\}"),
    re.compile(r"\^\{\s*\\prime\s*\}\s*\\prime"),
]
PRIME_WITH_PAREN_IN_EXP = re.compile(r"\^\{\s*(?:\\,|\{\\,\})?\s*\\prime\s*\(\s*1\s*\)\s*\}", re.IGNORECASE)
FAKE_M_AS_DOUBLE_PRIME = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])\s*\^\{\s*\{?\s*m\s*\}?\s*\(\s*1\s*\)\s*\}")
FAKE_ONE_AS_PRIME = re.compile(r"(?P<base>[A-Za-z0-9\\\}\)])\s*\^\{\s*[1l]\s*\}(\s*(?=[\(\[]))?")


def fix_primes_heuristic(s: str, allow_caret_n: bool = True, allow_one_as_prime: bool = True) -> str:
    out = s.replace(r"\,", "")
    for uni, repl in UNICODE_PRIMES:
        out = out.replace(uni, repl)

    def _rep(m: re.Match) -> str:
        base = m.group("base")
        primes = len(m.group(0)) - len(base)
        return f"{base}^{{" + r"\prime" * primes + "}}"
    out = PRIME_AFTER_TOKEN.sub(_rep, out)

    for rx in PRIME_GLUE_PATTERNS:
        out = rx.sub(lambda _m: r"^{\prime\prime}", out)

    out = PRIME_WITH_PAREN_IN_EXP.sub(lambda _m: r"^{\prime\prime}", out)
    out = FAKE_M_AS_DOUBLE_PRIME.sub(lambda m: f"{m.group('base')}^{{\\prime\\prime}}(1)", out)

    if allow_caret_n:
        out = re.sub(r"\^\{?\s*n\s*\}?",   lambda _m: r"^{\prime}",             out)
        out = re.sub(r"\^\{?\s*nn\s*\}?",  lambda _m: r"^{\prime\prime}",       out)
        out = re.sub(r"\^\{?\s*nnn\s*\}?", lambda _m: r"^{\prime\prime\prime}", out)

    if allow_one_as_prime:
        out = FAKE_ONE_AS_PRIME.sub(lambda m: f"{m.group('base')}^{{\\prime}}", out)
    return out


                                                      

_greek_r = re.compile(r'(?<![\\a-zA-Z])\\gamma(?![a-zA-Z])')
_theta_6 = re.compile(r'\\Theta(?=\s*(?:[\(\[]|\\bigl|\\left))')
_arrow_to_equal = re.compile(r'\\longrightarrow')
_prime_then_super = re.compile(r'(\^\{\s*\\prime\s*\})(?:\s*\{\}\s*)?(\^\{[^}]+\})')
_eta_sup_paren = re.compile(r'\^\{\s*\\eta\s*(\([^{}]*\))\s*\}')
_eta_sub_paren = re.compile(r'_\{\s*\\eta\s*(\([^{}]*\))\s*\}')
_eta_inline_paren = re.compile(r'\\eta(\s*\(\s*\d+\s*\))')
_cal_macro = re.compile(r'\\cal\s*\{')
_cal_i_token = re.compile(r'\\(?:math)?cal\s*\{I\}')
_cal_z_token = re.compile(r'\\(?:math)?cal\s*\{Z\}')
_brace_wrap_r = re.compile(r'\{(r(?:_\{[^{}]*\}|_[^{}\s]+)?(?:\^\{[^}]+\})?)\}')
_double_hyphen = re.compile(r'\s*--\s*')
_backslashed_double_hyphen = re.compile(r'\\\s*--\s*')
_backslashed_plus = re.compile(r'\\\s*\+')
_backslashed_minus = re.compile(r'\\\s*-')
_bigl_paren = re.compile(r'\\bigl\(')
_bigr_paren = re.compile(r'\\bigr\)')


def _replace_cal_token(match: re.Match) -> str:
    return 'r'



def fix_greek_confusions(s: str) -> str:
    out = s
    out = _greek_r.sub('r', out)
    out = _theta_6.sub('6', out)
    out = _arrow_to_equal.sub('=', out)
    out = _prime_then_super.sub(lambda m: m.group(2), out)
    out = _eta_sup_paren.sub(lambda m: f"^{{n{m.group(1)}}}", out)
    out = _eta_sub_paren.sub(lambda m: f"_{{n{m.group(1)}}}", out)
    out = _eta_inline_paren.sub(lambda m: f"n{m.group(1)}", out)
    out = _cal_macro.sub(r'\\mathcal{', out)
    out = _cal_i_token.sub(_replace_cal_token, out)
    out = _cal_z_token.sub(_replace_cal_token, out)
    out = _brace_wrap_r.sub(lambda m: m.group(1), out)
    out = _backslashed_double_hyphen.sub('-', out)
    out = _double_hyphen.sub('-', out)
    out = _backslashed_plus.sub('+', out)
    out = _backslashed_minus.sub('-', out)
    out = _bigl_paren.sub('(', out)
    out = _bigr_paren.sub(')', out)
    return out


                                   
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
        p = QPainter(self.backing)
        for screen in QGuiApplication.screens():
            g = screen.geometry()
            pix = screen.grabWindow(0)
            p.drawPixmap(g.topLeft() - self.virtual_geo.topLeft(), pix)
        p.end()

        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def showEvent(self, e):
        super().showEvent(e)
        self.activateWindow(); self.raise_(); self.setFocus()

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
        if not self.dragging: return
        cur = e.globalPosition().toPoint()
        rect = QRect(self.mapFromGlobal(self.origin), self.mapFromGlobal(cur)).normalized()
        self.rubber.setGeometry(rect)
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.dragging: return
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
            pen = p.pen(); pen.setWidth(2); p.setPen(pen); p.drawRect(sel)

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


                      
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen & LaTeX Helper")
        self.setMinimumSize(1220, 720)

                                      
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

        self.cb_high_acc = QCheckBox("Высокая точность (медленнее)")                       
        self.cb_fix_primes = QCheckBox("Исправлять штрихи (')")
        self.cb_fix_primes.setChecked(True)
        self.cb_keep_caret_n = QCheckBox("Не трогать ^n (степени)")
        self.cb_keep_caret_n.setChecked(False)
        self.cb_anti_confuse = QCheckBox("Анти-подмена символов (γ→r, Θ→6, η→n, →=)")
        self.cb_anti_confuse.setChecked(False)

                                                  
        self.pre_combo = QComboBox()
        self.pre_combo.addItems([
            "Стандартная",
            "Мягкая Hi-Acc",
            "Без предобработки",
            "Адаптивная бинаризация"
        ])
        self.pre_combo.setCurrentIndex(1)                              

        btn_show_ocr_in = QPushButton("Показать вход OCR")
        btn_show_ocr_in.clicked.connect(self._show_ocr_input)

        shot_buttons = QHBoxLayout()
        shot_buttons.addWidget(btn_region); shot_buttons.addStretch()
        shot_buttons.addWidget(btn_copy); shot_buttons.addWidget(btn_save)

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
        left_wrap = QWidget(); left_wrap.setLayout(left)

                                                    
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
        btn_copy_tex   = QPushButton("Копировать TeX")
        btn_save_tex   = QPushButton("Сохранить .tex")
        btn_copy_latex.clicked.connect(self.copy_latex_to_clipboard)
        btn_copy_tex.clicked.connect(self.copy_tex_to_clipboard)
        btn_save_tex.clicked.connect(self.save_tex_file)

        right = QVBoxLayout()
        right.addWidget(QLabel("Переменные (через пробел):")); right.addWidget(self.var_hint)
        right.addWidget(QLabel("Выражение:")); right.addWidget(self.expr_input, 1)
        right.addWidget(btn_to_latex)
        right.addWidget(QLabel("LaTeX код (фрагмент):")); right.addWidget(self.latex_output)
        hl = QHBoxLayout(); hl.addWidget(btn_copy_latex); hl.addWidget(btn_copy_tex); hl.addWidget(btn_save_tex)
        right.addLayout(hl)
        right.addWidget(QLabel("TeX документ (.tex):")); right.addWidget(self.tex_output)
        right.addWidget(QLabel("Превью LaTeX:")); right.addWidget(self.preview_stack, 0)
        right_wrap = QWidget(); right_wrap.setLayout(right)

        splitter = QSplitter(); splitter.addWidget(left_wrap); splitter.addWidget(right_wrap)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

              
        file_menu = self.menuBar().addMenu("&Файл")
        act_save = QAction("Сохранить скрин как PNG…", self); act_save.triggered.connect(self.save_image)
        file_menu.addAction(act_save); file_menu.addSeparator()
        act_quit = QAction("Выход", self); act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

                   
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

                                               
    def capture_region(self):
        self.hide()
        def start_overlay():
            overlay = SnipOverlay()
            overlay.captured.connect(self._on_snip_captured)
            overlay.canceled.connect(self._on_snip_canceled)
            overlay.show(); self._overlay = overlay
        QTimer.singleShot(SNIP_HIDE_DELAY_MS, start_overlay)

    def _on_snip_captured(self, img: QImage):
        self.set_image(img); self.show(); self.activateWindow(); self.raise_(); self._overlay = None

    def _on_snip_canceled(self):
        self.show(); self.activateWindow(); self.raise_(); self._overlay = None

                     
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

                                        
    def _qimage_to_pil(self, qimg: QImage) -> Image.Image:
        buf = QBuffer(); buf.open(QBuffer.ReadWrite)
        qimg.save(buf, "PNG")
        pil = Image.open(BytesIO(bytes(buf.data()))).convert("RGB")
        return pil

    def _preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        """4 режима: стандарт, мягкая Hi-Acc, без предобработки, адаптивная бинаризация."""
        mode = self.pre_combo.currentText()

                       
        img = ImageOps.exif_transpose(img)

        if mode == "Без предобработки":
            return img.convert("RGB")

                 
        max_side = 2600 if mode == "Мягкая Hi-Acc" else (2400 if mode == "Адаптивная бинаризация" else 2200)
        w, h = img.size
        scale = min(max_side / max(w, h), 2.5)
        if scale > 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        if mode == "Мягкая Hi-Acc":
            gray = img.convert("L")
                                                     
            sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.3, percent=85, threshold=3))
            return Image.merge("RGB", (sharp, sharp, sharp))

        if mode == "Стандартная":
            gray = img.convert("L")
            gray = ImageOps.autocontrast(gray, cutoff=1)
            gray = gray.filter(ImageFilter.GaussianBlur(radius=0.5))
            sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
            return Image.merge("RGB", (sharp, sharp, sharp))

                                                          
        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=1)
        blur = gray.filter(ImageFilter.GaussianBlur(radius=1.2))
        arr = Image.blend(gray, blur, alpha=0.5)
        bw = arr.point(lambda p: 255 if p > 180 else 0, mode='1').convert("L")
        bw = bw.filter(ImageFilter.MedianFilter(size=3))
        return Image.merge("RGB", (bw, bw, bw))

    def _init_ocr_model(self):
        if self._ocr_model is not None: return
        if HAS_TORCH and torch.cuda.is_available():
            self._ocr_model = LatexOCR(device="cuda")
        else:
            self._ocr_model = LatexOCR()

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

            if self.cb_fix_primes.isChecked():
                allow_caret = not self.cb_keep_caret_n.isChecked()
                latex = fix_primes_heuristic(latex, allow_caret_n=allow_caret, allow_one_as_prime=True)

            if self.cb_anti_confuse.isChecked():
                latex = fix_greek_confusions(latex)

            if not latex:
                QMessageBox.information(self, "Пусто", "pix2tex не распознал формулу на скрине.")
                return

            self._show_latex_and_tex(latex)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка pix2tex", str(e))

                                    
    def _show_ocr_input(self):
        if self._last_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала сделайте скриншот области.")
            return
        pil = self._qimage_to_pil(self._last_image)
        pre = self._preprocess_for_ocr(pil)
        buf = BytesIO(); pre.save(buf, format="PNG")
        qimg = QImage.fromData(buf.getvalue(), "PNG")

        dlg = QDialog(self); dlg.setWindowTitle("Вход в pix2tex")
        lay = QVBoxLayout(dlg)
        lab = QLabel(); lab.setAlignment(Qt.AlignCenter)
        pm = QPixmap.fromImage(qimg)
        lab.setPixmap(pm.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(lab)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.resize(940, 680)
        dlg.exec()

                                        
    def _show_latex_and_tex(self, latex_str: str):
        self._last_latex = latex_str
        self.latex_output.setPlainText(latex_str)
        self.tex_output.setPlainText(build_tex_document(latex_str))

        qimg = render_latex_to_qimage_fast(latex_str)
        if not qimg.isNull() and not LATEX_ENV_RE.search(latex_str):
            pm = QPixmap.fromImage(qimg)
            scaled = pm.scaled(self.latex_preview_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.latex_preview_img.setPixmap(scaled)
            self.preview_stack.setCurrentIndex(0)
            return

        html = make_katex_html(latex_str)
                                                                            
        self.latex_preview_web.setHtml(html, QUrl("https://cdn.jsdelivr.net/"))
        self.preview_stack.setCurrentIndex(1)

    def copy_latex_to_clipboard(self):
        t = self.latex_output.toPlainText().strip()
        if t:
            QGuiApplication.clipboard().setText(t)
            QMessageBox.information(self, "Скопировано", "LaTeX скопирован.")

    def copy_tex_to_clipboard(self):
        t = self.tex_output.toPlainText().strip()
        if t:
            QGuiApplication.clipboard().setText(t)
            QMessageBox.information(self, "Скопировано", "TeX скопирован.")

    def save_tex_file(self):
        t = self.tex_output.toPlainText().strip()
        if not t:
            QMessageBox.information(self, "Пусто", "Пока нечего сохранять.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить TeX", "formula.tex", "TeX files (*.tex)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(t)
        QMessageBox.information(self, "Сохранено", f"Файл сохранён: {path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
