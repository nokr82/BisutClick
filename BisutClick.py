"""
비슷한 이미지 자동 클릭 프로그램 (GUI)
=================================
화면에서 템플릿 이미지와 유사한 부분을 모두 찾아두고,
사용자가 화면 어디를 클릭하든 그 클릭에 맞춰
캡처해둔 이미지와 비슷한 위치들을 전부 함께 자동 클릭해준다.

사용법
------
1. 프로그램 실행
2. "템플릿 영역 선택" 버튼(또는 F2)을 눌러 화면에서 기준 이미지를 드래그로 선택
3. 등록되면 자동으로 실행 상태가 되어, 화면 아무 곳이나 클릭할 때마다
   유사한 위치들이 함께 자동 클릭됨
4. "일시정지" 버튼(또는 F8)으로 일시정지/재개
5. 창을 닫거나 F9/ESC로 종료

주의: 마우스를 화면 맨 왼쪽 위 모서리로 이동시키면
      pyautogui의 안전장치(FAILSAFE)가 동작해 즉시 중단된다.
"""

import ctypes
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageGrab, ImageTk
from pynput import keyboard, mouse

# 모니터마다 배율(스케일)이 다른 듀얼 모니터 환경에서 좌표가 어긋나는 것을 막기 위해
# 최대한 이른 시점(윈도우 생성 전)에 프로세스를 모니터별 DPI 인식 상태로 전환한다.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# pyautogui 안전장치: 마우스를 화면 좌상단 끝으로 보내면 즉시 예외 발생 -> 프로그램 중단
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # 내부 호출 간 기본 지연은 우리가 직접 제어한다


def get_virtual_screen_rect():
    """모든 모니터를 합친 가상 데스크톱 영역 (x, y, w, h)을 반환한다.
    보조 모니터가 주 모니터의 왼쪽/위쪽에 있으면 x, y는 음수가 될 수 있다."""
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    user32 = ctypes.windll.user32
    x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return x, y, w, h


def screenshot_virtual(region=None):
    """모든 모니터를 포함해 캡처한다. region은 (x, y, w, h) 형태의 절대(가상 데스크톱)
    좌표이며, 반환값은 (이미지, 가상화면 원점 x, 가상화면 원점 y)이다."""
    img = ImageGrab.grab(all_screens=True)
    vx, vy, _, _ = get_virtual_screen_rect()
    if region is not None:
        x, y, w, h = region
        img = img.crop((x - vx, y - vy, x - vx + w, y - vy + h))
    return img, vx, vy


def get_monitors():
    """연결된 모든 모니터의 (x, y, w, h) 목록을 왼쪽->오른쪽, 위->아래 순서로 반환한다."""
    from ctypes import wintypes

    monitors = []

    def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        r = lprcMonitor.contents
        monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
        ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
    )
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_callback), 0)
    monitors.sort(key=lambda m: (m[0], m[1]))
    return monitors


@dataclass
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def contains(self, px, py, margin=0):
        return (self.x - margin <= px <= self.x + self.w + margin and
                self.y - margin <= py <= self.y + self.h + margin)


def non_max_suppression(matches, iou_threshold=0.3):
    """겹치는 매칭들 중 점수가 높은 것만 남긴다."""
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m.score, reverse=True)
    kept = []
    for m in matches:
        overlap = False
        for k in kept:
            ix1 = max(m.x, k.x)
            iy1 = max(m.y, k.y)
            ix2 = min(m.x + m.w, k.x + k.w)
            iy2 = min(m.y + m.h, k.y + k.h)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            union = m.w * m.h + k.w * k.h - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(m)
    return kept


class SimilarImageAutoClicker:
    """화면 스캔 + 자동 클릭을 담당하는 핵심 로직 (GUI와 분리)"""

    def __init__(self, threshold=0.85, scan_interval=1.0, max_matches=30,
                 click_delay=0.08, hit_margin=4, on_log=None, on_match_count=None):
        self.threshold = threshold
        self.scan_interval = scan_interval
        self.max_matches = max_matches
        self.click_delay = click_delay
        self.hit_margin = hit_margin

        self.on_log = on_log or (lambda msg: None)
        self.on_match_count = on_match_count or (lambda n: None)

        self.template_gray = None
        self.template_size = None  # (w, h)

        self.target_region = None  # None이면 전체(모든 모니터), 아니면 (x, y, w, h)로 스캔 범위를 제한

        self.matches = []
        self.lock = threading.Lock()

        self.paused = True  # 템플릿 등록 전까지는 사실상 일시정지 상태
        self.stop_event = threading.Event()
        self.ignore_clicks = threading.Event()  # 우리가 만든 합성 클릭을 무시하기 위한 플래그

    # ---------- 템플릿 등록 ----------
    def set_template(self, gray_arr, w, h):
        with self.lock:
            self.template_gray = gray_arr
            self.template_size = (w, h)
            self.matches = []
        self.paused = False
        self.on_match_count(0)
        self.on_log(f"[등록됨] 템플릿 크기 {w}x{h}")

    # ---------- 화면 스캔 ----------
    def scan_once(self):
        with self.lock:
            template = self.template_gray
            size = self.template_size
        if template is None:
            return

        region = self.target_region
        if region is not None:
            shot, _, _ = screenshot_virtual(region=region)
            ox, oy = region[0], region[1]
        else:
            shot, ox, oy = screenshot_virtual()  # 모든 모니터를 포함해 캡처
        screen = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)

        tw, th = size
        if screen.shape[0] < th or screen.shape[1] < tw:
            return

        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= self.threshold)
        candidates = []
        for x, y in zip(xs, ys):
            # 캡처 이미지 좌표 -> 실제 화면(가상 데스크톱) 좌표로 변환
            candidates.append(Match(int(x) + ox, int(y) + oy, tw, th, float(result[y, x])))

        candidates = non_max_suppression(candidates)
        candidates = sorted(candidates, key=lambda m: m.score, reverse=True)[: self.max_matches]

        with self.lock:
            self.matches = candidates
        self.on_match_count(len(candidates))

    def scan_loop(self):
        while not self.stop_event.is_set():
            if not self.paused and self.template_gray is not None:
                try:
                    self.scan_once()
                except Exception as e:
                    self.on_log(f"[스캔 오류] {e}")
            time.sleep(self.scan_interval)

    # ---------- 마우스 클릭 처리 ----------
    def on_click(self, x, y, button, pressed):
        if self.stop_event.is_set():
            return False  # 리스너 종료
        if button != mouse.Button.left or pressed:
            return  # 버튼을 뗀(release) 시점 = 사용자의 실제 클릭이 완전히 끝난 시점
        if self.ignore_clicks.is_set():
            return  # 우리가 만든 합성 클릭 -> 무시
        if self.paused:
            return

        with self.lock:
            matches = list(self.matches)

        if not matches:
            return

        # 어디를 클릭하든 상관없이, 이미 클릭한 지점(커서 바로 아래)만 제외하고
        # 화면에 있는 유사 이미지 위치를 전부 자동 클릭한다
        targets = [m for m in matches if not m.contains(x, y, margin=self.hit_margin)]
        if not targets:
            return

        self.on_log(f"[감지] ({x},{y}) 클릭됨 -> 유사 위치 {len(targets)}곳 자동 클릭")
        self.ignore_clicks.set()
        try:
            for m in targets:
                cx, cy = m.center
                pyautogui.click(cx, cy)
                time.sleep(self.click_delay)
        finally:
            # 합성 클릭 이벤트가 리스너에 늦게 도달할 수 있으므로 약간의 여유를 둔다
            time.sleep(0.15)
            self.ignore_clicks.clear()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BisutClick - 비슷한 이미지 자동 클릭기")
        self.geometry("460x480")
        self.minsize(420, 420)

        self.log_queue = queue.Queue()
        self.template_thumb_img = None  # PhotoImage 참조 유지(GC 방지)
        self.monitors = get_monitors()  # [(x, y, w, h), ...] 왼쪽->오른쪽 순

        self.clicker = SimilarImageAutoClicker(
            on_log=self._enqueue_log,
            on_match_count=self._enqueue_match_count,
        )

        self._build_widgets()

        self.scan_thread = threading.Thread(target=self.clicker.scan_loop, daemon=True)
        self.scan_thread.start()

        self.kb_listener = keyboard.Listener(on_press=self._on_press_key)
        self.kb_listener.start()

        self.ms_listener = mouse.Listener(on_click=self.clicker.on_click)
        self.ms_listener.start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._poll_queue)

    # ---------- 위젯 구성 ----------
    def _build_widgets(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.thumb_label = ttk.Label(top, text="[템플릿\n없음]", width=12, anchor="center",
                                      relief="groove", justify="center")
        self.thumb_label.pack(side=tk.LEFT, padx=(0, 10), ipady=20)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.select_btn = ttk.Button(btn_frame, text="① 템플릿 영역 선택 (F2)", command=self.start_capture)
        self.select_btn.pack(fill=tk.X, pady=2)

        self.toggle_btn = ttk.Button(btn_frame, text="② 일시정지 (F8)", command=self.toggle_pause,
                                      state=tk.DISABLED)
        self.toggle_btn.pack(fill=tk.X, pady=2)

        self.match_count_var = tk.StringVar(value="매칭: -")
        ttk.Label(btn_frame, textvariable=self.match_count_var).pack(anchor="w", pady=(4, 0))

        opts = ttk.LabelFrame(self, text="설정", padding=10)
        opts.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(opts, text="유사도 임계값").grid(row=0, column=0, sticky="w")
        self.threshold_var = tk.DoubleVar(value=self.clicker.threshold)
        self.threshold_label = ttk.Label(opts, text=f"{self.clicker.threshold:.2f}")
        threshold_scale = ttk.Scale(opts, from_=0.50, to=0.99, variable=self.threshold_var,
                                     command=self._on_threshold_change)
        threshold_scale.grid(row=0, column=1, sticky="ew", padx=5)
        self.threshold_label.grid(row=0, column=2)
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="대상 모니터").grid(row=1, column=0, sticky="w", pady=(6, 0))
        monitor_names = ["전체 화면 (모든 모니터)"] + [
            f"{i + 1}번 모니터 ({w}x{h})" for i, (_x, _y, w, h) in enumerate(self.monitors)
        ]
        self.monitor_var = tk.StringVar(value=monitor_names[0])
        self.monitor_combo = ttk.Combobox(opts, textvariable=self.monitor_var, values=monitor_names,
                                           state="readonly")
        self.monitor_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=(6, 0))
        self.monitor_combo.bind("<<ComboboxSelected>>", self._on_monitor_change)

        status_frame = ttk.Frame(self, padding=(10, 4))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="대기 중 - 템플릿을 먼저 선택하세요")
        ttk.Label(status_frame, textvariable=self.status_var, foreground="blue").pack(anchor="w")

        log_frame = ttk.LabelFrame(self, text="로그", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        hint = ttk.Label(self, text="F2 선택 · F8 일시정지/재개 · F9/ESC 종료 (창 밖에서도 동작)",
                          foreground="gray")
        hint.pack(pady=(0, 8))

    # ---------- 설정 변경 ----------
    def _on_threshold_change(self, value):
        v = round(float(value), 2)
        self.clicker.threshold = v
        self.threshold_label.config(text=f"{v:.2f}")

    def _on_monitor_change(self, event=None):
        idx = self.monitor_combo.current()  # 0 = 전체 화면, 1부터 각 모니터
        if idx <= 0:
            self.clicker.target_region = None
            self._log("[모니터] 전체 화면(모든 모니터) 대상으로 스캔합니다")
        else:
            region = self.monitors[idx - 1]
            self.clicker.target_region = region
            self._log(f"[모니터] {idx}번 모니터만 대상으로 스캔합니다 {region}")

    def _selected_monitor_rect(self):
        """현재 선택된 모니터의 (x, y, w, h)를 반환. 전체 선택 시 가상 데스크톱 전체."""
        if self.clicker.target_region is not None:
            return self.clicker.target_region
        return get_virtual_screen_rect()

    # ---------- 템플릿 캡처 ----------
    def start_capture(self):
        self._log("[선택 모드] 템플릿으로 사용할 영역을 드래그하세요 (ESC 취소)")
        self.withdraw()  # 선택 중에는 메인 창이 화면을 가리지 않도록 숨김
        self.after(150, self._do_capture)

    def _do_capture(self):
        region = self._select_region()
        self.deiconify()
        if not region:
            self._log("[취소됨] 영역이 선택되지 않았습니다")
            return

        x, y, w, h = region
        shot, _, _ = screenshot_virtual(region=(x, y, w, h))
        arr_gray = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)
        self.clicker.set_template(arr_gray, w, h)
        self._update_thumbnail(shot)

        self.toggle_btn.config(state=tk.NORMAL, text="② 일시정지 (F8)")
        self.status_var.set("실행 중 - 화면을 클릭하면 유사 이미지도 함께 클릭됩니다")

    def _select_region(self):
        # "-fullscreen"은 창이 열린 모니터 하나만 덮으므로, 지오메트리를 직접 지정한다.
        # 특정 모니터가 선택되어 있으면 그 모니터만, 아니면 가상 데스크톱 전체를 덮는다.
        vx, vy, vw, vh = self._selected_monitor_rect()

        def _fmt(v):
            return f"+{v}" if v >= 0 else str(v)

        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.geometry(f"{vw}x{vh}{_fmt(vx)}{_fmt(vy)}")
        top.attributes("-alpha", 0.25)
        top.attributes("-topmost", True)
        top.configure(bg="gray")
        top.config(cursor="cross")

        canvas = tk.Canvas(top, bg="gray", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        result = {"region": None}
        state = {"start": None, "start_root": None, "rect": None}

        def on_press(event):
            state["start"] = (event.x, event.y)
            state["start_root"] = (event.x_root, event.y_root)
            state["rect"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="red", width=2
            )

        def on_drag(event):
            if state["rect"] is None:
                return
            sx, sy = state["start"]
            # 사각형 미리보기는 캔버스 로컬 좌표를 써야 창이 화면 원점(0,0)이 아닌
            # 보조 모니터 위치에 있어도 올바르게 그려진다.
            canvas.coords(state["rect"], sx, sy, event.x, event.y)

        def on_release(event):
            sx, sy = state["start_root"]
            ex, ey = event.x_root, event.y_root
            x1, x2 = sorted((sx, ex))
            y1, y2 = sorted((sy, ey))
            if x2 - x1 > 3 and y2 - y1 > 3:
                result["region"] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            top.destroy()

        def on_escape(event):
            result["region"] = None
            top.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        top.bind("<Escape>", on_escape)
        top.focus_force()
        top.grab_set()
        self.wait_window(top)
        return result["region"]

    def _update_thumbnail(self, pil_shot):
        thumb = pil_shot.copy()
        thumb.thumbnail((96, 96), Image.LANCZOS)
        self.template_thumb_img = ImageTk.PhotoImage(thumb)
        self.thumb_label.config(image=self.template_thumb_img, text="")

    # ---------- 일시정지/재개 ----------
    def toggle_pause(self):
        if self.clicker.template_gray is None:
            return
        self.clicker.paused = not self.clicker.paused
        if self.clicker.paused:
            self.toggle_btn.config(text="② 재개 (F8)")
            self.status_var.set("일시정지됨")
            self._log("[일시정지]")
        else:
            self.toggle_btn.config(text="② 일시정지 (F8)")
            self.status_var.set("실행 중 - 화면을 클릭하면 유사 이미지도 함께 클릭됩니다")
            self._log("[재개]")

    # ---------- 로그/상태 (백그라운드 스레드 -> GUI 스레드 안전 전달) ----------
    def _enqueue_log(self, msg):
        self.log_queue.put(("log", msg))

    def _enqueue_match_count(self, count):
        self.log_queue.put(("count", count))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "count":
                    self.match_count_var.set(f"매칭: {payload}곳")
        except queue.Empty:
            pass
        if not self.clicker.stop_event.is_set():
            self.after(100, self._poll_queue)

    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ---------- 전역 핫키 (백그라운드 스레드에서 호출됨 -> GUI 스레드로 위임) ----------
    def _on_press_key(self, key):
        if key == keyboard.Key.f2:
            self.after(0, self.start_capture)
        elif key == keyboard.Key.f8:
            self.after(0, self.toggle_pause)
        elif key in (keyboard.Key.f9, keyboard.Key.esc):
            self.after(0, self.on_close)
            return False  # 리스너 종료

    # ---------- 종료 ----------
    def on_close(self):
        self.clicker.stop_event.set()
        try:
            self.kb_listener.stop()
        except Exception:
            pass
        try:
            self.ms_listener.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
