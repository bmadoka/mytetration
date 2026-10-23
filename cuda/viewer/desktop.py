"""Desktop protection checks (script runs of the viewer). Read-only Win32 queries; no input injection."""
import ctypes
import ctypes.wintypes as wt
import subprocess

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
MIN_IDLE_S = 300.0


class _LII(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


class _DD(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("DeviceName", wt.WCHAR * 32), ("DeviceString", wt.WCHAR * 128),
                ("StateFlags", wt.DWORD), ("DeviceID", wt.WCHAR * 128), ("DeviceKey", wt.WCHAR * 128)]


def attached_display_adapters():
    """Adapter names of all displays attached to the desktop (EnumDisplayDevices, DISPLAY_DEVICE_ATTACHED_TO_DESKTOP).
    Read-only; no window or GPU context involved."""
    out, i = [], 0
    while True:
        dd = _DD()
        dd.cb = ctypes.sizeof(dd)
        if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            return out
        if dd.StateFlags & 0x00000001:
            out.append(dd.DeviceString)
        i += 1


def pointer_pos():
    """Mouse pointer position in screen pixels (GetCursorPos; read only, the pointer is never moved)."""
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def window_pos_away_from_pointer(w=1000, h=620, margin=40):
    """Top-left of a w x h window (960x540 client + frame) on the primary screen that does not contain the pointer."""
    px, py = pointer_pos()
    sw = user32.GetSystemMetrics(0)
    for x, y in ((60, 60), (max(60, sw - w - 60), 60)):
        if not (x - margin <= px <= x + w + margin and y - margin <= py <= y + h + margin):
            return x, y
    return 60, 60


class _MI(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", wt.RECT), ("rcWork", wt.RECT), ("dwFlags", ctypes.c_ulong)]


def idle_seconds():
    """Seconds since the last user keyboard/mouse input (GetLastInputInfo)."""
    lii = _LII()
    lii.cbSize = ctypes.sizeof(lii)
    user32.GetLastInputInfo(ctypes.byref(lii))
    return ((kernel32.GetTickCount() - lii.dwTime) & 0xFFFFFFFF) / 1000.0


def foreground():
    return user32.GetForegroundWindow()


def fullscreen_app_in_front():
    """True if the foreground window covers its whole monitor (a game etc.), desktop/shell windows excluded."""
    h = user32.GetForegroundWindow()
    if not h:
        return False
    cls = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(h, cls, 256)
    if cls.value in ("Progman", "WorkerW", "Shell_TrayWnd", ""):
        return False
    r = wt.RECT()
    user32.GetWindowRect(h, ctypes.byref(r))
    mi = _MI()
    mi.cbSize = ctypes.sizeof(mi)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(h, 2), ctypes.byref(mi))
    m = mi.rcMonitor
    return r.left <= m.left and r.top <= m.top and r.right >= m.right and r.bottom >= m.bottom


def free_virtual_kb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def may_open_window():
    """(ok, reason): idle >= 5 min and no full-screen app in front."""
    idle = idle_seconds()
    if idle < MIN_IDLE_S:
        return False, f"user active {idle:.0f} s ago (< {MIN_IDLE_S:.0f} s)"
    if fullscreen_app_in_front():
        return False, "a full-screen application is in front"
    return True, f"idle {idle:.0f} s, no full-screen app"
