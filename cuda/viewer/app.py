"""mytetration real-time preview viewer.

Interactive:   python -m viewer --interactive       (drag = move, wheel = zoom; window 960x540, render target 1920x1080)
               keys 1-4 = pattern locations m37 / m191 / m59 / m60b at eps 1e-15, key 0 = start view (0, 0, 5);
               values in viewer/presets.json, HUD line over the window (viewer/presets.py). Script event:
               {"t": .., "type": "key", "key": "0".."4"}; a script may set "hud": true (default off).
Script replay: python -m viewer --script viewer/scripts/D1.json --mode timed  --log DIR [--readback precise|all|record]
               python -m viewer --script viewer/scripts/deepzoom2_responsive.json --mode threaded --log DIR
               python -m viewer --script viewer/scripts/D1.json --mode verify --log DIR   (one input per frame, read-backs)
               python -m viewer --script ... --mode media --log DIR  (timed, saves render-target snapshots for a GIF)
               python -m viewer --script ... --mode callbacks --log DIR (timed, events go through the registered
                                                                  GLFW mouse callbacks — the interactive input path)

While inputs arrive or zoom is settling: adaptive f32/f64 fast mode (viewer/fast.py). Once display equals target
and no input came for STOP_DELAY seconds: one precise frame (the existing double implementation). Script events are handed to the same
handler the mouse callbacks use (no OS input injection). Every frame is logged (frames.jsonl).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import params  # noqa: E402
import hybrid  # noqa: E402
from deep import hp, limits  # noqa: E402
from viewer import deep_state, notice, deep_view, presets  # noqa: E402

STOP_DELAY = 0.25          # s without input before the precise frame is computed (fixed before measuring)
TRANSITION_LIMIT = 2.0     # s, judged upper bound of last input -> precise frame on screen (fixed before measuring)


def gpu_used_mib():
    """Used MiB of the chosen GPU (nvidia-smi; --gpu / MYTET_GPU)."""
    from mytet_cuda import device
    return device.memory(device.choose()["uuid"])["used_mib"]


class Viewer:
    def __init__(self, start, mode, readback, log_dir, start_hp=None, hud=None):
        t_init = time.perf_counter()
        import desktop
        import glfw
        import render
        self.glfw, self.render, self.desktop = glfw, render, desktop
        self.mode, self.readback, self.log_dir = mode, readback, log_dir
        self.state, self.hp = deep_state.initial_states(start, start_hp)
        # Number-key presets: name of the last preset key (None = start view / no preset) and whether a drag moved
        # the view since. The HUD shows them over the window; on by default only for a person at the PC.
        self.preset_active, self.preset_moved = None, False
        presets.table()  # read and check viewer/presets.json now, not at the first key press
        self.hud = (mode == "interactive") if hud is None else bool(hud)
        self.limit_until = None
        self.notice_targets = {}
        self.deep_controller = None
        self.deep_fallback = None
        self.deep_draw_key = None
        self.display = self.state
        self.decision_t = None
        self.prev_state = self.state
        self.applied = -1            # index of the last applied input
        self.drawn = -1              # last input reflected by the last presented frame
        self.last_params = None
        self.last_kind = None
        self.first_new_input_t = None
        self.last_input_t = None
        self.input_t = {}            # input index -> delivery time
        self.input_log = []          # inputs as the handler received them
        self.press_sources = []      # where each mouse press took its start position from
        self.frames = []
        self.fresh = []
        self.next_fresh_id = 0
        self.source_map = None
        self.hybrid_version = 3
        self.last_precise = None
        self.margin_maps = {}
        self.shown_maps = []  # (first present seq, fresh record, immutable map), newest first
        self.shown_ids = set()
        self.source_texture = False
        self.bound_source_fresh_id = None
        self.fresh_bits = []
        self.job = None
        self.job_record = None
        self.source = None
        self.hybrid_mode = "fresh-only"
        self.skipped_same_params = 0
        self.pool_max = 0
        self.media = []
        self.recorded = []           # readback "record": every presented window frame (R channel), for the video
        self.meta = {"start": list(self.state), "mode": mode, "stop_delay": STOP_DELAY,
                     "transition_limit": TRANSITION_LIMIT, "pid": os.getpid(), "t_viewer_init": t_init}
        self.meta['start_hp'] = [str(v) for v in self.hp[:2]]
        used0 = gpu_used_mib()
        # script runs: open where the (resting) mouse pointer is not, so no real mouse message reaches the window
        pos = (60, 60) if mode == "interactive" else desktop.window_pos_away_from_pointer()
        self.meta.update({"pointer_at_start": list(desktop.pointer_pos()), "window_pos": list(pos)})
        self.disp = render.Display(visible=True, on_top=(mode != "interactive"), pos=pos)
        self.meta["hud"] = self.hud
        self._update_hud()
        used_gl = gpu_used_mib()
        self.meta.update({"monitor_refresh_hz": self.disp.refresh_hz, "hybrid_version": 3,
                          "renderer": self.disp.renderer, "hwnd": int(self.disp.hwnd or 0),
                          "topmost_set": self.disp.topmost_set, "window_state_after_show": self.disp.window_state(),
                          "foreground_is_viewer_after_show": int(desktop.foreground()) == int(self.disp.hwnd or -1),
                          "gpu_used_mib_before_gl": used0, "gpu_used_mib_after_gl": used_gl})
        from mytet_cuda import gpu
        self.meta["cuda_device"] = gpu.device_info()  # GPU rules checked here, before any allocation
        time.sleep(0.5)
        used_gl = gpu_used_mib()  # again, right before the CUDA context
        gpu.cp.empty(1)          # first allocation creates the CUDA context
        gpu.cp.cuda.Device().synchronize()
        sys.path.insert(0, str(HERE.parent / 'scripts'))
        from perturb_tools import audit_gpu
        self.meta['gpu_process'] = audit_gpu('viewer context', log_dir or HERE.parent / 'logs/perturb/interactive')
        time.sleep(0.5)
        used_cuda = gpu_used_mib()
        self.meta.update({"gpu_used_mib_after_cuda_context": used_cuda, "cuda_context_mib": used_cuda - used_gl,
                          "gl_context_mib": used_gl - used0})
        import fast
        self.fast = fast
        from margins import MarginScheduler
        self.margins = MarginScheduler(
            lambda r, rect: fast.MarginChunk(r["params"],r["precision"],rect,fast.MX,fast.MY),
            fast.margin_rectangles(),fast.MX,fast.MY,self._margin_complete)
        self.meta["margin_width"] = [fast.MX,fast.MY]
        self.meta["margin_mode"] = self.margins.mode
        self._drag_from = None
        self._cursor = None
        self.shot_mid = False
        # Mouse callbacks: registered for interactive use and for the «callbacks» replay (which drives these very
        # registered callbacks with script events). timed / verify / media replays take no mouse input at all.
        if mode in ("interactive", "callbacks"):
            glfw.set_cursor_pos_callback(self.disp.win, self._on_cursor)
            glfw.set_mouse_button_callback(self.disp.win, self._on_button)
            glfw.set_scroll_callback(self.disp.win, self._on_scroll)
            glfw.set_key_callback(self.disp.win, self._on_key)

    # ---- input (mouse callbacks and script events use the same handler) ----
    def _on_key(self, win, key, scancode, action, mods):
        """Digit keys 0-4 (top row or keypad), press only: jump to that preset. Every other key, and key release /
        repeat, is ignored here and never reaches on_input."""
        if action != self.glfw.PRESS:
            return
        g = self.glfw
        digit = next((d for d in "01234" if key in (getattr(g, "KEY_" + d), getattr(g, "KEY_KP_" + d))), None)
        if digit is not None:
            self.on_input({"type": "key", "key": digit})

    def _update_hud(self):
        if getattr(self, "hud", False):
            self.disp.set_hud(presets.hud_text(getattr(self, "preset_active", None),
                                               getattr(self, "preset_moved", False), self.state[2]))

    def _on_button(self, win, button, action, mods):
        if button == self.glfw.MOUSE_BUTTON_LEFT:
            if action == self.glfw.PRESS:
                if self._cursor is not None:
                    self._drag_from = self._cursor
                    self.press_sources.append("cursor_callback")
                else:  # no cursor notification yet (first press): ask GLFW where the cursor is
                    self._drag_from = tuple(self.glfw.get_cursor_pos(win))
                    self.press_sources.append("get_cursor_pos")
            else:
                self._drag_from = None

    def _on_cursor(self, win, x, y):
        self._cursor = (x, y)
        if self._drag_from is not None:
            fx, fy = self._drag_from
            self._drag_from = (x, y)
            # window pixels -> render-target pixels (the window shows the render target down-scaled by SCALE)
            dx, dy = params.window_to_render_target(x - fx, y - fy, self.render.SCALE)
            self.on_input({"type": "drag", "dx": dx, "dy": dy})

    def _on_scroll(self, win, xoff, yoff):
        self.on_input({"type": "wheel", "steps": yoff})

    def on_input(self, ev):
        t = time.perf_counter()
        self.prev_state = self.state  # parameters before this input (kept for the frame log / diagnostics)
        self._input_before_hp = self.hp if getattr(self, "hp", None) is not None else hp.from_floats(*self.state)
        if getattr(self, 'hp', None) is None:  # CPU fixtures without __init__
            self.hp = hp.from_floats(*self.state)
        logged = dict(ev)
        if ev['type'] == 'key':
            # Preset jump: the target state is the preset (bit for bit); KeyError for a key that is not 0-4.
            self.state, self.hp = presets.states(ev['key'])
            logged['resync'] = False
            logged['preset'] = presets.name(ev['key']) or 'start'
            self.preset_active, self.preset_moved = presets.name(ev['key']), False
            self.limit_until = None
            self.snap_display = True  # the next frame shows the preset itself, not a smooth zoom towards it
        else:
            self.state, self.hp, clamped = deep_state.apply_event(self.state, self.hp, ev, logged)
            if clamped:
                logged['clamped'] = True
                self.limit_until = t + 4.0
            elif ev['type'] == 'wheel' and ev['steps'] < 0:
                self.limit_until = None
            if ev['type'] == 'drag' and getattr(self, 'preset_active', None) is not None:
                self.preset_moved = True
        self._update_hud()
        logged['t_processed'] = t
        logged.setdefault('t_scheduled', t)
        logged.setdefault('t_enqueued', t)
        logged['before_hp'] = hp.to_record(self.__dict__.get('_input_before_hp', self.hp))
        logged['after_hp'] = hp.to_record(self.hp)
        logged['crossed_deep'] = (self.prev_state[2] < limits.DEEP_BELOW) != (self.state[2] < limits.DEEP_BELOW)
        self.input_log.append(logged)  # the inputs as the handler received them (checked against the script)
        self.applied += 1
        self.input_t[self.applied] = t
        self.last_input_t = t
        if self.first_new_input_t is None:
            self.first_new_input_t = t

    # ---- frames ----
    def _fresh_id(self):
        ident = self.next_fresh_id
        self.next_fresh_id += 1
        return ident

    def _archive(self):
        from deferred import MapArchive
        if not hasattr(self, "map_archive"):
            self.map_archive = MapArchive()
        return self.map_archive

    def _complete_fresh(self, m, rec, upload=True, t_end=None):
        rec.update(t_end=time.perf_counter() if t_end is None else t_end)
        self._archive().register(("fresh",rec["id"]),m,rec)
        self.fresh.append(rec)
        self._set_source(m, rec, upload)

    def _margin_complete(self, rec, m):
        self.margin_maps[rec["id"]] = m
        # Only shown maps get margins, so completion is already after first present.
        margin_rec = rec.setdefault("margin_archive", {})
        self._archive().register(("margin",rec["id"]),m,margin_rec)
        self.map_archive.shown(("margin",rec["id"]),rec["first_shown_t"])

    def _finish_archive(self):
        last = max((f["t_present"] for f in self.frames),default=0.)
        self._archive().finish(last)
        if self.log_dir:
            self.fresh_bits = [self.map_archive.bits(("fresh",r["id"])) for r in self.fresh]

    def _set_source(self, m, rec, upload):
        """Timed completions and recorded first-use events share this source update."""
        self.source = rec
        self.source_map = m
        self.source_texture = False
        if upload:
            self._upload_source()

    def _upload_source(self):
        self.disp.upload_source(self.source_map)
        self.bound_source_fresh_id = self.source["id"]
        self.source_texture = True

    def _drain_fresh(self):
        """Teardown only; the moving display loop always polls without waiting."""
        if self.job is not None:
            self.job.event.synchronize()
            self._poll_fresh()

    def _poll_fresh(self):
        if getattr(self,"hybrid_version",1)==3:
            self.margins.tick(True, launch=False)
        if self.job is not None:
            m = self.job.poll()
            if m is not None:
                self._complete_fresh(m, self.job_record)
                self.job = self.job_record = None

    def _start_fresh(self, p, t_decide):
        if self.job is not None or not (self.applied > self.drawn or p != self.state):
            return
        # HYBRID_REPLACEMENT_ANCHOR
        computing = [self.job_record] if self.job_record is not None else []
        if hybrid.duplicate_params(p, computing, self.fresh):  # HYBRID_DUPLICATE_GUARD
            return
        self.job = self.fast.FreshJob(*p)
        self.job_record = {"id": self._fresh_id(), "params": list(p), "params_hex": [v.hex() for v in p],
                           "display_params_hex": [v.hex() for v in p], "t_decide": t_decide, "k": self.applied,
                           "t_start": self.job.t_start, "precision": self.job.precision, "async": True,
                           "mode": self.hybrid_mode}

    def frame(self, kind, t_decide=None):
        t_decide = time.perf_counter() if t_decide is None else t_decide
        p = self.state
        if kind == "fast" and not getattr(self, "snap_display", False):
            p = params.smooth_step(self.display, self.state, t_decide - self.decision_t)
        self.snap_display = False  # a preset key jumps straight to its state (no smooth zoom); one frame only
        if p[2] < limits.DEEP_BELOW:
            return self._deep_frame(kind, p, t_decide)
        if getattr(self, 'deep_controller', None) is not None:
            self.deep_controller.suspend()
        self.deep_draw_key = self.deep_fallback = None
        self.disp._deep_texture_id = None
        frame_type = "new_input" if self.applied > self.drawn else "interp"
        self._poll_fresh()
        candidate, _ = hybrid.decide_mode(self.hybrid_mode, self.fresh)
        rec = {"seq": len(self.frames), "kind": kind, "last_input": self.applied, "params": list(p),
               "params_hex": [v.hex() for v in p], "t_first_new_input": self.first_new_input_t,
               "t_last_input": self.last_input_t}
        rec.update({"t_decide": t_decide, "k": self.applied, "display_params": list(p),
                    "display_params_hex": [v.hex() for v in p],
                    "target_params_hex": [v.hex() for v in self.state], "frame_type": frame_type})
        if kind != "fast" or candidate == "fresh-only":
            rec["t_compute_start"] = time.perf_counter()
            m = (self.fast.fast_map if kind == "fast" else self.fast.precise_map)(*p)
            rec["t_compute_end"] = time.perf_counter()
            rec["compute_ms"] = 1000 * (rec["t_compute_end"] - rec["t_compute_start"])
            if kind == "fast":
                rec["precision"] = next(reversed(self.fast.calls["precision"]), None)
                fresh = {"id": self._fresh_id(), "params": list(p), "params_hex": [v.hex() for v in p],
                         "display_params_hex": [v.hex() for v in p], "t_decide": t_decide, "k": self.applied,
                         "t_start": rec["t_compute_start"], "precision": rec["precision"], "async": False,
                         "mode": candidate}
                self._complete_fresh(m, fresh, upload=False, t_end=rec["t_compute_end"])
                rec["map_sha256"] = None
            else:
                rec["map_sha256"] = None
                rec["precise_id"] = "precise_" + str(rec["seq"])
                self._archive().register(("precise",rec["precise_id"]),m,dict(params=list(p),id=rec["precise_id"]))
        else:
            # A warped present does not claim a new computation. The source has its own start/end record.
            rec["t_compute_start"] = rec["t_compute_end"] = t_decide
            rec["compute_ms"] = 0.0
        if kind == "fast":
            # The only authoritative mode decision: after any synchronous map finished,
            # immediately before draw/present, from the records available right now.
            rec["t_mode_decide"] = time.perf_counter()
            self.hybrid_mode, median = hybrid.decide_mode(self.hybrid_mode, self.fresh)
            rec.update(mode=self.hybrid_mode, m=median, source_fresh_id=self.source["id"],
                       precision=self.source["precision"], cycle_exit=bool(self.source["precision"] == "f64"
                                                                          and self.fast.USE_CYCLE_EXIT))
            if self.hybrid_mode == "warped":
                rec["present_kind"] = "warped"
            else:
                rec.update(present_kind="fresh", scale=1.0, offset=[0.0, 0.0])
        else:
            rec["t_mode_decide"] = time.perf_counter()
            self.hybrid_mode, median = hybrid.decide_mode(self.hybrid_mode, self.fresh)
            rec.update(present_kind=kind, precision="double", mode=self.hybrid_mode, m=median,
                       source_fresh_id=None, scale=1.0, offset=[0.0, 0.0])
        # Start the next async screen job before drawing; keep this present's source unchanged.
        if kind == "fast" and self.hybrid_mode == "warped":
            self._start_fresh(p, t_decide)
        if getattr(self,"hybrid_version",1) == 3:
            self.margins.tick(self.job is not None, launch=self.hybrid_mode == "warped")
        self._draw_present(rec, m if kind != "fast" or candidate == "fresh-only" else None)
        rec['compute_path'] = ('warped' if rec['present_kind'] == 'warped' else rec['precision']) if kind == 'fast' else 'double'
        if self.readback == "all" or (self.readback == "precise" and kind != "fast"):
            bb_for_hash = self.disp.read_back_buffer()
        if self.readback == "record":
            self.recorded.append(self.disp.read_back_buffer()[..., 0].copy())
        rec["present_hr"] = self.disp.present()
        rec["t_present"] = time.perf_counter()
        key = ("fresh", rec["source_fresh_id"]) if kind == "fast" else ("precise",rec["precise_id"])
        self._archive().bind(rec,key)
        self.map_archive.shown(key,rec["t_present"])  # SHA_AFTER_PRESENT
        if kind != "fast" and getattr(self,"hybrid_version",1) == 3:
            self.last_precise = ({"id":rec["precise_id"],"params":list(p)},m)
            self.shown_maps, self.shown_ids = [],set()
            self.margins.retain(set())
            keep=self.source["id"] if self.source is not None else None
            self.margin_maps={i:m for i,m in self.margin_maps.items() if i==keep}
        if self.readback == "all" or (self.readback == "precise" and kind != "fast"):
            rec["bb_sha256"] = hashlib.sha256(bb_for_hash.tobytes()).hexdigest()
        if getattr(self,"hybrid_version",1) == 3:
            if kind == "fast" and self.hybrid_mode == "warped":
                self.margins.shown(self.source,self.source_map,rec["seq"])
            self.margins.tick(self.job is not None, launch=self.hybrid_mode == "warped")
        if self.readback == "all" or (self.readback == "precise" and kind != "fast"):
            rec["rt_sha256"] = hashlib.sha256(self.disp.read_render_target().tobytes()).hexdigest()
        self.pool_max = max(self.pool_max, self.fast.pool_bytes())
        rec["pool_bytes"] = self.fast.pool_bytes()
        if getattr(self, 'hp', None) is not None:
            rec.update(deep_state.diagnostics(self.state, self.hp, p[2]))
        if getattr(self, 'hud', False):
            rec['hud'] = self.disp.hud_text  # the HUD line drawn over this present
        self.frames.append(rec)
        self.drawn = self.applied
        self.display = p
        self.decision_t = t_decide
        self.last_params = p
        self.last_kind = kind
        self.first_new_input_t = None
        if self.mode == "media":
            self._snapshot(rec)

    def _deep_frame(self, kind, p, t_decide):
        """Nonblocking deep present: poll child, schedule, transform, present every tick."""
        import numpy as np
        from viewer.deep_worker import DeepController, ProcessWorker, engine_factory_for
        self.source_texture = False
        self.bound_source_fresh_id = None
        high = self.hp[0], self.hp[1], p[2]
        controller = getattr(self, 'deep_controller', None)
        if getattr(self, 'deep_fallback', None) is None and (controller is None or controller.source is None):
            # Snapshot the last shallow B3 composition BEFORE starting deep work.
            # This changes no shallow draw, shader, or candidate-selection rule.
            previous = self.frames[-1] if self.frames else None
            if previous:
                image = (self.disp.read_render_target()[...,0] > 127).T.copy()
                origin = hp.from_floats(*previous['display_params'])
            else:
                image, origin = np.zeros((1920,1080), bool), high
            self.deep_fallback = image, origin
        if getattr(self, 'deep_controller', None) is None:
            directory = self.log_dir or HERE.parent/'logs/deepzoom2/interactive'
            self.deep_controller = DeepController(ProcessWorker(directory, engine_factory_for(self.mode)), directory)
        moving = (self.last_input_t is not None and t_decide-self.last_input_t < STOP_DELAY) or p != self.state
        source = self.deep_controller.tick(high, moving, self.applied)
        rec = dict(seq=len(self.frames), kind=kind, last_input=self.applied, k=self.applied,
                   params=list(p), params_hex=[v.hex() for v in p], display_params=list(p),
                   display_params_hex=[v.hex() for v in p], target_params_hex=[v.hex() for v in self.state],
                   t_decide=t_decide, t_first_new_input=self.first_new_input_t, t_last_input=self.last_input_t,
                   frame_type='new_input' if self.applied > self.drawn else 'interp',
                   compute_path='perturb', precision='perturb', mode='warped', source_fresh_id=None,
                   t_compute_start=t_decide, t_compute_end=t_decide, compute_ms=0., map_sha256=None,
                   intermediate=True, source_request_id=None, source_stride=None, moving=moving)
        text = None
        info = source['info'] if source else dict(out_of_range=False, reason=None)
        # Force the notice immediately, including before the worker's first response.
        if os.environ.get('MYTET_FORCE_RANGE_EXIT') == '1':
            info = dict(info, out_of_range=True, reason='forced')
        if info['out_of_range']:
            text = notice.range_text(info['reason'])
            rec['notice'] = 'range:' + info['reason']
        elif self.limit_until is not None and time.perf_counter() < self.limit_until:
            text, rec['notice'] = notice.LIMIT_TEXT, 'limit'
        rec['deep'] = dict(info, x0=str(high[0]), y0=str(high[1]), eps_hex=p[2].hex())
        if source:
            rec.update(source_request_id=source['id'], source_stride=source['stride'],
                       source_hp=hp.to_record(source['hp']), source_t_start=source['t_start'], source_t_end=source['t_end'],
                       map_sha256=source['map_sha256'])
            exact = deep_view.state_key(high) == deep_view.state_key(source['hp'])
            rec['present_kind'] = ('deep_warped' if moving or not exact else
                                   'deep_final' if source['stride'] == 1 else 'deep_coarse')
            rec['intermediate'] = rec['present_kind'] != 'deep_final' or bool(text)
            image, origin, stride, ident = source['map'], source['hp'], source['stride'], ('deep',source['id'])
        else:
            rec['present_kind'] = 'deep_warped'
            image, origin = self.deep_fallback
            stride, ident = 1, ('shallow_fallback',id(image))
        draw_key = (ident, deep_view.state_key(high), text, moving)
        repeat = draw_key == getattr(self, 'deep_draw_key', None)
        if text:
            rec.update(present_kind='deep_notice', intermediate=True, notice_text=text,
                       notice_text_pixels=int(np.count_nonzero(notice.text_mask(text))))
            if not repeat:
                base = None if info['out_of_range'] else deep_view.warp_map(image,high,origin,stride)[0]
                self.disp.draw(notice.notice_map(base,text))
                self.disp._deep_texture_id = None
        elif not repeat:
            rec['transform'] = self.disp.draw_deep(image, high, origin, stride, ident)
        else:
            rec['transform'] = self.frames[-1].get('transform')
        if repeat:
            self.disp.hold_deep()
            rec['present_kind'] = 'deep_hold'
        # A hold retains whether the underlying picture is final or intermediate.
        rec['underlying_kind'] = ('deep_notice' if text else 'deep_warped' if moving else
                                  'deep_final' if source and exact and stride == 1 else
                                  'deep_coarse' if source and exact else 'deep_warped')
        if 'transform' not in rec:
            rec['transform'] = deep_view.warp_indices(high, origin, stride)[2]
        self.deep_draw_key = draw_key
        if self.readback == 'all' or (self.readback == 'precise' and not rec['intermediate']):
            rec['bb_sha256'] = hashlib.sha256(self.disp.read_back_buffer().tobytes()).hexdigest()
        if self.readback == 'record':
            self.recorded.append(self.disp.read_back_buffer()[...,0].copy())
        rec['present_hr'] = self.disp.present()
        rec['t_present'] = time.perf_counter()
        if self.readback == 'all' or (self.readback == 'precise' and not rec['intermediate']):
            rt = self.disp.read_render_target()
            rec['rt_sha256'] = hashlib.sha256(rt.tobytes()).hexdigest()
            if text:
                actual = (rt[...,0] > 127).T
                self.notice_targets[str(rec['seq'])] = np.packbits(actual)
                rec['notice_rendered_text_pixels'] = int(np.count_nonzero(actual & notice.text_mask(text)))
        rec.update(deep_state.diagnostics(self.state,self.hp,p[2]))
        if getattr(self, 'hud', False):
            rec['hud'] = self.disp.hud_text  # the HUD line drawn over this present
        self.frames.append(rec)
        self.drawn, self.display, self.decision_t = self.applied, p, t_decide
        self.last_params, self.last_kind, self.first_new_input_t = p, kind, None
        if self.mode == 'media':
            self._snapshot(rec)

    def _draw_present(self, rec, m=None, recorded_transform=False):
        """One draw path for timed and recorded-order verification presents."""
        if rec["kind"] == "fast" and getattr(self,"hybrid_version",1) == 3:
            self._draw_b3(rec, recorded_transform)
        elif rec["kind"] == "fast" and getattr(self,"hybrid_version",1) == 2:
            if self.source["id"] not in self.shown_ids:
                self.shown_ids.add(self.source["id"])
                self.shown_maps.append((rec["seq"], self.source, self.source_map))
                self.shown_maps.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
                self.shown_maps = self.shown_maps[:8]
            if recorded_transform:
                by_id = {r["id"]: (r, m) for _, r, m in self.shown_maps}
                candidates = [by_id[i] for i in rec["candidate_ids"]]
            else:
                candidates = [(r, m) for _, r, m in self.shown_maps]
                rec["candidate_ids"] = [r["id"] for r, _ in candidates]
            if rec["present_kind"] == "warped":
                transforms, counts = self.disp.draw_candidates(
                    rec["display_params"], candidates,
                    rec["candidate_transforms"] if recorded_transform else None)
                if not recorded_transform:
                    rec["candidate_transforms"], rec["pixel_counts"] = transforms, counts
                    rec["scale"], rec["offset"] = transforms[0]["scale"], transforms[0]["offset"]
                self.bound_source_fresh_id = self.disp.bound_candidate_ids[0]
            else:
                self._draw_fresh(self.source_map, rec["kind"])
                sources = [r for r, _ in candidates]
                cheap = hybrid.fresh_candidate_records(rec["display_params"], sources)
                transforms, counts = cheap if cheap is not None else hybrid.candidate_plan(rec["display_params"], sources)[1:]
                rec["candidate_transforms"], rec["pixel_counts"] = transforms, counts
            rec["bound_candidate_ids"] = (list(self.disp.bound_candidate_ids) if rec["present_kind"] == "warped"
                                          else list(rec["candidate_ids"]))
        elif rec["present_kind"] == "warped":
            if not self.source_texture:
                self._upload_source()
            if recorded_transform:
                transform = (rec["scale"], rec["offset"])
                self.disp.draw_warped(rec["display_params"], self.source["params"], transform=transform)
            else:
                rec["scale"], rec["offset"] = self.disp.draw_warped(rec["display_params"], self.source["params"])
        else:
            self._draw_fresh(self.source_map if rec["kind"] == "fast" else m, rec["kind"])
        if rec["kind"]!="fast" and getattr(self,"hybrid_version",1)==3:
            ident=rec["precise_id"]
            tr={"id":ident,"scale":1.0,"offset":[0.0,0.0]}
            if not recorded_transform:
                rec.update(candidate_ids=[ident],candidate_transforms=[tr],
                           candidates=[{"id":ident,"part":"precise","transform":tr}],
                           pixel_counts=dict(newest=0,older=0,margin=0,precise=1920*1080,edge=0))
            rec["bound_candidate_ids"]=[ident]
        # Sample at draw time, independently of the intended/logged source id.
        rec["bound_source_fresh_id"] = self.bound_source_fresh_id

    def _draw_b3(self, rec, recorded_transform):
        ident=self.source["id"]
        if ident not in self.shown_ids:
            self.shown_ids.add(ident)
            self.shown_maps.append((rec["seq"],self.source,self.source_map))
            self.shown_maps.sort(key=lambda t:(t[0],t[1]["id"]),reverse=True)
            self.shown_maps=self.shown_maps[:64]
        if not recorded_transform:
            retained={r["id"] for _,r,_ in self.shown_maps}
            self.margins.retain(retained)
            self.margin_maps={i:m for i,m in self.margin_maps.items() if i in retained}
        if recorded_transform:
            lookup={r["id"]:(r,m) for _,r,m in self.shown_maps}
            if self.last_precise:
                r,m=self.last_precise
                lookup[r["id"]]=(r,m)
            candidates=[]
            for c in rec["candidates"]:
                r,m=lookup[c["id"]]
                mxmy=r.get("margin_width",[0,0]) if c["part"]=="screen+margin" else [0,0]
                if c["part"]=="screen+margin":
                    m=self.margin_maps[c["id"]]
                candidates.append((dict(r,part=c["part"],margin=mxmy),m))
        else:
            candidates=hybrid.motion_candidates(self.shown_maps,self.last_precise,rec["t_decide"],self.margin_maps.get)
        if rec["present_kind"]=="warped":
            transforms,counts=self.disp.draw_candidates(rec["display_params"],candidates,
                rec["candidate_transforms"] if recorded_transform else None,version=3)
            self.bound_source_fresh_id=self.disp.bound_candidate_ids[0]
        else:
            self._draw_fresh(self.source_map,"fast")
            # Identity newest covers all pixels; preserve scalar shallow path.
            cheap=hybrid.fresh_candidate_records(rec["display_params"],[r for r,_ in candidates])
            if cheap is not None:
                transforms,_=cheap
                counts=dict(newest=1920*1080,older=0,margin=0,precise=0,edge=0)
            else:
                _,transforms,counts=hybrid.candidate_plan3(rec["display_params"],[r for r,_ in candidates])
        if not recorded_transform:
            rec["candidate_ids"]=[r["id"] for r,_ in candidates]
            rec["candidate_transforms"],rec["pixel_counts"]=transforms,counts
            rec["candidates"]=[{"id":r["id"],"part":r["part"],"transform":tr} for (r,_),tr in zip(candidates,transforms)]
            rec["scale"],rec["offset"]=transforms[0]["scale"],transforms[0]["offset"]
        rec["bound_candidate_ids"]=(list(self.disp.bound_candidate_ids) if rec["present_kind"]=="warped" else list(rec["candidate_ids"]))

    def _draw_fresh(self, m, kind):
        self.disp.draw(m)
        self.source_texture = kind == "fast"
        self.bound_source_fresh_id = self.source["id"] if kind == "fast" else None

    def _snapshot(self, rec):
        t = rec["t_present"]
        if rec["kind"] == "fast" and self.media and t - self.media[-1]["t"] < 0.1:
            return
        img = self.disp.read_render_target()[::-1, :, :3]  # render target, rows top->bottom for image files
        t0 = self.meta.get("t_playback_start")
        full = None
        if rec["kind"] == "precise" or (rec["kind"] == "fast" and not self.shot_mid and t0 and t - t0 >= 2.5):
            full = img.copy()
            self.shot_mid = self.shot_mid or rec["kind"] == "fast"
        self.media.append({"t": t, "seq": rec["seq"], "kind": rec["kind"], "small": img[::4, ::4].copy(), "full": full})

    def _registered_callbacks(self):
        """The callbacks GLFW really holds for this window (read back by swapping them out and in again)."""
        g, w = self.glfw, self.disp.win
        cbs = {}
        for name, setter in (("cursor", g.set_cursor_pos_callback), ("button", g.set_mouse_button_callback),
                             ("scroll", g.set_scroll_callback)):
            cbs[name] = setter(w, None)
            setter(w, cbs[name])
        return cbs

    def _registered_key_callback(self):
        """The key callback GLFW really holds for this window (read back the same way)."""
        g, w = self.glfw, self.disp.win
        cb = g.set_key_callback(w, None)
        g.set_key_callback(w, cb)
        return cb

    def _deliver_via_callbacks(self, ev, cbs, st):
        """Script event -> the registered GLFW callbacks, in the order GLFW would call them (no OS input injection, no
        write to the viewer's state). Drag: press (if not held), then a cursor move of (dx, dy)/SCALE window pixels
        (exact halves); consecutive drags are consecutive moves with the button held. The very first press comes with
        NO cursor notification before it (as when a window opens under a resting pointer), so the press handler has to
        ask glfw.get_cursor_pos, which during this replay is the deterministic stand-in installed by run(); later
        presses follow a cursor notification. Wheel: release a held button, move the cursor (button up: no input
        expected), then scroll `steps`. Key ("key": "0".."4"): release a held button the same way, then press and
        release that top-row digit key."""
        g, w = self.glfw, self.disp.win
        if ev["type"] == "drag":
            if not st["pressed"]:
                if not st["first_press_done"]:
                    st["first_press_done"] = True  # press straight away at the stand-in position st["pos"]
                else:
                    cbs["cursor"](w, *st["pos"])
                cbs["button"](w, g.MOUSE_BUTTON_LEFT, g.PRESS, 0)
                st["pressed"] = True
            st["pos"] = (st["pos"][0] + ev["dx"] / self.render.SCALE, st["pos"][1] + ev["dy"] / self.render.SCALE)
            cbs["cursor"](w, *st["pos"])
        else:
            if st["pressed"]:
                cbs["button"](w, g.MOUSE_BUTTON_LEFT, g.RELEASE, 0)
                st["pressed"] = False
                st["pos"] = (st["pos"][0] + 3.0, st["pos"][1] + 2.0)
                cbs["cursor"](w, *st["pos"])  # a move after release: no input expected
            if ev["type"] == "key":
                code = getattr(g, "KEY_" + ev["key"])
                cbs["key"](w, code, 0, g.PRESS, 0)
                cbs["key"](w, code, 0, g.RELEASE, 0)  # release: no input expected
            else:
                cbs["scroll"](w, 0.0, ev["steps"])

    def run_honesty(self, replay_dir):
        """Untimed full-target and window readback of every recorded present."""
        sys.path.insert(0, str(HERE.parent / "tests"))
        import viewer_check as vc
        replay_meta, frames, _ = vc.load(replay_dir)
        fresh, maps = vc.load_fresh(replay_dir)
        self.hybrid_version = vc.recording_version(replay_meta)
        records = vc.validate_present_records(frames, fresh,replay_meta)
        if self.hybrid_version==3:
            self.margin_maps,self.replay_precise=vc.load_auxiliary(replay_dir,fresh,frames)
            records.update(vc.b3_record_checks(frames,fresh))
        self.last_precise=None
        # Validate the whole recording before allowing any recorded transform to draw.
        replay_frames = [] if any(records.values()) else frames
        precise = vc.PreciseCache()
        checks = vc.ReplayChecks()
        source_id = None
        self.shown_maps, self.shown_ids = [], set()
        for f in replay_frames:
            self.glfw.poll_events()
            if self.glfw.window_should_close(self.disp.win):
                raise RuntimeError("honesty replay closed before all presents")
            ident = f["source_fresh_id"]
            if f["kind"] == "fast":
                if source_id != ident:
                    src = fresh[ident]
                    self._set_source(maps[ident], src, upload=src["async"])
                    source_id = ident
                expected = vc.displayed_map(f, fresh, maps, replay_meta, margins=getattr(self,"margin_maps",{}), precise_maps=getattr(self,"replay_precise",{}))
                m = None
            else:
                m = expected = (self.replay_precise[f["precise_id"]] if self.hybrid_version==3 else precise(*f["display_params"]))
                if self.hybrid_version==3:
                    self.last_precise=({"id":f["precise_id"],"params":f["display_params"]},m)
                    self.shown_maps,self.shown_ids=[],set()
            draw_record = dict(f)
            self._draw_present(draw_record, m, recorded_transform=True)
            rt = self.disp.read_render_target()
            bb = self.disp.read_back_buffer()
            pm = precise(*f["display_params"])
            checks.add(f, rt, bb, expected, pm, draw_record["bound_source_fresh_id"])
            if self.disp.present() != 0:
                raise RuntimeError("honesty DwmFlush failed")
        result = checks.finish(self.log_dir)
        result.update(replay=str(Path(replay_dir).resolve()), expected_frames=len(frames), records=records)
        self.meta["honesty"] = result
        (Path(self.log_dir) / "honesty.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        condition6 = {k: v for k, v in result.items() if k not in ("condition5", "quality")}
        (Path(self.log_dir) / "condition6.json").write_text(json.dumps(condition6, indent=1), encoding="utf-8")

    def run(self, events):
        """One loop for every mode. events = script events, or None for interactive use (until the window closes)."""
        glfw = self.glfw
        self.meta["t_warmup_start"] = time.perf_counter()
        # warm-up: compile both kernels; then the initial precise frame of the start parameters
        if self.state[2] >= limits.DEEP_BELOW:
            self.fast.fast_map(*self.state, precision="f32")
            warm = self.fast.fast_map(*self.state, precision="f64")
            if getattr(self, "hybrid_version", 1) == 3:
                self._warm_multi_draw(warm)
        self.frame("initial")
        warm_calls = dict(self.fast.calls)
        self.fast.calls.update({"fast": 0, "precise": 0, "f32": 0, "f64": 0, "precision": []})
        time.sleep(0.5)
        t0 = time.perf_counter()
        self.meta["t_playback_start"] = t0
        n = len(events) if events is not None else 0
        cbs, cb_state = None, {"pressed": False, "pos": (100.0, 40.0), "first_press_done": False}
        real_get_cursor_pos, gcp_calls = glfw.get_cursor_pos, []
        if self.mode == "callbacks":
            cbs = self._registered_callbacks()
            self.meta["registered_callbacks"] = {name: (cb is not None) for name, cb in cbs.items()}
            if not all(cbs.values()):
                raise RuntimeError(f"mouse callbacks not registered: {self.meta['registered_callbacks']}")
            cbs["key"] = self._registered_key_callback()
            self.meta["registered_key_callback"] = cbs["key"] is not None
            if cbs["key"] is None and any(ev["type"] == "key" for ev in events or []):
                raise RuntimeError("key callback not registered")

            def stand_in(win):
                """Deterministic glfw.get_cursor_pos for this replay: the virtual cursor; every call is logged."""
                gcp_calls.append(list(cb_state["pos"]))
                return cb_state["pos"]
            glfw.get_cursor_pos = stand_in
        self.replay = deep_view.ThreadedReplay(events, t0).start() if self.mode == 'threaded' else None
        try:
            self._loop(events, cbs, cb_state, t0, n)
            k = self._k
        finally:
            glfw.get_cursor_pos = real_get_cursor_pos
            if self.replay is not None:
                self.replay.close()
        self.meta["get_cursor_pos_calls"] = gcp_calls
        self._finish(warm_calls, k)

    def _warm_multi_draw(self, m):
        """Warm-up of the B3 multi-candidate draw (65-layer source array, multi-candidate program) before playback:
        its first use otherwise falls in the first replacement gap (about 0.1 s once). Draws into the render target
        only (the initial frame overwrites it before the first present); no frame, record or count."""
        self.disp.draw_candidates(self.state, [({"id": "warm-up", "params": list(self.state), "part": "screen",
                                                  "margin": [0, 0]}, m)], version=3)
        self.disp.source_layers, self.disp.bound_candidate_ids = {}, []
        self.meta["warm_multi_draw"] = True

    def _loop(self, events, cbs, cb_state, t0, n):
        glfw = self.glfw
        k = 0
        self._k = 0
        while not glfw.window_should_close(self.disp.win):
            glfw.poll_events()
            if self.display[2] >= limits.DEEP_BELOW:
                self._poll_fresh()
            if self.display[2] >= limits.DEEP_BELOW and getattr(self,"hybrid_version",1) == 3:
                # Retire before deciding; submit after any new screen job below.
                self.margins.tick(self.job is not None, launch=False)
            now = time.perf_counter()
            if events is not None:
                if self.mode == 'threaded':
                    for ev in self.replay.drain():
                        self.on_input(ev)
                        k += 1
                    now = time.perf_counter()
                elif self.mode == "verify":
                    if k < n and self.drawn == self.applied:
                        self.on_input(events[k])
                        k += 1
                        now = time.perf_counter()
                else:
                    while k < n and t0 + events[k]["t"] <= now:
                        if cbs is not None:
                            self._deliver_via_callbacks(events[k], cbs, cb_state)
                        else:
                            self.on_input(events[k])
                        k += 1
                        # Refresh immediately before checking the next due event. The final
                        # read is also the decision time: ties are delivered before choosing P.
                        now = time.perf_counter()
            if self.display[2] < limits.DEEP_BELOW and self.state[2] < limits.DEEP_BELOW:
                self.frame('fast', now)
                final = self.deep_controller.final_for(self.hp)
                ready = final is not None or self.frames[-1].get('notice','').startswith('range:')
                if events is not None and k >= n and ready and now >= t0+self.meta.get('script_duration',0.) \
                        and now-(self.last_input_t or t0) >= STOP_DELAY and self.display == self.state:
                    break
                continue
            unsettled = [v.hex() for v in self.display] != [v.hex() for v in self.state]
            active_warp = (self.hybrid_mode == "warped" and self.last_kind == "fast"
                           and (self.job is not None or now - self.last_input_t < STOP_DELAY))
            active_fresh = (self.hybrid_mode == "fresh-only" and self.last_kind == "fast"
                            and now - self.last_input_t < STOP_DELAY)
            if self.applied > self.drawn or unsettled or active_warp or active_fresh:
                self.frame("fast", now)
            elif self.last_kind == "fast" and self.job is None and now - self.last_input_t >= STOP_DELAY:
                self.frame("precise", now)
            elif getattr(self, 'limit_until', None) is not None and now >= self.limit_until                     and self.frames[-1].get('notice') == 'limit':
                self.limit_until = None
                self.frame('precise', now)
            elif events is not None and k >= n and self.last_kind in ("precise", "initial") and self.applied >= n - 1 \
                    and now >= t0 + self.meta.get("script_duration", 0.0):
                break
            else:
                time.sleep(0.0005)
            if self.display[2] >= limits.DEEP_BELOW and getattr(self,"hybrid_version",1) == 3:
                self.margins.tick(self.job is not None, launch=self.hybrid_mode == "warped")
        self._k = k

    def _finish(self, warm_calls, k):
        self._drain_fresh()
        if getattr(self,"hybrid_version",1)==3:
            self.margins.drain_active()
        self._finish_archive()
        self.meta["window_state_at_end"] = self.disp.window_state()
        if getattr(self, 'hp', None) is not None:
            self.meta['end_hp'] = [str(v) for v in self.hp[:2]]
        self.meta['end_state'] = list(self.state)
        self.meta.update({"fast_calls_playback": self.fast.calls["fast"], "precise_calls_playback": self.fast.calls["precise"],
                          "precision_calls_playback": list(self.fast.calls["precision"]),
                          "warmup_calls": warm_calls, "skipped_same_params": self.skipped_same_params,
                          "pool_max_bytes": int(self.pool_max), "events_delivered": k,
                          "input_t": {str(i): t for i, t in self.input_t.items()}, "t_end": time.perf_counter(),
                          "inputs": self.input_log, "press_sources": self.press_sources,
                          "gpu_used_mib_at_end": gpu_used_mib()})

    def save(self):
        self._drain_fresh()
        if getattr(self,"hybrid_version",1)==3:
            self.margins.drain_active()
        self._finish_archive()
        d = Path(self.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        import numpy as np
        if getattr(self, 'deep_controller', None) is not None:
            self.deep_controller.poll()
            self.deep_controller.save_final_archive(d)
            (d/'deep_jobs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.deep_controller.records),encoding='utf-8')
        (d/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in getattr(self, 'input_log', [])),encoding='utf-8')  # CPU fixtures built without __init__
        (d / "fresh.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.fresh), encoding="utf-8")
        np.save(d / "fresh_maps.npy", np.stack(self.fresh_bits) if self.fresh_bits else np.empty((0, 259200), np.uint8))
        if getattr(self,"hybrid_version",1)==3:
            np.savez(d / "fresh_margins.npz",**{str(r["id"]):self.map_archive.bits(("margin",r["id"])) for r in self.fresh if "margin_archive" in r})
            np.savez(d / "precise_maps.npz",**{key[1]:entry["bits"] for key,entry in self.map_archive.entries.items() if key[0]=="precise"})
        if any(key[0] == 'deep' for key in self.map_archive.entries):
            np.savez(d / 'deep_maps.npz', **{key[1]: entry['bits'] for key, entry in self.map_archive.entries.items() if key[0] == 'deep'})
        if getattr(self, 'notice_targets', None):
            np.savez(d / 'notice_targets.npz', **self.notice_targets)
        presents = [dict(r, kind=r["present_kind"]) for r in self.frames if r["kind"] == "fast"]
        (d / "presents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in presents), encoding="utf-8")
        with open(d / "frames.jsonl", "w", encoding="utf-8") as fh:
            for r in self.frames:
                fh.write(json.dumps(r) + "\n")
        if self.recorded:  # one bit per window pixel (rows bottom -> top as read), frame k = frames.jsonl line k
            import numpy as np
            self.meta["recorded_frames"] = len(self.recorded)
            self.meta["recorded_non_binary_pixels"] = int(sum(np.count_nonzero((f != 0) & (f != 255))
                                                              for f in self.recorded))
            np.save(d / "window_frames_bits.npy", np.stack([np.packbits(f > 127) for f in self.recorded]))
        (d / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        if self.media:
            from PIL import Image
            frames = [Image.fromarray(s["small"]).convert("P") for s in self.media]
            durations = [max(20, int(round(1000 * (b["t"] - a["t"])))) for a, b in zip(self.media, self.media[1:])]
            frames[0].save(d / "preview.gif", save_all=True, append_images=frames[1:], duration=durations + [2000],
                           loop=0, optimize=False)
            mid = [s for s in self.media if s["kind"] == "fast" and s["full"] is not None]
            fin = [s for s in self.media if s["kind"] == "precise" and s["full"] is not None]
            if mid:
                Image.fromarray(mid[0]["full"]).save(d / "during_manipulation.png")
            if fin:
                Image.fromarray(fin[-1]["full"]).save(d / "after_precise.png")
            (d / "media.json").write_text(json.dumps([{k: s[k] for k in ("t", "seq", "kind")} for s in self.media]),
                                          encoding="utf-8")

    def close(self):
        # Teardown only: hold CUDA buffers until their work finishes, even on early window close.
        try:
            self._drain_fresh()
            if getattr(self,"hybrid_version",1)==3:
                self.margins.drain_active()
            self._archive().close(max((f["t_present"] for f in self.frames),default=0.))
        finally:
            try:
                if getattr(self, 'deep_controller', None) is not None:
                    self.deep_controller.close()
            finally:
                self.disp.close()


def gpu_free_mib():
    """(chosen GPU entry, its free MiB). Raises device.GpuChoiceError for a bad index or a failed nvidia-smi."""
    from mytet_cuda import device
    g = device.choose()
    return g, device.memory(g["uuid"])["free_mib"]


def gpu_room_reason():
    """None if the chosen GPU has >= 2,048 MiB free, else the reason (also for a bad index or a failed nvidia-smi)."""
    from mytet_cuda import device
    try:
        g, free = gpu_free_mib()
    except device.GpuChoiceError as e:
        return str(e)
    if free < device.MIN_FREE_MIB:
        return f"GPU {g['index']} ({g['name']}) free memory {free} MiB < 2,048 MiB"
    return None


def preflight_interactive():
    """A person starts the viewer at the PC (python -m viewer --interactive / run_viewer.cmd): open at once.
    Only the GPU rule stays: the chosen GPU has >= 2,048 MiB free (mytet_cuda.gpu checks the device again).
    No idle and no full-screen check. Returns None if fine, else the reason (the caller does not wait)."""
    return gpu_room_reason()


def preflight():
    """Script runs (tests, measurements, recordings): before any window or GPU context — chosen GPU free >= 2,048 MiB,
    FreeVirtualMemory >= 8 GB, no full-screen application in front, user idle >= 5 min.
    Returns None if all fine, else the reason."""
    import desktop
    why = gpu_room_reason()
    if why:
        return why
    mem = desktop.free_virtual_kb()
    if mem < 8_388_608:
        return f"FreeVirtualMemory {mem} KB < 8,388,608 KB"
    if desktop.fullscreen_app_in_front():
        return "a full-screen application is in front"
    if desktop.idle_seconds() < desktop.MIN_IDLE_S:
        return f"user active {desktop.idle_seconds():.0f} s ago (< 300 s)"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m viewer")
    ap.add_argument("--script")
    ap.add_argument("--mode", choices=["timed", "verify", "media", "callbacks", "honesty", "threaded"], default="timed")
    ap.add_argument("--readback", choices=["none", "precise", "all", "record"], default="none")
    ap.add_argument("--log")
    ap.add_argument("--replay", help="timed hybrid run to redraw in honesty mode")
    ap.add_argument("--interactive", action="store_true",
                    help="open the window for mouse use (no script), at once (only the GPU-memory rule applies)")
    ap.add_argument("--gpu", metavar="N", help="nvidia-smi index of the GPU (default: $MYTET_GPU, else 0)")
    a = ap.parse_args(argv)
    if not a.script and not a.interactive:
        ap.error("give --script FILE or --interactive")
    if a.script and a.interactive:
        ap.error("--script and --interactive exclude each other")
    if a.mode == "honesty" and (not a.script or not a.replay or not a.log):
        ap.error("honesty requires --script, --replay and --log")
    if a.gpu is not None:
        from mytet_cuda import device
        try:
            device.use(a.gpu)  # checked before any window or CUDA
        except device.GpuChoiceError as e:
            print(f"not opening the window: {e}", file=sys.stderr)
            return 2
    # a person at the PC -> open now; script runs (tests, measurements, recordings) keep every protection rule
    why = preflight() if a.script else preflight_interactive()
    if why:
        print(f"not opening the window: {why}", file=sys.stderr)
        return 3
    if a.script:
        data = json.loads(Path(a.script).read_text(encoding="utf-8"))
        start, events = data["start"], data["events"]
        readback = "all" if a.mode == "verify" else a.readback
    else:
        start, events, readback = (0.0, 0.0, 5.0), None, "none"
    hp_kw = {'start_hp': data['start_hp']} if a.script and 'start_hp' in data else {}
    if a.script and 'hud' in data:  # scripts draw no HUD unless they ask for it (keeps every recorded window sha)
        hp_kw['hud'] = bool(data['hud'])
    v = Viewer(start, a.mode if a.script else "interactive", readback, a.log, **hp_kw)
    try:
        if events is not None:
            v.meta["script"] = a.script
            v.meta["script_duration"] = data.get("duration", 0.0)
        if a.mode == "honesty":
            v.run_honesty(a.replay)
        else:
            v.run(events)
    finally:
        try:
            if a.log:
                v.save()
        finally:
            v.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
