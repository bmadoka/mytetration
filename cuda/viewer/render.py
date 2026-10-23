"""OpenGL display path: bool map -> render target (1920 x 1080 RGBA8 texture in an FBO) -> window (960 x 540).

Picture rule (= original 16by9_H imshow(map.T, origin='lower'), black->white colormap):
    render-target pixel (px, py), px = 0..1919 left->right, py = 0..1079 bottom->top, is
    white (255, 255, 255, 255) if map[px, py] (diverged) else black (0, 0, 0, 255).
Window rule (nearest-neighbour down-scale by SCALE = 2, fixed sample position):
    window pixel (u, v), u = 0..959 left->right, v = 0..539 bottom->top  =  render-target pixel (2u, 2v).
glReadPixels returns rows bottom->top, so a read-back render target equals the array R[py, px].
"""

import ctypes

import glfw
import numpy as np
from OpenGL import GL

import params
import hybrid

SCALE = 2
WIN_W, WIN_H = params.NX // SCALE, params.NY // SCALE

_VS = """#version 330 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}"""
_FS_MAP = """#version 330 core
uniform sampler2D mapTex;   // R8, width 1080 (j = y index), height 1920 (i = x index)
uniform isampler1D ixTex;
uniform isampler1D iyTex;
uniform bool warped;
out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);           // p.x = px (x index), p.y = py (y index, from bottom)
    if (warped) {
        p = ivec2(texelFetch(ixTex, p.x, 0).r, texelFetch(iyTex, p.y, 0).r);
    }
    float v = texelFetch(mapTex, ivec2(p.y, p.x), 0).r;
    color = v > 0.5 ? vec4(1.0, 1.0, 1.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}"""
_FS_MULTI = """#version 330 core
uniform sampler2DArray sources;
uniform isampler2D xIndices;
uniform isampler2D yIndices;
uniform int count;
uniform int layers[65];
uniform ivec4 bounds[65];
uniform ivec2 origins[65];
out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 newest = ivec2(0);
    float v = 0.0;
    bool found = false;
    for (int c = 0; c < count; ++c) {
        ivec2 q = ivec2(texelFetch(xIndices, ivec2(p.x, c), 0).r,
                       texelFetch(yIndices, ivec2(p.y, c), 0).r);
        ivec4 b = bounds[c];
        if (c == 0) newest = clamp(q, ivec2(b.x,b.z), ivec2(b.y-1,b.w-1)) + origins[c];
        if (q.x >= b.x && q.x < b.y && q.y >= b.z && q.y < b.w) {
            q += origins[c];
            v = texelFetch(sources, ivec3(q.y, q.x, layers[c]), 0).r;
            found = true;
            break;
        }
    }
    if (!found) v = texelFetch(sources, ivec3(newest.y, newest.x, layers[0]), 0).r;
    color = v > 0.5 ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}"""
_FS_WIN = """#version 330 core
uniform sampler2D rtTex;
uniform int scale;
out vec4 color;
void main() {
    color = texelFetch(rtTex, ivec2(gl_FragCoord.xy) * scale, 0);
}"""
_FS_HUD = """#version 330 core
uniform sampler2D hudTex;   // RGBA8, window size, rows bottom -> top; alpha is 0 or 255 only
out vec4 color;
void main() {
    vec4 c = texelFetch(hudTex, ivec2(gl_FragCoord.xy), 0);
    if (c.a < 0.5) discard;  // transparent: the window keeps the picture pixel exactly
    color = vec4(c.rgb, 1.0);
}"""


def _program(vs, fs):
    def sh(src, kind):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(s))
        return s
    p = GL.glCreateProgram()
    GL.glAttachShader(p, sh(vs, GL.GL_VERTEX_SHADER))
    GL.glAttachShader(p, sh(fs, GL.GL_FRAGMENT_SHADER))
    GL.glLinkProgram(p)
    if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(p))
    return p


def chosen_gpu_name():
    """nvidia-smi name of the GPU chosen for CUDA (--gpu / MYTET_GPU; replaceable in tests)."""
    from mytet_cuda import device
    return device.choose()["name"]


def machine_rule_applies():
    """True when this machine's GPU rule (mytet_cuda.device, local_gpu.json) applies here (replaceable in tests)."""
    from mytet_cuda import device
    return device.machine_rule() is not None


def attached_display_adapters():
    """Adapters of every display attached to the desktop (replaceable in tests)."""
    import desktop
    return desktop.attached_display_adapters()


def renderer_device(renderer):
    """Device part of a GL_RENDERER string: "NVIDIA GeForce RTX 3090 Ti/PCIe/SSE2" -> "NVIDIA GeForce RTX 3090 Ti"."""
    return renderer.split("/")[0].strip()


class Display:
    def __init__(self, title="mytetration preview", visible=True, pos=(60, 60), on_top=False):
        # Which GPU an OpenGL context lands on is the driver's choice (GLFW cannot pick one; NVIDIA's «OpenGL rendering
        # GPU» setting, usually «Auto-select», which follows the display). Rule: right after the context is created,
        # before any drawing or allocation, the GL_RENDERER device must be exactly the GPU chosen for CUDA (same name
        # as nvidia-smi gives); otherwise the window and context are destroyed at once and the viewer refuses.
        self.gpu_name = chosen_gpu_name()
        if machine_rule_applies():
            # This machine's GPU rule also covers OpenGL: before any window or context exists, every display attached
            # to the desktop must be driven by the GPU chosen for CUDA (the context follows the display); a display on
            # another adapter, none listed, or a failed lookup refuses here.
            self.adapters = attached_display_adapters()
            other = [a for a in self.adapters if a != self.gpu_name]
            if not self.adapters or other:
                raise RuntimeError(f"displays attached to the desktop: {self.adapters!r}; a display not on the GPU chosen "
                                   f"for CUDA ({self.gpu_name!r}), or none - no window / OpenGL context created")
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)       # create hidden, show without activation below
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.FALSE)
        self.win = glfw.create_window(WIN_W, WIN_H, title, None, None)
        if not self.win:
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(self.win)
        try:
            r = GL.glGetString(GL.GL_RENDERER)  # first GL call on the new context
            self.renderer = r.decode(errors="replace") if r else ""  # NULL (query failed) -> "" -> refused below
        except Exception as e:  # the query itself failed: same cleanup and refusal
            self.renderer = f"<GL_RENDERER query failed: {type(e).__name__}>"
        if renderer_device(self.renderer) != self.gpu_name:
            glfw.make_context_current(None)
            glfw.destroy_window(self.win)
            glfw.terminate()
            raise RuntimeError(f"OpenGL renderer is {self.renderer!r}, not the GPU chosen for CUDA ({self.gpu_name!r}); "
                               "window and context destroyed")
        glfw.set_window_pos(self.win, *pos)
        monitor = glfw.get_window_monitor(self.win) or glfw.get_primary_monitor()
        video = glfw.get_video_mode(monitor)
        self.refresh_hz = float(video.refresh_rate)
        glfw.swap_interval(0)
        fb = glfw.get_framebuffer_size(self.win)
        if tuple(fb) != (WIN_W, WIN_H):
            raise RuntimeError(f"window framebuffer {fb} != {(WIN_W, WIN_H)}")
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        self.map_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.map_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R8, params.NY, params.NX, 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, None)
        self.rt_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.rt_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, params.NX, params.NY, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        for t in (self.map_tex, self.rt_tex):
            GL.glBindTexture(GL.GL_TEXTURE_2D, t)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.rt_tex, 0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("render-target FBO incomplete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.p_map = _program(_VS, _FS_MAP)
        self.index_tex = GL.glGenTextures(2)
        for tex, n in zip(self.index_tex, (params.NX, params.NY)):
            GL.glBindTexture(GL.GL_TEXTURE_1D, int(tex))
            GL.glTexImage1D(GL.GL_TEXTURE_1D, 0, GL.GL_R32I, n, 0, GL.GL_RED_INTEGER, GL.GL_INT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glUseProgram(self.p_map)
        for name, unit in (("mapTex", 0), ("ixTex", 1), ("iyTex", 2)):
            GL.glUniform1i(GL.glGetUniformLocation(self.p_map, name), unit)
        self.p_multi = _program(_VS, _FS_MULTI)
        self.sources_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.sources_tex)
        import fast
        self.max_candidates = 65
        self.texture_shape = (params.NX+2*fast.MX, params.NY+2*fast.MY)
        GL.glTexImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, GL.GL_R8, self.texture_shape[1], self.texture_shape[0], 65,
                        0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        self.multi_indices = GL.glGenTextures(2)
        for tex, n in zip(self.multi_indices, (params.NX, params.NY)):
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32I, n, 65, 0, GL.GL_RED_INTEGER, GL.GL_INT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        self.source_layers = {}
        self.bound_candidate_ids = []
        GL.glUseProgram(self.p_multi)
        for name, unit in (("sources", 0), ("xIndices", 1), ("yIndices", 2)):
            GL.glUniform1i(GL.glGetUniformLocation(self.p_multi, name), unit)
        self.p_win = _program(_VS, _FS_WIN)
        GL.glUseProgram(self.p_win)
        GL.glUniform1i(GL.glGetUniformLocation(self.p_win, "scale"), SCALE)
        if visible:
            glfw.show_window(self.win)  # FOCUS_ON_SHOW = false: shown without activation (SW_SHOWNA)
        self.hwnd = glfw.get_win32_window(self.win)
        self.topmost_set = None
        if visible and on_top:
            self.keep_on_top_without_focus()

    def upload_source(self, divergence_map):
        """Replace the full source only on completion of a fresh map (or precise draw)."""
        m = np.ascontiguousarray(divergence_map, dtype=np.bool_).view(np.uint8) * np.uint8(255)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.map_tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, params.NY, params.NX, GL.GL_RED, GL.GL_UNSIGNED_BYTE, m)

    def draw(self, divergence_map):
        """Fresh/precise path, retaining the original picture and window samples."""
        self.upload_source(divergence_map)
        self._draw(False)

    def draw_warped(self, display, source, transform=None):
        ix, iy, scale, offset = hybrid.warp_indices(display, source, transform=transform)
        for unit, tex, axis in zip((1, 2), self.index_tex, (ix, iy)):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_1D, int(tex))
            GL.glTexSubImage1D(GL.GL_TEXTURE_1D, 0, 0, len(axis), GL.GL_RED_INTEGER, GL.GL_INT, axis)
        self._draw(True)
        return scale, offset

    def draw_deep(self, divergence_map, display, source, stride, source_id):
        """Deep-only draw: HP axes, coarse samples repeated from their grid anchors."""
        from viewer.deep_view import warp_indices
        if getattr(self, '_deep_texture_id', None) != source_id:
            expanded = (divergence_map if stride == 1 else
                        np.repeat(np.repeat(divergence_map, stride, axis=0), stride, axis=1)[:1920, :1080])
            self.upload_source(expanded)
            self._deep_texture_id = source_id
        ix, iy, transform = warp_indices(display, source)
        transform['stride'] = stride
        for unit, tex, axis in zip((1, 2), self.index_tex, (ix, iy)):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_1D, int(tex))
            GL.glTexSubImage1D(GL.GL_TEXTURE_1D, 0, 0, len(axis), GL.GL_RED_INTEGER, GL.GL_INT, axis)
        self._draw(True)
        return transform

    def hold_deep(self):
        self._draw_window()

    def draw_candidates(self, display, candidates, transforms=None, version=2):
        """Exact newest-first GL selection from at most 64 extended maps and one precise picture."""
        ids = [r["id"] for r, _ in candidates]
        if not 1 <= len(ids) <= 65 or len(set(ids)) != len(ids):
            raise ValueError("expected 1..65 distinct candidate maps")
        plan = hybrid.candidate_plan3 if version==3 else hybrid.candidate_plan
        axes, records, counts = plan(display, [r for r, _ in candidates], transforms)
        keys = [(r["id"],r.get("part","screen")) for r,_ in candidates]
        self.source_layers = {i: slot for i, slot in self.source_layers.items() if i in keys}
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.sources_tex)
        for (rec, m),ident in zip(candidates,keys):
            if ident not in self.source_layers:
                slot = next(i for i in range(65) if i not in self.source_layers.values())
                data = np.ascontiguousarray(m, dtype=np.bool_).view(np.uint8) * np.uint8(255)
                GL.glTexSubImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, slot, m.shape[1], m.shape[0], 1,
                                  GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)
                self.source_layers[ident] = slot
        for unit, tex, component in zip((1, 2), self.multi_indices, (0, 1)):
            data = np.ascontiguousarray([pair[component] for pair in axes], dtype=np.int32)
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, data.shape[1], len(ids),
                              GL.GL_RED_INTEGER, GL.GL_INT, data)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, params.NX, params.NY)
        GL.glUseProgram(self.p_multi)
        GL.glUniform1i(GL.glGetUniformLocation(self.p_multi, "count"), len(ids))
        GL.glUniform1iv(GL.glGetUniformLocation(self.p_multi, "layers[0]"), len(ids),
                         np.asarray([self.source_layers[i] for i in keys], dtype=np.int32))
        margins=[r.get("margin",(0,0)) for r,_ in candidates]
        GL.glUniform4iv(GL.glGetUniformLocation(self.p_multi,"bounds[0]"),len(ids),
                        np.asarray([(-mx,1920+mx,-my,1080+my) for mx,my in margins],dtype=np.int32))
        GL.glUniform2iv(GL.glGetUniformLocation(self.p_multi,"origins[0]"),len(ids),np.asarray(margins,dtype=np.int32))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        self.bound_candidate_ids = list(ids)
        self._draw_window()
        return records, counts

    def _draw(self, warped):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.map_tex)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, params.NX, params.NY)
        GL.glUseProgram(self.p_map)
        GL.glUniform1i(GL.glGetUniformLocation(self.p_map, "warped"), int(warped))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        self._draw_window()

    def _draw_window(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, WIN_W, WIN_H)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.rt_tex)
        GL.glUseProgram(self.p_win)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        if getattr(self, "hud_text", None):
            self._draw_hud()

    def set_hud(self, text):
        """One line of text over the window (None = no HUD). Window only: the render target is never touched."""
        self.hud_text = text

    def _draw_hud(self):
        from viewer import presets
        if getattr(self, "p_hud", None) is None:
            self.p_hud = _program(_VS, _FS_HUD)
            GL.glUseProgram(self.p_hud)
            GL.glUniform1i(GL.glGetUniformLocation(self.p_hud, "hudTex"), 0)
            self.hud_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.hud_tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, WIN_W, WIN_H, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            self.hud_uploaded = None
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hud_tex)
        if self.hud_uploaded != self.hud_text:
            image = np.ascontiguousarray(presets.hud_image(self.hud_text, WIN_W, WIN_H)[::-1])  # GL rows bottom -> top
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, WIN_W, WIN_H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, image)
            self.hud_uploaded = self.hud_text
        GL.glUseProgram(self.p_hud)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def present(self):
        """Swap, wait for the GPU, then wait for the next DWM composition (DwmFlush). The return time is the
        frame's present time: the frame has been handed to the compositor and a composition pass followed it, so the
        next frame cannot replace it before it was composed (one composition per frame at most, so presents are
        capped at the display refresh)."""
        glfw.swap_buffers(self.win)
        GL.glFinish()
        return ctypes.windll.dwmapi.DwmFlush() & 0xFFFFFFFF  # HRESULT; 0 = S_OK (the caller logs and judges it)

    def keep_on_top_without_focus(self):
        """Script runs: keep the window above others so its pixels are on screen, without activating it."""
        import ctypes.wintypes as wt
        swp = ctypes.windll.user32.SetWindowPos
        swp.argtypes = [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        swp.restype = wt.BOOL
        flags = 0x0001 | 0x0002 | 0x0010 | 0x0040  # NOSIZE | NOMOVE | NOACTIVATE | SHOWWINDOW
        # HWND_TOPMOST = (HWND)-1 must be passed as a full pointer-sized value (a plain int is truncated to 32 bits)
        self.topmost_set = bool(swp(wt.HWND(self.hwnd), wt.HWND(-1), 0, 0, 0, 0, flags))

    def window_state(self):
        """Visible, not minimised, topmost flag, foreground window is not the viewer."""
        u = ctypes.windll.user32
        u.GetWindowLongW.restype = ctypes.c_long
        ex = u.GetWindowLongW(self.hwnd, -20)  # GWL_EXSTYLE
        return {"visible": bool(u.IsWindowVisible(self.hwnd)), "minimised": bool(u.IsIconic(self.hwnd)),
                "topmost": bool(ex & 0x00000008), "foreground_is_viewer": int(u.GetForegroundWindow()) == int(self.hwnd)}

    def read_render_target(self):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        buf = GL.glReadPixels(0, 0, params.NX, params.NY, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        return np.frombuffer(buf, dtype=np.uint8).reshape(params.NY, params.NX, 4).copy()

    def read_back_buffer(self):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        buf = GL.glReadPixels(0, 0, WIN_W, WIN_H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(buf, dtype=np.uint8).reshape(WIN_H, WIN_W, 3).copy()

    def close(self):
        glfw.destroy_window(self.win)
        glfw.terminate()
