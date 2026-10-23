"""Number-key location presets (keys 1-4) and the start view (key 0) of the viewer, plus its one-line HUD.

The values live in the tracked file viewer/presets.json (80-digit centres of the four
pattern locations, also in tests/data/deepzoom2_positions.json). A preset sets the target state exactly like a script that
starts with `start_hp`: float centre = float(string), high-precision centre = the string, eps = the stored float.

HUD (drawn over the window only, never into the 1920 x 1080 render target, see render.Display.set_hud), one line
= name field + DEPTH_SEP + depth field:
  name field:  no preset active (start, or after key 0) -> GUIDE
               preset active                            -> its name, e.g. "m37"
               preset active and the view was dragged   -> name + MOVED_SUFFIX (the wheel keeps the centre)
  depth field: always (shallow views too) -> "eps 3.2e-17 · N 587" = the eps of the view the inputs asked for (the
               target state) and the max_iter its picture is drawn with (deep.limits.max_iter_for; 500 at eps >= 1e-13)
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from deep import limits
from viewer import deep_state

FILE = Path(__file__).resolve().parent / 'presets.json'
GUIDE = '1~4: 무늬 위치, 0: 처음으로'
MOVED_SUFFIX = ' · 이동함'
DEPTH_SEP = ' · '
HUD_FONT = os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf")  # Windows Korean font (Malgun Gothic)
HUD_SIZE = 18          # px, window pixels (the window is the render target shrunk by 2)
HUD_MARGIN = 10        # px from the top and right window edges
HUD_PAD = 6            # px of opaque black box around the text


@lru_cache(maxsize=1)
def table():
    """{key: entry} for '0'..'4' from presets.json; entry has key, name (None for '0'), x0, y0, eps (float)."""
    data = json.loads(FILE.read_text(encoding='utf-8'))
    out = {}
    for entry in [data['start'], *data['presets']]:
        eps = float.fromhex(entry['eps_hex'])
        if eps != float(entry['eps']):
            raise ValueError(f"preset {entry['key']}: eps {entry['eps']} != eps_hex {entry['eps_hex']}")
        out[entry['key']] = dict(entry, eps=eps)
    if sorted(out) != ['0', '1', '2', '3', '4']:
        raise ValueError(f'presets.json keys {sorted(out)} != 0-4')
    return out


def is_key(key):
    return key in table()


def states(key):
    """(float state, high-precision state) for a preset key; KeyError for any other key."""
    entry = table()[key]
    return deep_state.initial_states([0.0, 0.0, entry['eps']], [entry['x0'], entry['y0']])


def name(key):
    return table()[key]['name']


def depth_text(eps):
    """"eps 3.2e-17 · N 587": two significant digits of eps and the max_iter of that eps."""
    return f'eps {eps:.1e} · N {limits.max_iter_for(eps)}'


def hud_text(active, moved, eps):
    name_field = GUIDE if active is None else active + (MOVED_SUFFIX if moved else '')
    return name_field + DEPTH_SEP + depth_text(eps)


@lru_cache(maxsize=16)
def hud_image(text, width=960, height=540):
    """RGBA uint8 [height, width, 4], rows top -> bottom. Binary on purpose: every pixel is either fully transparent
    (alpha 0) or fully opaque (alpha 255; white text without anti-aliasing on a black box), so the window shows exactly
    the HUD pixels or exactly the picture and the result can be checked bit for bit."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(HUD_FONT, HUD_SIZE)
    probe = ImageDraw.Draw(Image.new('1', (1, 1)))
    probe.fontmode = '1'
    left, _, right, _ = probe.textbbox((0, 0), text, font=font)
    ascent, descent = font.getmetrics()  # fixed line height: the box does not change size with the glyphs
    box_right = width - HUD_MARGIN
    box_left = box_right - (right - left) - 2 * HUD_PAD
    box_top = HUD_MARGIN
    box_bottom = box_top + ascent + descent + 2 * HUD_PAD
    mask = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.fontmode = '1'
    draw.text((box_left + HUD_PAD - left, box_top + HUD_PAD), text, font=font, fill=1)
    text_px = np.asarray(mask, dtype=np.uint8) > 0
    out = np.zeros((height, width, 4), np.uint8)
    out[box_top:box_bottom, max(0, box_left):box_right] = (0, 0, 0, 255)
    out[text_px] = (255, 255, 255, 255)
    out.setflags(write=False)
    return out
