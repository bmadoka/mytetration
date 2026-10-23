"""Binary 1920x1080 notices; image (r,col) becomes map[col,1079-r]."""
from functools import lru_cache
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf")  # Windows Korean font (Malgun Gothic)
LIMIT_TEXT = '더 깊이는 지원하지 않음 (하한 L = 1e-40)'
BAND_HEIGHT = 96


def range_text(reason):
    return f'계산 범위 이탈: {reason} — 이 화면은 계산하지 않음'


@lru_cache(maxsize=32)
def text_mask(text):
    image = Image.new('1', (1920, 1080), 0)
    draw = ImageDraw.Draw(image)
    draw.text((24, 22), text, font=ImageFont.truetype(FONT, 32), fill=1)
    # Canonical 0/1 bools: np.asarray(mode '1' image, dtype=bool) keeps the 255 bytes, and the
    # renderer's .view(np.uint8) * 255 then wraps them to 1 (black). Compare as uint8 instead.
    mask = (np.asarray(image, dtype=np.uint8) > 0)[::-1].T.copy()
    mask.setflags(write=False)
    return mask


def notice_map(base_map_or_None, text):
    if base_map_or_None is None:
        return text_mask(text).copy()
    out = np.asarray(base_map_or_None, dtype=bool).copy()
    if out.shape != (1920, 1080):
        raise ValueError('notice requires bool[1920,1080]')
    out[:, 1080-BAND_HEIGHT:] = False
    out |= text_mask(text)
    return out
