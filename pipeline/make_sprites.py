"""Generate the mod's sprites and .anm2 files.

Everything is drawn here rather than copied from the game, so the mod ships
self-contained and does not depend on any vanilla resource path.

  resources/gfx/familiar_flybrain.png       4-frame wingbeat, top-down, facing up
  resources/gfx/familiar_flybrain.anm2
  resources/gfx/items/collectibles/collectibles_flybrain.png

Run:  python pipeline/make_sprites.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
GFX = ROOT / "mod/resources/gfx"

# palette
OUTLINE = (16, 12, 20, 255)
BODY_D = (40, 34, 46, 255)
BODY_L = (74, 66, 84, 255)
EYE = (196, 46, 46, 255)
EYE_HI = (255, 150, 150, 255)
WING = (206, 226, 240, 110)
WING_E = (150, 180, 205, 170)
NEURON = (120, 240, 255, 255)
NEURON_D = (40, 150, 190, 255)
GLOW = (90, 200, 230, 90)

CELL = 32


def fly_frame(wing_phase: float) -> Image.Image:
    """One 32x32 top-down fly, head pointing up (-Y)."""
    im = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = 16, 17

    # wings sweep between swept-back and spread
    spread = 3.5 + 4.5 * wing_phase
    for sgn in (-1, 1):
        x0 = cx + sgn * 1
        x1 = cx + sgn * (1 + spread)
        box = sorted([x0, x1])
        d.ellipse([box[0], cy - 8, box[1], cy + 2], fill=WING, outline=WING_E)

    # abdomen
    d.ellipse([cx - 4, cy - 1, cx + 3, cy + 8], fill=BODY_D, outline=OUTLINE)
    for y in (cy + 2, cy + 5):
        d.line([cx - 3, y, cx + 2, y], fill=OUTLINE)

    # thorax
    d.ellipse([cx - 4, cy - 7, cx + 3, cy + 1], fill=BODY_L, outline=OUTLINE)

    # the connectome: a few bright nodes wired together on the thorax
    nodes = [(cx - 2, cy - 6), (cx + 1, cy - 5), (cx - 1, cy - 3), (cx + 2, cy - 2)]
    for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
        d.line([nodes[a], nodes[b]], fill=NEURON_D)
    for nx, ny in nodes:
        d.point((nx, ny), fill=NEURON)

    # head + compound eyes
    d.ellipse([cx - 3, cy - 11, cx + 2, cy - 6], fill=BODY_D, outline=OUTLINE)
    d.point((cx - 2, cy - 10), fill=EYE)
    d.point((cx + 1, cy - 10), fill=EYE)
    d.point((cx - 2, cy - 9), fill=EYE_HI)
    d.point((cx + 1, cy - 9), fill=EYE_HI)

    # legs
    for sgn in (-1, 1):
        d.line([cx + sgn * 3, cy - 2, cx + sgn * 6, cy + 1], fill=OUTLINE)
        d.line([cx + sgn * 3, cy + 1, cx + sgn * 6, cy + 4], fill=OUTLINE)
    return im


def make_familiar():
    frames = [fly_frame(p) for p in (0.0, 0.55, 1.0, 0.55)]
    sheet = Image.new("RGBA", (CELL * len(frames), CELL), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, (i * CELL, 0))
    (GFX).mkdir(parents=True, exist_ok=True)
    sheet.save(GFX / "familiar_flybrain.png")
    return len(frames)


def make_collectible():
    """32x32 pedestal / HUD icon: a fly with a lit-up brain."""
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = 16, 18

    for sgn in (-1, 1):
        box = sorted([cx + sgn * 2, cx + sgn * 13])
        d.ellipse([box[0], cy - 9, box[1], cy + 3], fill=WING, outline=WING_E)

    d.ellipse([cx - 6, cy - 2, cx + 5, cy + 12], fill=BODY_D, outline=OUTLINE)
    for y in (cy + 2, cy + 6, cy + 9):
        d.line([cx - 5, y, cx + 4, y], fill=OUTLINE)
    d.ellipse([cx - 6, cy - 9, cx + 5, cy + 1], fill=BODY_L, outline=OUTLINE)

    # brain: a glowing two-lobed network sitting on the head
    d.ellipse([cx - 8, cy - 19, cx + 7, cy - 7], fill=GLOW)
    d.ellipse([cx - 7, cy - 18, cx + 6, cy - 8], fill=(58, 48, 70, 255), outline=OUTLINE)
    nodes = [(cx - 4, cy - 15), (cx, cy - 16), (cx + 3, cy - 14),
             (cx - 3, cy - 11), (cx + 1, cy - 12), (cx + 4, cy - 10),
             (cx - 1, cy - 9)]
    edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (3, 6), (4, 6), (5, 6)]
    for a, b in edges:
        d.line([nodes[a], nodes[b]], fill=NEURON_D)
    for nx, ny in nodes:
        d.point((nx, ny), fill=NEURON)

    d.point((cx - 4, cy - 6), fill=EYE)
    d.point((cx + 3, cy - 6), fill=EYE)
    d.point((cx - 4, cy - 5), fill=EYE_HI)
    d.point((cx + 3, cy - 5), fill=EYE_HI)

    out = ROOT / "mod/resources/gfx/items/collectibles"
    out.mkdir(parents=True, exist_ok=True)
    im.save(out / "collectibles_flybrain.png")


PIXEL_ANM2 = """<?xml version="1.0" encoding="UTF-8"?>
<AnimatedActor>
\t<Info CreatedBy="fly-brain mod" CreatedOn="pipeline/make_sprites.py" Version="1" Fps="30"/>
\t<Content>
\t\t<Spritesheets>
\t\t\t<Spritesheet Path="pixel.png" Id="0"/>
\t\t</Spritesheets>
\t\t<Layers>
\t\t\t<Layer Name="px" Id="0" SpritesheetId="0"/>
\t\t</Layers>
\t\t<Nulls/>
\t\t<Events/>
\t</Content>
\t<Animations DefaultAnimation="Idle">
\t\t<Animation Name="Idle" FrameNum="1" Loop="true">
\t\t\t<RootAnimation>
\t\t\t\t<Frame XPosition="0" YPosition="0" Delay="1" Visible="true" XScale="100" YScale="100" RedTint="255" GreenTint="255" BlueTint="255" AlphaTint="255" RedOffset="0" GreenOffset="0" BlueOffset="0" Rotation="0" Interpolated="false"/>
\t\t\t</RootAnimation>
\t\t\t<LayerAnimations>
\t\t\t\t<LayerAnimation LayerId="0" Visible="true">
\t\t\t\t\t<Frame XPosition="0" YPosition="0" XPivot="0" YPivot="0" XCrop="0" YCrop="0" Width="4" Height="4" XScale="100" YScale="100" Delay="1" Visible="true" RedTint="255" GreenTint="255" BlueTint="255" AlphaTint="255" RedOffset="0" GreenOffset="0" BlueOffset="0" Rotation="0" Interpolated="false"/>
\t\t\t\t</LayerAnimation>
\t\t\t</LayerAnimations>
\t\t\t<NullAnimations/>
\t\t\t<Triggers/>
\t\t</Animation>
\t</Animations>
</AnimatedActor>
"""


def make_pixel():
    """A 4x4 white square plus its anm2.

    The Isaac API has no primitive for drawing a rectangle, so the HUD tints
    and scales this one sprite to draw every bar and graph column.  The pivot
    is at (0,0) so the sprite grows right and down from its render position.
    """
    out = ROOT / "mod/resources/gfx/ui"
    out.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(out / "pixel.png")
    (out / "pixel.anm2").write_text(PIXEL_ANM2, encoding="utf-8")


ANM2 = """<?xml version="1.0" encoding="UTF-8"?>
<AnimatedActor>
\t<Info CreatedBy="fly-brain mod" CreatedOn="pipeline/make_sprites.py" Version="1" Fps="30"/>
\t<Content>
\t\t<Spritesheets>
\t\t\t<Spritesheet Path="familiar_flybrain.png" Id="0"/>
\t\t</Spritesheets>
\t\t<Layers>
\t\t\t<Layer Name="body" Id="0" SpritesheetId="0"/>
\t\t</Layers>
\t\t<Nulls/>
\t\t<Events/>
\t</Content>
\t<Animations DefaultAnimation="Idle">
{animations}
\t</Animations>
</AnimatedActor>
"""

ANIM = """\t\t<Animation Name="{name}" FrameNum="{total}" Loop="true">
\t\t\t<RootAnimation>
\t\t\t\t<Frame XPosition="0" YPosition="0" Delay="{total}" Visible="true" XScale="100" YScale="100" RedTint="255" GreenTint="255" BlueTint="255" AlphaTint="255" RedOffset="0" GreenOffset="0" BlueOffset="0" Rotation="0" Interpolated="false"/>
\t\t\t</RootAnimation>
\t\t\t<LayerAnimations>
\t\t\t\t<LayerAnimation LayerId="0" Visible="true">
{frames}
\t\t\t\t</LayerAnimation>
\t\t\t</LayerAnimations>
\t\t\t<NullAnimations/>
\t\t\t<Triggers/>
\t\t</Animation>"""

FRAME = ('\t\t\t\t\t<Frame XPosition="0" YPosition="0" XPivot="16" YPivot="16" '
         'XCrop="{crop}" YCrop="0" Width="32" Height="32" XScale="100" YScale="100" '
         'Delay="{delay}" Visible="true" RedTint="255" GreenTint="255" BlueTint="255" '
         'AlphaTint="255" RedOffset="0" GreenOffset="0" BlueOffset="0" Rotation="0" '
         'Interpolated="false"/>')


def make_anm2(nframes: int):
    anims = []
    for name, delay in (("Idle", 2), ("Fly", 1)):
        frames = "\n".join(FRAME.format(crop=i * CELL, delay=delay) for i in range(nframes))
        anims.append(ANIM.format(name=name, total=delay * nframes, frames=frames))
    (GFX / "familiar_flybrain.anm2").write_text(
        ANM2.format(animations="\n".join(anims)), encoding="utf-8")


if __name__ == "__main__":
    n = make_familiar()
    make_collectible()
    make_anm2(n)
    make_pixel()
    print(f"wrote {n}-frame familiar sheet, anm2 and collectible icon under mod/resources/gfx/")
