from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WIDTH = 1200
HEIGHT = 630

BG = (17, 24, 39)        # dark slate
ACCENT = (37, 99, 235)   # blue
TEXT = (255, 255, 255)
SUBTLE = (156, 163, 175)

out_path = Path('web/static/og/solo-vs-pool-risk-first.png')
out_path.parent.mkdir(parents=True, exist_ok=True)

img = Image.new('RGB', (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

# Use default font if system fonts unavailable
try:
    title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 64)
    body_font = ImageFont.truetype('DejaVuSans.ttf', 36)
    small_font = ImageFont.truetype('DejaVuSans.ttf', 28)
except:
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

# Accent bar
draw.rectangle((0, 0, WIDTH, 16), fill=ACCENT)

# Title
draw.text((100, 150), "Solo vs Pool Mining", font=title_font, fill=TEXT)
draw.text((100, 230), "A Risk-First Framework", font=title_font, fill=TEXT)

# Metrics line
draw.text(
    (100, 350),
    "P(0 blocks)   •   P(loss)   •   Regret probability",
    font=body_font,
    fill=SUBTLE
)

# Footer branding
draw.text(
    (100, 520),
    "solo-odds.hefftools.dev",
    font=small_font,
    fill=SUBTLE
)

img.save(out_path)
print(f"Saved: {out_path}")