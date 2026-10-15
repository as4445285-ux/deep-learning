from PIL import Image, ImageDraw
import requests
import io
import json

img = Image.new('RGB', (512, 512), 'white')
d = ImageDraw.Draw(img)
for x0 in (130, 322):
    d.polygon([(x0, 130), (x0 + 30, 40), (x0 + 60, 130)], fill=(220, 220, 220))
for x, y in [(150, 160), (362, 160)]:
    d.ellipse([x - 40, y - 30, x + 40, y + 30], fill=(240, 240, 240))
d.ellipse([120, 180, 380, 400], fill=(240, 240, 240))
for x1, x2 in [(180, 70), (332, 440)]:
    d.line((x1, 250, x2, 220), fill=(0, 0, 0), width=2)
    d.line((x1, 250, x2, 230), fill=(0, 0, 0), width=2)
d.ellipse([238, 270, 274, 290], fill=(255, 140, 140))
d.ellipse([167, 215, 203, 255], fill=(0, 0, 0))
d.ellipse([309, 215, 345, 255], fill=(0, 0, 0))

buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
files = {'image': ('cat_test.png', buf.getvalue(), 'image/png')}
resp = requests.post(
    'http://localhost:5000/api/caption',
    files=files,
    data={'temperature': '1.0', 'max_len': '25'},
    timeout=180,
)
print('STATUS:', resp.status_code)
print(resp.text[:2000])
