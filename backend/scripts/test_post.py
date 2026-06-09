"""
Simple test helper to POST to /api/analyze_chart with query and history (and optional image).
Run: python scripts/test_post.py
"""
import requests
import json
import os

API = os.environ.get('API_URL','http://127.0.0.1:8000/api/analyze_chart')

history = [{"role":"user","content":"Hello world"}]

files = {
    'query': (None, 'Test from script with image'),
    'history': (None, json.dumps(history)),
}

# By default create/use a tiny 1x1 png to trigger the multimodal path
img_path = os.path.join(os.path.dirname(__file__), 'test_image.png')
if not os.path.exists(img_path):
    # 1x1 transparent PNG
    import base64
    tiny_png_b64 = (
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/\n'
        'wQACfsD/Qa1wHgAAAAASUVORK5CYII='
    )
    with open(img_path, 'wb') as f:
        f.write(base64.b64decode(tiny_png_b64))

files['image_file'] = (os.path.basename(img_path), open(img_path, 'rb'), 'image/png')

print('Posting to', API)
resp = requests.post(API, files=files, stream=True)
print('Status code:', resp.status_code)
print('Response headers:', resp.headers)
print('Streamed response snippet:')
for idx, chunk in enumerate(resp.iter_lines()):
    if chunk:
        try:
            print(chunk.decode('utf-8'))
        except Exception:
            print(chunk)
    if idx>10:
        break

print('Done')