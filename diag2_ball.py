import json, cv2, numpy as np
from core.ball_eval import _build_detector
cfg = json.load(open('config-side1.json'))
det = _build_detector(cfg)
cap = cv2.VideoCapture(cfg['source'])
maxes = []
while True:
    ok, f = cap.read()
    if not ok: break
    det.detect(f)
    if len(det._buffer) >= det.in_frames:
        maxes.append(float(np.nanmax(det._infer())))
cap.release()
m = np.array(maxes)
print("frames scored:", len(m))
print("heatmap_max: global_max=%.3f  mean=%.3f  p99=%.3f" % (m.max(), m.mean(), np.percentile(m,99)))
for t in (0.5,0.4,0.3,0.25,0.2,0.15,0.1):
    print(f"  frames with peak >= {t}: {(m>=t).sum()}")
