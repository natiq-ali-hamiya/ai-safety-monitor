from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
print('Show knife to camera. Press Q to quit.')
while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame, conf=0.1, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 43: print(f'KNIFE! {conf:.2%}')
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()