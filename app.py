from models import load_custom_model, load_coco_model
import cv2
import random

def draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.6
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)[0]

   end_x = pos[0] + txt_size[0] + margin
   end_y = pos[1] - txt_size[1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, (pos[0], pos[1] - 5), font_face, scale, color, 1, cv2.LINE_AA)


def generate_colors(num_classes):
    random.seed(42)
    return {class_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_id in range(num_classes)}

model = load_custom_model()
model_coco = load_coco_model()

num_classes = len(model.names)
class_colors = generate_colors(num_classes)

cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Brak dostępu do kamery.")
    exit()

while True:
    ret, frame = cap.read()
    prediction = model(frame)

    for p in prediction:
        boxes = p.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id=int(box.cls[0])
            confidence = box.conf[0]
            class_name = model.names[class_id]
            label = f"{class_name}: {confidence:.2f}"
            color = class_colors[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, (x1, y1), color)

    cv2.imshow("Podglad", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()