from models import load_custom_model, load_coco_model
import cv2

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.4
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

model = load_custom_model()
model_coco = load_coco_model()


cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Brak dostępu do kamery.")
    exit()

while True:
    ret, frame = cap.read()

    prediction = model_coco(frame)

    __draw_label(frame, 'Obiekt: {}'.format(prediction), (20, 20), (255, 0, 0))

    cv2.imshow("Podgląd", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()