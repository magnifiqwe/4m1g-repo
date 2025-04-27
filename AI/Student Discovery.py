from ultralytics import YOLO
import cv2
import random

def count_and_save_people_in_image(image_path, output_path):
    model = YOLO("yolov8n.pt")
    image = cv2.imread(image_path)
    results = model(image)

    person_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 0:
                person_count += 1
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(output_path, image)
    return person_count

image_path = "2.jpg"
output_path = "output_image_with_boxes.jpg"
num_people = count_and_save_people_in_image(image_path, output_path)

print(f"Количество людей на изображении: {num_people}")
print(f"Изображение сохранено по пути: {output_path}")