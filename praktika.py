import cv2

# Загрузка каскадов для обнаружения лиц, глаз и губ
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Функция для обнаружения и обводки лиц, глаз и губ на изображении
def detect_features(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Преобразование изображения в градации серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отрисовка прямоугольников вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Обнаружение глаз внутри области лица
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=11, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Обнаружение губ внутри области лица
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=11, minSize=(20, 20))
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)

    # Отображение обработанного изображения с обнаруженными лицами, глазами и губами
    cv2.imshow('Detected Features', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования функции
if __name__ == "__main__":
    image_path = 'girl.jpg'  # Путь к изображению
    detect_features(image_path)
