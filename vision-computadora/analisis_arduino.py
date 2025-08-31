import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen
img = cv2.imread("arduino_led.png")

if img is None:
    print("Error: no se pudo cargar la imagen. Revisa el nombre o la ruta.")
    exit()

#Aplicacion de filtros
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
output = img.copy()


# Deteccion LED
low_red1 = np.array([0, 150, 150])
high_red1 = np.array([10, 255, 255])
low_red2 = np.array([170, 150, 150])
high_red2 = np.array([180, 255, 255])
mask_led = cv2.inRange(hsv, low_red1, high_red1) | cv2.inRange(hsv, low_red2, high_red2)

contours, _ = cv2.findContours(mask_led, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 100:  # evitar puntos pequeños
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(output, "LED", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


low_blue = np.array([95, 80, 80])
high_blue = np.array([130, 255, 255])
mask_arduino = cv2.inRange(hsv, low_blue, high_blue)

contours, _ = cv2.findContours(mask_arduino, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 3000:  # solo objetos grandes
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(output, "Arduino UNO", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)


low_gray = np.array([0, 0, 200])
high_gray = np.array([180, 40, 255])
mask_proto = cv2.inRange(hsv, low_gray, high_gray)

contours, _ = cv2.findContours(mask_proto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 5000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w,y+h), (128,128,128), 2)
        cv2.putText(output, "Protoboard", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)

#Deteccion de cables
#rojo
low_red_cable = np.array([0, 100, 100])
high_red_cable = np.array([10, 255, 255])
mask_red_cable = cv2.inRange(hsv, low_red_cable, high_red_cable)

contours, _ = cv2.findContours(mask_red_cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 200:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(output, "Cable rojo", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Negro
low_black = np.array([0, 0, 0])
high_black = np.array([180, 255, 50])
mask_black = cv2.inRange(hsv, low_black, high_black)

contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 200:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(output, "Cable negro", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#Graficas
plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")

plt.subplot(2,2,2)
plt.imshow(gray, cmap="gray")
plt.title("Escala de grises")

plt.subplot(2,2,3)
plt.imshow(edges, cmap="gray")
plt.title("Bordes detectados (Canny)")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detección de componentes")

plt.tight_layout()
plt.show()