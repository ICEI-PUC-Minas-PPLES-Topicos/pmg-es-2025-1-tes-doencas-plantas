import cv2
import os
import numpy as np

IMG_DIR = "imagens"
SAVE_DIR = "mascaras"
os.makedirs(SAVE_DIR, exist_ok=True)

drawing = False
ix, iy = -1, -1
mask = None

def draw_circle(event, x, y, flags, param):
    global drawing, ix, iy, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), 10, (255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), 10, (255), -1)

def criar_mascaras():
    imagens = os.listdir(IMG_DIR)

    for nome in imagens:
        caminho_img = os.path.join(IMG_DIR, nome)
        img = cv2.imread(caminho_img)
        img = cv2.resize(img, (256, 256))

        global mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        cv2.namedWindow("Imagem")
        cv2.setMouseCallback("Imagem", draw_circle)

        while True:
            display = img.copy()
            display[mask > 0] = [0, 255, 0]  # mostra onde já marcou

            cv2.imshow("Imagem", display)
            cv2.imshow("Máscara", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(os.path.join(SAVE_DIR, nome), mask)
                print(f"Máscara salva para {nome}")
                break
            elif key == ord("q"):
                print("Pulando imagem.")
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    criar_mascaras()
