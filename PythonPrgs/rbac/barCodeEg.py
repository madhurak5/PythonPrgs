from barcode import EAN13
from barcode.writer import ImageWriter
# with open('barcode2.png', 'wb') as f:
#     EAN13('3453045687', writer=ImageWriter()).write(f)
import qrcode
from PIL import Image
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
qr.add_data("https://medium.com/@ngwaifoong92")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("qrcode.png")
from pyzbar.pyzbar import decode
img = Image.open('qrcode.png')
import cv2
img = cv2.imread('qrcode.png')
result = decode(img)
for i in result:
    print(i.data.decode("utf-8"))