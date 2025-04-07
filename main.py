from PIL import Image
import pytesseract

img = Image.open("sample.jpeg")
text = pytesseract.image_to_string(img)
print(text)
