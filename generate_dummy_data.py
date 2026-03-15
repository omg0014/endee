import os
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas

os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

# 1. Create a dummy PDF about Machine Learning
c = canvas.Canvas("data/pdfs/ai_notes.pdf")
c.drawString(100, 750, "Introduction to Machine Learning")
c.drawString(100, 730, "Machine learning is a field of study that gives computers the ability to learn")
c.drawString(100, 710, "without being explicitly programmed. Deep learning is a subset of machine")
c.drawString(100, 690, "learning that uses multi-layered artificial neural networks to deliver")
c.drawString(100, 670, "state-of-the-art accuracy in tasks such as object detection, speech")
c.drawString(100, 650, "recognition, and language translation.")
c.save()

# 2. Create another PDF about Vector Databases
c = canvas.Canvas("data/pdfs/vector_db.pdf")
c.drawString(100, 750, "Vector Databases Explained")
c.drawString(100, 730, "A vector database like Endee connects unstructured data mathematically.")
c.drawString(100, 710, "Sentence transformers map textual inputs like 'cat' to numerical dense")
c.drawString(100, 690, "vectors. Searches happen by calculating cosine similarity between queries")
c.drawString(100, 670, "and the stored document clusters.")
c.save()

# 3. Create a dummy Image
img = Image.new('RGB', (200, 200), color=(73, 109, 137))
d = ImageDraw.Draw(img)
d.text((10,10), "AI Brain Diagram", fill=(255,255,0))
img.save("data/images/brain_diagram.png")

# 4. Create another string Image
img2 = Image.new('RGB', (200, 200), color=(255, 100, 100))
d2 = ImageDraw.Draw(img2)
d2.text((10,10), "Self Driving Car", fill=(255,255,255))
img2.save("data/images/self_driving_car.png")

print("Created 2 dummy PDFs in data/pdfs/ and 2 dummy Images in data/images/")
