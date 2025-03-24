import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page as doctr_visualize_page
import io
from PIL import Image
import matplotlib.pyplot as plt

class TextExtractor:
    def __init__(self):
        self.ocr_model = ocr_predictor(pretrained=True)
    
    def process_image(self, image_bytes: bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        document = self.ocr_model([img_rgb])
        page = document.pages[0]
        
        # Extract text
        all_text = []
        for block in page.blocks:
            for line in block.lines:
                text = " ".join(word.value for word in line.words)
                all_text.append(text)
        
        # Create visualization
        page_dict = page.export()
        fig = doctr_visualize_page(page_dict, img_rgb)
        
        # Save visualization to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        return "\n".join(all_text), buf.getvalue() 
