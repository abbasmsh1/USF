from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import os
import cv2
from datetime import datetime
from pdf2image import convert_from_path
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import warnings
from PIL import Image
import uvicorn
import timm

def convertor(pdf_path, output_dir, image_format='JPEG', dpi=200):
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(r"pdfToJpeg/pdfs"):
        os.makedirs(r"pdfToJpeg/pdfs")
    for idx, page in enumerate(pages):
        image_path = f"{output_dir}/page_{idx + 1}.{image_format.lower()}"
        page.save(image_path, image_format)
        print(f"Page {idx + 1} saved as {image_path}")



class PDF_to_PDF():
    def __init__(self, path):
        convertor(path,output_dir=r"pdfToJpeg/jpegs")
        file_path=r"pdfToJpeg/jpegs/page_1.jpeg"
        image = Image.open(file_path).convert("RGB")

        # Load pre-trained image processor and model for table detection
        image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Expand the bounding box by 10 pixels in each direction
            expanded_box = [
                int(box[0] - 10),
                int(box[1] - 10),
                int(box[2] + 30),
                int(box[3] + 30)
            ]

            # Crop the image at the expanded bounding box
            cropped_image = image.crop(expanded_box)

            # Display the cropped image or perform further processing
            # cropped_image.show()

            filename = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join('pdfToJpeg', 'jpegs', f"{filename}.jpeg")
            cropped_image.save(save_path)

        # Load the first page of the original PDF for further processing
        pages = convert_from_path(r'./Sling Mock Timesheet V.3.pdf', dpi=200)
        pages[0].save('full_page_image.jpg', 'JPEG') # where x is your page number minus one
        full_page_image = cv2.imread('full_page_image.jpg')
        image_to_be_added = cv2.imread(r'./pdfToJpeg/jpegs/page_1.jpeg')

        x, y, _ = full_page_image.shape
        x_mid, y_mid = x // 2, y // 2

        # Calculate the dimensions of the region to replace
        region_height = x // 2
        region_width = y

        # Calculate the new height for the resized image, reduced by 20%
        new_height = int(region_height * 0.4)
        height = new_height

        # Calculate the new width for the resized image
        new_width = int(region_width * 0.9)  # Reduce width by 30%

        # Resize the image to be added, reducing the height and width
        resized_image = cv2.resize(image_to_be_added, (new_width, new_height))

        # Calculate the new top position for the region
        new_top = int(x_mid - new_height * 1.8)  # Raise image position by 20%

        # Calculate the bottom position of the region
        new_bottom = new_top + new_height

        # Calculate the left and right positions for the region
        new_left = y_mid - new_width // 2
        new_right = new_left + new_width

        # Replace the region in full_page_image with the resized image
        final_image = full_page_image.copy()
        final_image[new_top:new_bottom, new_left:new_right] = resized_image

        # Save the final image
        cv2.imwrite("final_image.jpg", final_image)
        final_image2 = Image.open('final_image.jpg')
        final_image3 = final_image2.convert('RGB')
        
        # Extract the name of the PDF for generating the final file name
        name = os.path.basename(path).replace("Sling Timesheets Report - ", "").replace('.pdf', '')
        final_name = os.path.join('pdfToJpeg', 'pdfs', f"{name} PPE {datetime.today().strftime('%m-%d-%Y')}.pdf")
        final_image3.save(final_name)
        self.final_image = final_image2

    def get_res(self):
        return self.final_image

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the "static" directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    try:
        # Create the "uploaded_files" directory if it doesn't exist
        upload_dir = Path("uploaded_files")
        upload_dir.mkdir(parents=True, exist_ok=True)

        messages = []  # List to store messages for each file

        for idx, file in enumerate(files, 1):  # Use enumerate to get the file number starting from 1
            # Save the uploaded file to the "uploaded_files" directory
            if ".pdf" == file.filename[-4:]:
                file_path = upload_dir / file.filename
                with file_path.open("wb") as buffer:
                    buffer.write(await file.read())
                # Process the PDF and save the result
                PDF_to_PDF(file_path)

                # Append the status and filename to the messages list with file number
                messages.append({"file_number": idx, "filename": file.filename, "status": "Uploaded and processed successfully!"})
            else:
                messages.append({"file_number": idx, "filename": file.filename, "status": "Incorrect format, File is not a pdf"})
    except Exception as e:
        # Append the status and filename to the messages list with file number
        messages.append({"file_number": idx, "filename": file.filename, "status": f"File upload failed: {str(e)}"})

    return templates.TemplateResponse("upload.html", {"request": request, "messages": messages})


def serve():
    """Serve the web application."""
    uvicorn.run(app,host= "0.0.0.0", port=4040)  # or whatever port number you want.

if __name__ == "__main__":
    serve()
