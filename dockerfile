# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN apt-get install -y poppler-utils

# Set the working directory in the container
WORKDIR /app

# Install any needed packages
RUN pip install fastapi
RUN pip install pathlib
RUN pip install opencv-python
RUN pip install transformers -U
RUN pip install torch
RUN pip install pdf2image
RUN pip install tqdm
RUN pip install Pillow
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install timm -U

# Copy the rest of the application code into the container at /app
COPY . /app
COPY pdfToJpeg /app/pdfToJpeg

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
