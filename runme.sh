#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker."
    exit 1
fi

# Run the Docker container
docker load -i app.tar
docker run -v < output path on pc >:/app/pdfToJpeg/pdfs -p 4040:4040 main
# Example
# docker run -v D:\Projects\FrontEnd\outs:/app/pdfToJpeg/pdfs -p 4040:4040 main