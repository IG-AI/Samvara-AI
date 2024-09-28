# Base image
FROM tensorflow/tensorflow:2.12.0-gpu

# Set up working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install required packages
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python", "main.py"]
