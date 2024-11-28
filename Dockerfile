# Use Python 3.10 slim image as the base
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Update packages and install dependencies
RUN apt-get update -y && apt-get install -y \
    awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Specify the default command to run the application
CMD ["python3", "app.py"]
