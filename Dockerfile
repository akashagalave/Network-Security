# Use a lightweight Python base image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the working directory
COPY . /app

# Update the package lists and install necessary system dependencies
RUN apt update -y && \
    apt install -y awscli python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Specify the command to run the application
CMD ["python3", "app.py"]
