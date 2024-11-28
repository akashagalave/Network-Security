# Use Python 3.10 slim image as the base
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install AWS CLI and update dependencies
RUN apt-get update -y && apt-get install -y \
    awscli python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Specify the command to run the application
CMD ["python3", "app.py"]
