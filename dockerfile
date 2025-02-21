# Use an official Python runtime as a parent image
FROM python:3.14-rc-alpine3.21

# Set a working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that the app will listen on
EXPOSE 5001

# Use Gunicorn to serve the application in production
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]