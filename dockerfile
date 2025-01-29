# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Create a working directory inside the container
WORKDIR /app

# Copy your requirements file first, install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code into the container
COPY . /app

# Expose port 5001 (the port your app listens on)
EXPOSE 5001

# If your app runs with a built-in dev server:
# CMD ["python", "app.py"]

# However, for production, best practice is to use Gunicorn (or uWSGI):
# We'll do a quick Gunicorn example that listens on 0.0.0.0:5001:
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]