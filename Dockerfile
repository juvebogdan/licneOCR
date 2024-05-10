# Start with a Python 3.9 base image
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to be accessible from the outside
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# The command to run the app using Flask's built-in server
# The command to run the app using Gunicorn in production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
