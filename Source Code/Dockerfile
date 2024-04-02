# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000
 
# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
