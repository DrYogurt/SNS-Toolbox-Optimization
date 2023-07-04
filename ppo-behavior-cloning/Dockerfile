# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app
# Set the environment variables required by MuJoCo

# Copy the requirements.txt file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update && apt upgrade -y
RUN apt install xvfb libgl1-mesa-glx -y

# Copy the current directory contents into the container at /app
COPY . /app

# Set the entry point and pass the command line arguments to the script
ENTRYPOINT ["python", "sweep.py"]