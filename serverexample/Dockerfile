# Use an official Python runtime as a parent image
FROM python:2.7-slim
#FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7

ENV LISTEN_PORT=5000
EXPOSE 5000

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

ADD https://storage.googleapis.com/audioset/vggish_model.ckpt .
ADD https://storage.googleapis.com/audioset/vggish_pca_params.npz .