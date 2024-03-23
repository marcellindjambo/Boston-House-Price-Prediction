# This Dockerfile is used to build a Docker image for the Python application.
# It starts with a Python base image, copies all files in the working directory,
# sets the working directory, installs dependencies, exposes a port, and runs
# two applications: a pipeline and a Streamlit app.

# Base image
FROM python:3.9.18-slim


# Set the working directory
WORKDIR /app

# Copy all files to the container's working directory
COPY . .

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install streamlit

# Expose port
EXPOSE 5000

# 
CMD ["sh", "-c", "python run_pipeline.py && streamlit run streamlit_app.py"]
