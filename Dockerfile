# Use the official Python base image with version 3.8
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

COPY setup.py .

RUN pip install --upgrade pip


# Install the Python dependencies
RUN pip install -r requirements.txt


# Copy the entire project to the working directory
COPY . .


# Set the default command to run when the container starts
CMD ["streamlit", "run", "--server.port", "8501", "credit_app.py"]