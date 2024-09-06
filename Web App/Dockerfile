FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Create the app directory
RUN mkdir /fastapi_app

# Set working directory
WORKDIR /fastapi_app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

RUN chmod a+x docker/*.sh

RUN alembic upgrade HEAD

# Set the command to run the app
CMD gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
