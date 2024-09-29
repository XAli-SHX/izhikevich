FROM python:3.12

# Install libgl1
RUN apt-get update && apt-get install -y libgl1

# Copy your application code
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Run your application
# CMD ["python", "main.py"]
