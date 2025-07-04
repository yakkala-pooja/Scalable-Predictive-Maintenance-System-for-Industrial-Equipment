# Use official Python image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install extra dependencies for FastAPI and model
RUN pip install --no-cache-dir fastapi uvicorn[standard] joblib pandas torch scikit-learn

# Copy all code
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"] 