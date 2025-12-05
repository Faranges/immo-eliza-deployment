# -----------------------------
# 1. Use official lightweight Python
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# 2. Create app working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Install system dependencies (needed for numpy, xgboost)
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4. Copy only the API requirements
# -----------------------------
COPY api/requirement.txt ./requirements.txt

# -----------------------------
# 5. Install Python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 6. Copy API files
# -----------------------------
COPY api/ ./api/

# -----------------------------
# 7. Copy model folder & .pkl file
# -----------------------------
COPY models/ ./models/

# -----------------------------
# 8. Expose Render port
# -----------------------------
EXPOSE 8000

# -----------------------------
# 9. Run the FastAPI app with Uvicorn
# -----------------------------
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
