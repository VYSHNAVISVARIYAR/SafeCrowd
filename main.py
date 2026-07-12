from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "SafeCrowd API running"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded successfully"}