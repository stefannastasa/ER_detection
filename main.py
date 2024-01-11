import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()
model_dir = "saved_model/"
model = load_model(model_dir)

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.post("/predict")
async def process_image(image: UploadFile = File(...)):
    contents = await image.read()

    #convert
    np_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    #resize
    img_resized = cv2.resize(img, (287, 287))

    #transpose
    img_transposed = np.transpose(img_resized, (2, 0, 1))

    pred = model.predict(img_transposed)
    if pred > 0.5:
        return {"prediction":"sanatos"}
    else:
        return{"prediction":"bolnav"}

