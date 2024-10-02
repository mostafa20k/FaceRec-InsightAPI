from fastapi import FastAPI, File, UploadFile
from operations import evaluate_model, initialize_model
from pydantic import BaseModel
import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        module="insightface.utils.transform")

app = FastAPI(title="Face recognition")
model = None


class ModelInitializationRequest(BaseModel):
    modelName: str
    dataset: str


class PredictRequest(BaseModel):
    file1: UploadFile
    file2: UploadFile


@app.post("/initialize_model")
def initialize(request: ModelInitializationRequest):
    global model
    try:
        model = initialize_model(request.modelName)
        return {"message": f"Model {request.modelName} and {request.dataset} dataset has been initialized successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=400,
                            detail="Model is not initialized.")
    img_1 = await file1.read()
    img_2 = await file2.read()
    prediction = evaluate_model(img_1, img_2, model)
    return {"predicted label": prediction}
