import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


def initialize_model(model_name: str):
    """Initialize and return the face recognition model."""
    if model_name == "buffalo_s" or model_name == "buffalo_l":
        model = FaceAnalysis(name=model_name)
        model.prepare(ctx_id=0, det_size=(480, 480))
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model


def preprocess_image(image):
    """Decode given files from the client"""
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image


def get_embedding(image, model):
    """Return the model output for face recognition"""
    image = preprocess_image(image)
    faces = model.get(image)
    if faces:
        return faces[0].embedding
    else:
        return None


def compute_similarity(embedding1, embedding2):
    """Compute the cosine similarity between two embeddings."""
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def evaluate_model(img1, img2, model, threshold=0.2):
    """Check if images are pair or not"""
    embedding1 = get_embedding(img1, model)
    embedding2 = get_embedding(img2, model)

    if embedding1 is not None and embedding2 is not None:
        similarity = compute_similarity(embedding1, embedding2)
        predicted_label = 1 if similarity > threshold else 0
    else:
        predicted_label = 0

    return predicted_label
