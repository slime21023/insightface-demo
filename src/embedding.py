from typing import List
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
import numpy as np
import cv2
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

rec_model_path = "buffalo_s/w600k_mbf.onnx"
rec_model = model_zoo.get_model(f"{BASE_DIR}/models/{rec_model_path}")


def get(img: cv2.typing.MatLike, face: Face) -> np.array:
    rec_model.get(img, face)
    return face.normed_embedding


def get_all(img: cv2.typing.MatLike, faces: List[Face]) -> List[np.array]:
    embeddings = []
    for face in faces:
        rec_model.get(img, face)
        embeddings.append(face.normed_embedding)
    return embeddings
