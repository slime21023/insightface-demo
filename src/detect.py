from typing import List
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
import cv2
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

det_model_path = "buffalo_s/det_500m.onnx"
det_model = model_zoo.get_model(f"{BASE_DIR}/models/{det_model_path}")

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)


def single_face(img: cv2.typing.MatLike) -> Face:
    infos, kpss = det_model.detect(img, max_num=1, metric="defualt")

    bbox = infos[0, :4]
    det_score = infos[0, 4]
    kps = kpss[0]
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    return face


def faces(img: cv2.typing.MatLike) -> List[Face]:
    faces = []
    infos, kpss = det_model.detect(img, max_num=0, metric="defualt")
    for info, kps in zip(infos, kpss):
        bbox = info[:4]
        det_score = info[4]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        faces.append(face)

    return faces


def draw_rectangle(
    img: cv2.typing.MatLike, faces: List, color: tuple = (0, 0, 255)
) -> cv2.typing.MatLike:
    dimg = img.copy()

    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int32)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
    return dimg
