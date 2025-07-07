import sys
import os
import numpy as np
from openpose import pyopenpose as op

class OpenPoseProcessor:
    def __init__(self, model_folder):
        params = {"model_folder": model_folder}
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

    def process_frame(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        vec = op.VectorDatum()
        vec.append(datum)
        self.op_wrapper.emplaceAndPop(vec)
        kps = datum.poseKeypoints
        return kps[0] if kps is not None and len(kps) > 0 else None
