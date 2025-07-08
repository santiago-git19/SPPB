from phases.balance import BalancePhase
from phases.gait_speed import GaitSpeedPhase
from phases.chair_rise import ChairRisePhase
from utils.openpose_proc import OpenPoseProcessor
from results import SPPBResult
import cv2

class SPPBTest:
    def __init__(self, config):
        self.config = config
        self.openpose = OpenPoseProcessor(config.model_folder)
        self.balance = BalancePhase(self.openpose, config)
        self.gait = GaitSpeedPhase(self.openpose, config)
        self.chair = ChairRisePhase(self.openpose, config)

    def run(self, video_path, camera_id):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir {video_path}")
        balance_result = self.balance.run(cap, camera_id, self.config.duration)
        gait_result = self.gait.run(cap, camera_id)
        chair_result = self.chair.run(cap, camera_id)
        cap.release()
        return SPPBResult(balance_result, gait_result, chair_result)
