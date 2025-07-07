from sppb_test import SPPBTest
from utils.config import Config

if __name__ == "__main__":
    config = Config()
    sppb = SPPBTest(config)
    result = sppb.run("video.mp4", 1)
    print(result.to_dict())
