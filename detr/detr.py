import mmcv
from mmdet.apis import inference_detector, init_detector
import wget
from utils import *

class Detr():
    def __init__(self, config_path=None, model_path=None, device='cpu'):
        if not config_path:
            config_path = 'model/detr_config.py'
        if not model_path:
            model_path = 'model/detr.pth'
        
        if not os.path.exists(model_path):
            url = 'https://github.com/cbddobvyz/digitaleye-mammography/releases/download/shared-models.v1/detr.pth'
            wget.download(url, out='model/')
        
        self.model = init_detector(config_path, model_path, device)
    
    def run(self, image_array):
        result = inference_detector(self.model, mmcv.imfrombytes(image_array))
        result = apply_nms(result, image_array, 1)
        return result[0]
