import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        # keypoints = [
        # 655.49, 597.08,
        # 1319.64, 596.96,
        # 324.78, 1039.64,
        # 1663.63, 1042.84,
        # 704.43, 598.73,
        # 425.6, 1043.38,
        # 1271.43, 595.72,
        # 1561.0, 1043.38,
        # 648.97, 688.5,
        # 1327.83, 688.5,
        # 571.15, 809.95,
        # 1408.35, 809.95,
        # 983.65, 688.5,
        # 986.86, 810.08
        # ]

        return keypoints

    # def draw_keypoints(self, image, keypoints):
    #     # Plot keypoints on the image
    #     for i in range(0, len(keypoints), 2):
    #         x = int(keypoints[i])
    #         y = int(keypoints[i+1])
    #         cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #         cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    #     return image
    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        # coordinates = [
        # [655.49, 597.08],
        # [1319.64, 596.96],
        # [324.78, 1039.64],
        # [1663.63, 1042.84],
        # [704.43, 598.73],
        # [425.6, 1043.38],
        # [1271.43, 595.72],
        # [1561.0, 1043.38],
        # [648.97, 688.5],
        # [1327.83, 688.5],
        # [571.15, 809.95],
        # [1408.35, 809.95],
        # [983.65, 688.5],
        # [986.86, 810.08]
        # ]

        # coordinates = [
        # [652.56, 474.73],
        # [1261.95, 474.59],
        # [349.11, 988.11],
        # [1577.59, 991.82],
        # [697.47, 476.65],
        # [441.62, 992.44],
        # [1217.72, 473.15],
        # [1483.42, 992.44],
        # [646.58, 580.78],
        # [1269.47, 580.78],
        # [575.17, 721.66],
        # [1343.36, 721.66],
        # [953.66, 580.78],
        # [956.62, 721.81]
        # ]

        # coordinates = [
        # [294.32, 339.19],
        # [827.33, 344.31],
        # [121.07, 713.7],
        # [1004.13, 716.38],
        # [331.82, 339.54],
        # [187.56, 716.83],
        # [787.18, 343.27],
        # [936.44, 716.83],
        # [297.32, 432.6],
        # [824.39, 435.73],
        # [260.6, 538.69],
        # [862.89, 538.69],
        # [560.85, 433.65],
        # [560.89, 538.8]
        # ]

        # coordinates = [
        # [572.9, 301.6],
        # [1325.2, 304.85],
        # [365.7, 860.56],
        # [1561.08, 857.31],
        # [668.53, 304.85],
        # [515.52, 857.31],
        # [1229.56, 304.85],
        # [1398.51, 857.31],
        # [649.41, 392.59],
        # [1258.25, 392.59],
        # [566.53, 681.82],
        # [1341.13, 681.82],
        # [949.05, 395.84],
        # [958.61, 681.82]
        # ]

        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

        # for i in range(0, len(coordinates), 1):
        #     x = (int)(coordinates[i][0])
        #     y = (int)(coordinates[i][1])
        #     cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #     cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        # return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames