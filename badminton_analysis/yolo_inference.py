from ultralytics import YOLO 

# model = YOLO('yolo5_last.pt')
model = YOLO('models/best3.pt')

# result = model.track('input_videos/input_video5.mp4',conf=0.1, save=True)
result = model.predict('input_videos/input_vid.mp4',conf=0.075 , save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)

