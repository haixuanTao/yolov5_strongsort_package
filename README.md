# Strong_sort algorithm to use on top of yolov5.

This package only includes the strong sort algorithm.

Usage example:

```python
import cv2
import torch

from strong_sort import StrongSORT

device = torch.device("cpu")

model_yolov5 = torch.hub.load(
    "ultralytics/yolov5", "yolov5s", pretrained=True
)  # or yolov5n - yolov5x6, custom

model_strongsort = StrongSORT(
    "osnet_ibn_x1_0_msmt17.pt",
    device,
    False,
)
model_strongsort.model.warmup()

# Preprocessing Yolov5
img = "./frame_00_delay-0.13s.jpg"
img = cv2.imread(img)
results = model_yolov5([img])
pred = results.xyxy[0]


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# Post Processing yolov5
xywhs = xyxy2xywh(pred[:, 0:4])
confs = pred[:, 4]
clss = pred[:, 5]

# Running strong_sort
with torch.no_grad():
    outputs = model_strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
    # Results
    print(outputs)
```

This is mainly a packaging of [Yolov5_StrongSORT_OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)

