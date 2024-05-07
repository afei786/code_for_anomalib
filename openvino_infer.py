from matplotlib import pyplot as plt
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib import TaskType
import cv2
import time 

t1 = time.time()

openvino_model_path = r"D:\git_file\test_anomalib\weights\openvino\model.xml"
image_path = r"D:\git_file\test_anomalib\datasets\3\abnormal1.jpg"
image = read_image(path=image_path)
plt.imshow(image)
metadata_path = r"D:\git_file\test_anomalib\weights\openvino\metadata.json"
# print(openvino_model_path.exists(), metadata_path.exists())
inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata_path,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)
print('inferencer time:', time.time() - t1)
print(image.shape)
predictions = inferencer.predict(image=image)

# from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
# from PIL import Image

# visualizer = ImageVisualizer(mode=VisualizationMode.SIMPLE, task=TaskType.SEGMENTATION)
# output_image = visualizer.visualize_image(predictions)
# # Image.fromarray(output_image)
# # cv2.imwrite("output.png", output_image)
# output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
# # cv2.imwrite("output.png", output_image)

output_image = cv2.cvtColor(predictions.heat_map, cv2.COLOR_RGB2BGR)
cv2.imshow("output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()