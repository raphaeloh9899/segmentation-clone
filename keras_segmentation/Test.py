from keras_segmentation.models.segnet import segnet
import matplotlib.pyplot as plt
from keras_segmentation.predict import predict_video
import cv2
def image_preprocessing(image):
    return image + 1

model = segnet(n_classes=50, input_height=480, input_width=360)

model.train(
        train_images="/content/drive/MyDrive/Colab Notebooks/dataset1/images_prepped_train/",
        train_annotations="/content/drive/MyDrive/Colab Notebooks/dataset1/annotations_prepped_train/",
        checkpoints_path="/tmp/segnet_1", epochs=5,
        preprocessing=image_preprocessing
)

out = model.predict_segmentation(
        inp="/content/drive/MyDrive/Colab Notebooks/dataset1/images_prepped_test/0016E5_08049.png",
        out_fname="/tmp/out.png"
)

plt.imshow(out)
plt.show()

checkpoints_path = "/tmp/segnet_1"
input_video = "/content/drive/MyDrive/Colab Notebooks/dataset1/testing_video_1.mp4"
output_file = "/content/drive/MyDrive/Colab Notebooks/dataset1/output/test_out.mp4"

video_capture = cv2.VideoCapture(input_video)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # 프레임을 예측
    predicted_frame = predict_video(checkpoints_path, frame)

    # 예측된 프레임을 출력 파일에 저장
    output_writer.write(predicted_frame)

video_capture.release()
output_writer.release()

#predict_video(
 #   checkpoints_path=checkpoints_path,
  #  inp=input_video,
   # output=output_file,
    #display=True
#)