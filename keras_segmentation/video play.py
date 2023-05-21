from keras_segmentation.predict import predict_video

checkpoints_path = "/tmp/segnet_1"
input_video = "/content/drive/MyDrive/Colab Notebooks/dataset1/testing_video_1.mp4"
output_file = "/tmp/out.mp4"

predict_video(checkpoints_path, input_video, output_file, display=True)
