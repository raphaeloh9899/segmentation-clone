from keras_segmentation.models.segnet import segnet

model = segnet(n_classes=50, input_height=480, input_width=360)

print( model.evaluate_segmentation(inp_images_dir="/content/drive/MyDrive/Colab Notebooks/dataset1/images_prepped_test/",annotations_dir="/content/drive/MyDrive/Colab Notebooks/dataset1/annotations_prepped_test/") )