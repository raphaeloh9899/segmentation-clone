from keras_segmentation.models.segnet import segnet

model = segnet(n_classes=50, input_height=480, input_width=360)

o = model.predict_segmentation(
    inp="/content/drive/MyDrive/Colab Notebooks/dataset1/images_prepped_test/0016E5_08049.png",
    out_fname="/tmp/out.png" , overlay_img=True, show_legends=True,
    class_names = [ "Sky", "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"]

)

import matplotlib.pyplot as plt
plt.imshow(o)
plt.show()