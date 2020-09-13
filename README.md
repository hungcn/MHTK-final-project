# **Web-app: COCO object detection and Mask detection with YOLOv3 and Streamlit**

### **Use pre-trained weights from the author - Joseph Redmon**
* Install **opencv**, **streamlit** using pip
* Download **yolov3.weights** from this: https://pjreddie.com/media/files/yolov3. and put it in folder **yolov3_weights**
* Put 2 folder **yolov3_cfg** and **yolov3_weights** in same directory with app.py
* I use OpenCV to draw bounding boxes and scores on pictures (with FiroMono font)
* Just run ```stream run app.py``` in terminal and this web-app will start on your browser


### **For training custom object: detect Mask and Without_mask**
* Clone darknet from​ https://github.com/pjreddie/darknet
* Delete all content in **darknet/cfg** and **darknet/data**
* Copy 2 folder **data** and **cfg** from **for custom training** to folder darknet \
(You can change the config in yolov3_custom.cfg for better result)
* Zip foler darknet, upload **darknet.zip** and **train_custom.ipynb** to google drive folder
* Make folder name **backup** in that same gdrive folder
* Run notebook on google colab to get result the weights file from **backup**
* Change it's name to *yolov3_custom.weights*, download and put it in folder **yolov3_weights**

Dataset: https://www.kaggle.com/alexandralorenzo/maskdetection

### **Demo**
<img src="gui.png" alt="main gui" width="100%"/>
<img src="demo_1.png" alt="object detection" width="600"/>
<img src="demo_2.png" alt="mask detection" width="600"/>
