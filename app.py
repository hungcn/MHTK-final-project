from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import numpy as np
import time
import math
import cv2
import os


@st.cache(allow_output_mutation=True)
def load_model(cfg_path, weights_path, labels_path):
    # load class labels of train dataset
    labels = open(labels_path).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # load pre-trained model
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # determine only the OUPUT layer names that we need from YOLO
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return [net, layer_names, labels, colors]

# @st.cache
def process_image(image_path, model_info, score_threshold=0.25, overlap_threshold=0.3):
    net, layer_names, labels, colors = model_info
    # load our input image and grab its spatial dimensions
    image = img
    H, W = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    # initialize our lists of detected bounding boxes, scores, and class IDs, respectively
    boxes = []
    scores = []
    classIDs = []
    for output in layerOutputs:
        # For each detected object, compute the bounding box, find the score, ignore if below threshold
        for detection in output:
            confidences = detection[5:]
            classID = np.argmax(confidences)
            score = confidences[classID]
            if score >= score_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                scores.append(float(score))
                classIDs.append(classID)

    # apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, overlap_threshold)

    # custom font for showing label on objects
    font = ImageFont.truetype(font='fonts/FiraMono-Medium.otf', size=np.floor(0.01 * (W + H) + 0.5).astype('int32'))
    # calculate suitable thickness of bounding box
    thickness = (H + W) // (420 + len(idxs) // 10)
    
    # convert to PIL image for easier drawing
    img_pil = Image.fromarray(image)

    # ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates, get label
            left, top, right, bottom = boxes[i]
            top = max(0, int(math.floor(top + 0.5)))
            left = max(0, int(math.floor(left + 0.5)))
            right = min(W, int(math.floor(right + left + 0.5)))
            bottom = min(H, int(math.floor(bottom + top + 0.5)))

            draw = ImageDraw.Draw(img_pil)
            color = tuple([int(c) for c in colors[classIDs[i]]])
            label = "{}: {:.3f}".format(labels[classIDs[i]], scores[i])
            labelSize = draw.textsize(label, font)

            origin = np.array([left, top + 1])
            if top - labelSize[1] >= 0:
                origin[1] = top - labelSize[1]
                
            # draw a bounding box rectangle and label on the image
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)
            draw.rectangle([tuple(origin), tuple(origin + labelSize)], fill=color)
            draw.text(origin, label, fill=(0, 0, 0), font=font)
            del draw
    
    image = np.array(img_pil)[:,:, [2, 1, 0]]
    return [image, classIDs, scores]


st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("About")
st.sidebar.text("Intro to ML course's final project")
st.sidebar.text("Author: Cao Nhon Hung - 1712475")
st.sidebar.text("FIT-HCMUS")


# Add a title and sidebar
st.title("Image Object Detection with YOLOv3 and Streamlit")

st.sidebar.markdown("## Choose detection type:")
model_type = st.sidebar.radio("", ("COCO dataset objects detection", "Face Mask detection"))

st.sidebar.markdown("## Confidence threshold:")
score_threshold = st.sidebar.slider("", 0.0, 1.0, 0.4, 0.01)
st.sidebar.markdown("## Overlap threshold:")
overlap_threshold = st.sidebar.slider("", 0.0, 1.0, 0.3, 0.01)


#=============================== PATH TO config, weights, class names
if model_type == "Face Mask detection":
    start = time.time()
    model_info = load_model("yolov3_cfg/yolov3_custom.cfg", "yolov3_weights/yolov3_custom.weights", "custom.names")
    end = time.time()
    model_info[-1] = np.array([[0, 235, 43], [0, 0, 255]], dtype="uint8")
else:
    start = time.time()
    model_info = load_model("yolov3_cfg/yolov3.cfg", "yolov3_weights/yolov3.weights", "coco.names")
    end = time.time()

# styling
st.markdown(
f"""
<style>
    .my-progress-bar {{
        width: 60%;
        height: 10px;
        background-color: #cddefc;
    }}
    .my-progress-bar-fill {{
        display: block;
        height: 10px;
        background-color: #4c7adb;
        transition: width 500ms ease-in-out;
    }}
    .reportview-container .main .block-container{{
        width: {80}%;
        max-width: {900}px;
        margin: 0 auto;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Model YOLOv3 time load:")
st.sidebar.text("{:.4f} seconds".format(end - start))

st.subheader("Upload an image file for dectection")
img_stream = st.file_uploader("", ['jpg', 'jpeg', 'png'])

if img_stream:
    # read image from stream and swap RGB to BGR
    img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.subheader("INPUT:")
    st.image(img[:,:, [2, 1, 0]], use_column_width=True)
    start = time.time()
    image, classIDs, scores = process_image(img, model_info, score_threshold, overlap_threshold)
    end = time.time()
    # st.balloons()
    st.subheader("RESULT:")
    st.image(image, use_column_width=True)
    st.success("Detection time: {:.4f} seconds".format(end - start))
    st.markdown("**Predictions:**")
    if model_type == "Face Mask detection":
        st.write("Found ", classIDs.count(0), " people with mask and ", classIDs.count(1), " people without mask")
    else:
        st.write("Found ", len(classIDs), " objects")
    labels, colors = model_info[2:]
    for i, j in enumerate(classIDs):
        color = tuple([int(c) for c in colors[classIDs[i]]])[-1::-1]
        st.markdown("""
        <div style='display: flex; justify-content: space-between; width: 50%; align-items:center;'>
            <div style='text-align: left; flex:1; color:rgb{}; font-weight: 500; text-shadow: 1px 0px;'>{}:</div>
            <div style='display:flex; justify-content: space-between; align-items:center; flex:1'>
                <div class="my-progress-bar" style='text-align: left'>
                    <span class="my-progress-bar-fill" style="width: {}%;"></span>
                </div>
                <div style='text-align: right'>{:.2f}</div>
            </div>
        </div>
        """.format(color, labels[j], scores[i] * 100, scores[i])
        , unsafe_allow_html=True)
