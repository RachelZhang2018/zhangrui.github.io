---

layout: post

title: "Lecture8-Object Detection and Segmentation"

---

**Image Classification**: the process of taking an input image and outputting a class (like “cat”) or a probability that the input is a particular class.

**Image Localization**: (**Single Object**) a regression task, the process of taking an input image and outputting a bounding box of the target object.

**Object Detection**: (**Multiple Objects**) detect instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos.

**Segmentation**: **Semantic Segmentation** and **Instance Segmentation**. 
Semantic segmentation **labels each pixel** in the image with a category label without without caring about the object instances.
Instance segmentation highlight **individual instances** of a class differently.

* * *

# Object Localization

It is a regression problem, output $(x,y,w,h)$ of the bounding box of the object in a digital image.

**Loss Function**: Usually, L2 norm (MSE, **Mean Squared Error**) is applied.

**Evaluation Metric (mAP, mean Average Precision)**

- See [this tutorial](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19y879s5j30m5082q3a.jpg" alt="cad6c5d930b92bb7047fa1d11fd39c5c.png" style="zoom:50%;" />


- IoU, Intersection over Union

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19yb50tbj30n006t3yj.jpg" alt="ef5fb7406b321eea409bb1a373eb525f.png" style="zoom:50%;" />

$$TP (Ture\; Positive) = IoU > 0.5 (threshold)$$\\
$$Precision = \frac{TP}{TP+FP}$$\\
$$Recall = \frac{TP}{TP+FN}$$\\
$$AP = \int_0^1 PR-curve\; dR$$\\ 
$$mAP = average\; AP\; over\;classes$$

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19y794w9j30nw0awdg7.jpg" alt="a13df0ac75eb1fbd286b090ffd8ba873.png" style="zoom:50%;" />



**Regression Head**
Head: Set of fully-connected layers.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19y9612wj30zw0evwk9.jpg" alt="3a17f1012c79db844f34bc5a344640c9.png" style="zoom:50%;" />

***

# Object Detection

- See [this tutorial](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9).

- See [FPN](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c).

## Sliding Window

Slide a window over the image and use a standard CNN classifier (like AlexNet) for each window position. It is very slow!

**Efficient Sliding Window**
Convert fully connected layers to convolutional layers. See [the Stanford note](http://cs231n.github.io/convolutional-networks/#convert).
When the input is a larger image, say $384\times 384$ (original $224\times 224$), using convolutional layers saves a lot of computation. 

**Converted ConvNet:**

$224$: We get a $7\times 7$ feature map before FC layers.
$384$: We get a $12\times 12$ feature map.

$224$: Output size 1x1x1000.
$384$: Output size 6x6x1000.

For the final score, we simply average the scores of each class. Only one pass is needed for larger images.

**Standard ConvNet:**

> valuating the original ConvNet (with FC layers) independently across 224x224 crops of the 384x384 image in strides of 32 pixels gives an identical result to forwarding the converted ConvNet one time.

We need 6x6=36 passes.

**Varying Scales and Aspect Ratios**

Without localization, if the window size is fixed, we can only detect objects of fixed size, and windows are square, but not all objects are square. One solution is to run sliding window with different window sizes.

**Overfeat:**

**Add localization**: Predict both class labels and bounding box at each window position. (the predicted bounding box is in the larger image, can be out of the range of the window.)

**Resize** the images to different scales and increase resolutions.

**Merge** bounding boxes and scores greedily by NMS (**Non-Maximum Supression**). Go down the list of detections starting from highest scoring. Eliminate any detection that overlaps highly with a higher scoring detection.

<span style="color:red">Why is it possible to predict BB out of the range of the sliding window?</span>
Guess: The ground truth is the bounding box in the larger image, so we can still predict the bounding box in the larger image based on the given cropped region (window).
<span style="color:red">Why we need sliding window in different scales and aspect ratios?</span>
Guess: Because a too large sliding window may crop multiple objects, then we cannot distinguish them. Sliding window with different aspect ratio may help in finding multiple shape objects (get a better BB prediction).
<span style="color:red">How to distinguish bounding boxes from different objects in the same class?</span>
Non-maximum supression. Bounding boxes of different objects have little possibility to overlap.

## Region Proposals

**Basic idea:** Find image regions that are likely to contain objects (selective search) and run classifier only on those regions. It is much faster than sliding window (exhaustive search).

You can think of region proposals as a “class-agnostic” object detector.

It is kind of cluster-based image segmentation (groups similar pixels together). There are many different segmentation algorithms (like k-means on color, k-means on color+position, etc.) with many hyperparameters (number of clusters, weights on edges).

- The following pictures are from [here](https://towardsdatascience.com/understanding-object-detection-9ba089154df8).

## R-CNN

**Architecture**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19y8pkirj30ie0cin1y.jpg" alt="a832658878e65d84f69a09787c624dcf.png" style="zoom: 67%;" />

![223b48f4e713a07205ff7de5b605b5b1.png](https://tva1.sinaimg.cn/large/008i3skNly1gs19yaad2vj314v0cgtdl.jpg)

**Problems:**

1. **Slow at test-time:** need to run full forward pass of CNN for each of 2000 region proposals (takes around 47 seconds for each test image).
2. **SVMs and regressors are post-hoc:** CNN features not updated/trained in response to SVMs and regressors.
3. **Complex multistage training pipeline**, as you would have to classify 2000 region proposals per image.
4. The **selective search algorithm is a fixed algorithm**. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.

## Fast R-CNN

**Architecture**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19y6sgvkj30hh0fdtcx.jpg" alt="0702abc3ef140773aaa58622e7cf2c21.png" style="zoom: 67%;" />


![5bdf16ef3ec0dc49e8a6cb38b7fa4792.png](https://tva1.sinaimg.cn/large/008i3skNly1gs19yc45l9j314x0dp777.jpg)


For solving R-CNN problems:

1. Share computation of convolutional layers between proposals for an image. **Project region proposals** to the feature map got from CNN. Forward pass once for all proposals.

2+3. Train the system end-to-end.

5. **RoI Pooling**. Fully-connected layer expects fixed input shape. So, we do **max pooling** of each proposals to get an expected input shape. RoI pooling is based on the whole feature map taking region proposals as reference.

**Problem**
**Region proposals become bottlenecks** in Fast R-CNN algorithm affecting its performance.

## Faster R-CNN

Solution: Make the CNN do region proposals.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19yd0ybsj30nv0fo443.jpg" alt="add2e3c511535567ef5b2180792d3493.png" style="zoom:67%;" />


![97733068e27d741c529b2d2255766c53.png](https://tva1.sinaimg.cn/large/008i3skNly1gs19y7r4r0j31580e2q5l.jpg)

- Use $n$ anchors at each location. Those anchors are translation invariant.

## YOLO: You Only Look Once (One-Stage Object Detection)

**NB:** No region proposal!!!

![6dc1dd5ee7121527f5fcf0731a07593c.png](https://tva1.sinaimg.cn/large/008i3skNly1gs19yedz1nj30zv0fuwtt.jpg)

## SSD: Single Shot Detector (One-Stage Object Detection)

Similar to YOLO, but **denser grid map**, **multi-scale grid maps** + data augmentation + hard negative mining + other design choices in the network.

***

# Semantic Segmentation

Semantic segmentation **labels each pixel** in the image with a category label without without caring about the object instances.

## One-Hot Encoding

We create our target by one-hot encoding the class labels - essentially **creating an output channel for each of the possible classes**. Each pixel is labeled with its most possible class.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19ycfkz5j30ui0dc0yp.jpg" alt="ad7f3dbe0adb86b37e01d7545bf9ab7a.png" style="zoom:67%;" />

## FCN (Fully Convolutional Network)

FCN replace fully connected layers with 1x1xM convolutions.

Conventional CNN has fixed input shape and output shape (class scores), whereas FCN can have various input shape and output shape (not class scores anymore). 

1x1 convolution layers output a spatial map / heatmap. But we need the output to be the same shape as the input image, so we need **upsampling**.

**Upsampling**
**No Parameter:**
<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19yddpdyj30no0entbj.jpg" alt="9fc31e2345b1fee70fca303b93496e3c.png" style="zoom:50%;" />

**Learnable Parameter + Filters: Transpose Convolution**
Transpose convolution takes a single value from the low-resolution feature map and multiply all of the weights in our filter by this value, projecting those weighted values into the output feature map.

However, the upsampling module (decoder) struggles to produce **fine-grained** segmentations.

**Skip Layers**

**Semantic:** global information from late layers resolves ***what***.
**Location:** local information from early layers resolves ***where***.

So, we need combine fine layers and coarse layers.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19ydtinkj30zj0fgq98.jpg" alt="18e062c4c0b78e5ed248780a4b54d8ac.png" style="zoom: 50%;" />

We extract layers from different stages, then upsampling them to have the same size and combine them.

## U-Net

U-Net improves upon the FCN architecture primarily through **expanding the capacity of the decoder module of the network**.

The U-Net architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs19ybhrcuj30dy09u0sr.jpg" alt="6771df62c77eb60ce89f51559f6a987b.png" style="zoom:50%;" />

***

# Instance Segmentation

Along with pixel level classification, we expect the model to classify each instance of a class separately.

## Mask R-CNN

- See [this tutorial](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272).

1. Perform object detection to draw bounding boxes around each instance of a class.
2. Perform semantic segmentation on each of the bounding boxes.

