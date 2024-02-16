# ETHZ: Computer Vision 2023
This repository contains the code for the projects of the Computer Vision course at ETH Zurich. I received full grades for all my submissions. 

### Project 1: Harris Corner Detector
Implementation of the Harris Corner Detector, to find unique, descriptive points in images. The following images show the detected positions in red.
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%201%20-%20Harris%20Corner%20Detector/Results/blocks_harris.png?raw=true" style="width: 30%;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%201%20-%20Harris%20Corner%20Detector/Results/house_harris.png?raw=true" style="width: 30%;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%201%20-%20Harris%20Corner%20Detector/Results/I1_harris.png?raw=true" style="width: 30%;">
</div>

We can use the detected patches to match features between images. The following image shows the matches between two images using the Harris Corner Detector. 
<div style="display: flex; justify-content: space-between; center;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%201%20-%20Harris%20Corner%20Detector/Results/match_mutual.png?raw=true" style="width: 80%;">
</div>

### Project 2: Image Classification
Simple machine learning classifiers for the MNIST dataset.

### Project 3: Bag of Words and VGG16
Training a Bag of Words model to classify images, if they contain a car (positive sample) or not (negative sample). The following are examples of images, which the algorithm is able to differentiate between.

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://camo.githubusercontent.com/7a907a5aac851fcf7f832963ac522c7022f0828e9097c9acea39b72ad4305f26/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f642f64322f5376675f6578616d706c655f7371756172652e7376672f35313270782d5376675f6578616d706c655f7371756172652e7376672e706e67" alt="image alt >" style="width: 100%;">
        <figcaption>Negative Sample</figcaption>
    </figure>
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://camo.githubusercontent.com/7a907a5aac851fcf7f832963ac522c7022f0828e9097c9acea39b72ad4305f26/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f642f64322f5376675f6578616d706c655f7371756172652e7376672f35313270782d5376675f6578616d706c655f7371756172652e7376672e706e67" alt="image alt <" style="width: 100%;">
        <figcaption>Positive Sample</figcaption>
    </figure>
</div>

#### VGG16
Implemented the [VGG16](https://arxiv.org/abs/1409.1556) model to classify images of the CIFAR-10 dataset.

### Project 4: Image Segmentation

#### Mean Shift Segmentation
Using the Mean-shift algorithm applied to the RGB space, we can segment an image into different regions. The following images show the input and output of the algorithm. The algorithm was applied to the image of the ETH building.


<div style="display: flex; justify-content: space-between;">
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/eth.jpg?raw=true" alt="image alt >" style="width: 100%;">
        <figcaption>Input</figcaption>
    </figure>
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/result_beta=3.0.png?raw=true" alt="image alt <" style="width: 100%;">
        <figcaption>Output</figcaption>
    </figure>
</div>

#### SegNet
Implementation of a lite version of [SegNet](https://arxiv.org/abs/1511.00561). The images below show the input and output of the network. The network was trained on a modified version of the MNIST dataset. The model is supposed to detect each digit in an image, label it accordingly, and color code all the pixels of the image according to the label.

Input:
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/seg_net_inp1.png?raw=true" style="width: 30%;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/seg_net_inp2.png?raw=true" style="width: 30%;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/seg_net_inp3.png?raw=true" style="width: 30%;">
</div>
Output:
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/seg_net_out1.png?raw=true" style="width: 30%;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/seg_net_out2.png?raw=true" style="width: 30%;">
    <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%204%20-%20Image%20Segmentation/Results/seg_net_out3.png?raw=true" style="width: 30%;">
</div>

### Project 5: Object Tracking
Using the Conditional Density Propagation (Condensation) algorithm to track an object in a video. The following GIF shows the tracking of a persons hands, where the blue box represents the prior belief of the position of the object, and the red box the updated posterior belief. The blue dots, represents samples from the prior belief.

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/ericbill21/ETHZ_Computer_Vision_2023/blob/main/Project%205%20-%20Object%20Tracking/Result/Video2.gif?raw=true" style="width: 80%;">
</div>

### Project 6: Structure from Motion
Given a set of images and correspondences between them, we can reconstruct the 3D structure of the scene, by estimating the relative camera poses and triangulating the 3D points. The following images show the reconstruction of a scene using the SfM algorithm.

Example of the input images:
<div style="display: flex; justify-content: space-between;">
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%206%20-%20Structure%20from%20Motion/Code/data/images/0002.png?raw=true" alt="image alt >" style="width: 100%;">
        <figcaption></figcaption>
    </figure>
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%206%20-%20Structure%20from%20Motion/Code/data/images/0008.png?raw=true;">
        <figcaption></figcaption>
    </figure>
</div>

Resulting 3D reconstruction and camera poses:
<div style="display: flex; justify-content: space-between;">
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%206%20-%20Structure%20from%20Motion/Result/Figure_4.png?raw=true" alt="image alt >" style="width: 100%;">
    </figure>
    <figure style="width: 50%; margin-right: 5px; margin-left: 5px;">
        <img src="https://github.com/ericbill21/ETHZ-Computer_Vision_2023/blob/main/Project%206%20-%20Structure%20from%20Motion/Result/Figure_5.png?raw=true" alt="image alt <" style="width: 100%;">
    </figure>
</div>

Each camera is depicted as a blue box, and each 3D point as a black dot.
