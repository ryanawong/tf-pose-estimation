# tf-pose-estimation

This project forks the repository from https://github.com/ildoonet/tf-pose-estimation.

## Setup

This was intended for the Nvidia Jetson TX2, but was successfully tested on a dual boot using Ubuntu 16.04.

### Dependencies

You need dependencies below.

- python3/2.7
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk

### Flashing the Nvidia Jetson Board

Download the "64-bit Driver Package", and the "Sample Root File System" from https://developer.nvidia.com/embedded/linux-tegra-archive. 

Use the instructions from https://developer.download.nvidia.com/embedded/L4T/r28_Release_v2.0/GA/BSP/l4t_quick_start_guide.txt.

## Check Python Version

```bash
python -V
python3 -V
```

# Install OpenCV3

Based on https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/ with some modifications.

Install dependencies

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev python3.5-dev
```

Install OpenCV

```bash
cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.0.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install numpy
```

Build and make OpenCV

```bash
cd ~/opencv-3.0.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D FORCE_VTK=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D WITH_GDAL=ON -D WITH_XINE=ON -D BUILD_EXAMPLES=ON ..
make
```

If there are any issues during the building and making of OpenCV, run the following commands. Else ignore

```bash
make clean
cd ..
sudo rm -r build
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D FORCE_VTK=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D WITH_GDAL=ON -D WITH_XINE=ON -D BUILD_EXAMPLES=ON ..
make
```

Finish by actually install OpenCV

```bash
sudo make install
sudo ldconfig
```

If using Python 2.7

```bash
cd ~/.virtualenvs/cv/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
```

If using Python 3+

```bash
cd /usr/local/lib/python3.5/site-packages/
sudo mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
cd ~/.virtualenvs/cv/lib/python3.5/site-packages/
ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
```

Validate installation
```bash
python
import cv2
cv2.__version__
```

Your output should show the version of OpenCV you just installed. If you get any other issues, you may need to Google your issue, redo the "Build and Make" steps, or even reflash the Jetson.

### Update Nvidia Drivers

Default drivers are from Nouveau who tried to reverse engineering the Nvidia drivers but they don't work with CUDA or cuDNN. Check http://www.nvidia.com/object/unix.html for the latest version of the Nvidia driver -- this project used 390.

The following instructions was based on http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux. 

Go through the following commands

```bash
sudo apt-get purge nvidia*
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-[number]
```

Where [number] for me was 390 based on the above link.

```bash
lsmod | grep nvidia
```

If no output, then something went wrong, try Update Nvidia Drivers steps from the beginning again

```bash
lsmod | grep nouveau
```

There should be no output from there

```bash
sudo reboot
```

### Install CUDA Toolkit 9.1

- 05/07/2018: I used CUDA Toolkit 9.1, but there may be a newer version that may have issues that you'd need to Google to fix.

Based on https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/ with some modifications, and ignoring the fact that this is for Caffe.

Go to https://developer.nvidia.com/cuda-downloads and download the CUDA Toolkit according to your OS, Architecture, Distribution, and Version. I used Linux, x86_64, Ubuntu, and 16.04. For the Installer Type, I chose "deb (network)". We will call this downloaded file [file].

```bash
cd ~/cuda-downloads
sudo dpkg -i [file].deb
sudo apt-get update
sudo apt-get install cuda
```

### Install cuDNN v7.1.3

- 05/07/2018: I used cuDNN v7.1.3, but there may be a newer version that may have issues that you'd need to Google to fix.

Based on https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/ with some modifications, and ignoring the fact that this is for Caffe.

Go to https://developer.nvidia.com/cudnn, sign up as an Nvidia developer and download the file for "cuDNN v7.1.3 for CUDA 9.1". Usually this is the "cuDNN v7.1.3 Libary for Linux" file. We will call this downloaded file [file].

```bash
cd ~/Downloads
tar -xvf [file].tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64
sudo cp include/* /usr/local/cuda/include
```

### Install Tensorflow 1.7

Based on https://www.tensorflow.org/install/install_linux#ValidateYourInstallation with some modifications.

Install some stuff for Python

```bash
sudo apt-get install python-pip python-dev python3-pip python3-dev
```

I found from https://github.com/tensorflow/tensorflow/issues/15604 that the latest version of Tensorflow 1.8 (from using "sudo pip install tensorflow-gpu" from their website) does not support CUDA 9.0, so we need to use Tensorflow 1.7

If you are using Python 2.7

```bash
sudo pip upgrade
sudo pip install --upgrade tensorflow-gpu==1.7
```

If you are using Python 3+

```bash
sudo pip3 upgrade
sudo pip3 install --upgrade tensorflow-gpu==1.7
```

Validate the installation

```bash
python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

This should output "Hello, TensorFlow!"

### Install ryanawong/tf-pose-estimation

```bash
cd ~
git clone https://github.com/ryanawong/tf-pose-estimation
cd tf-pose-estimation
```

If using Python 2.7

```bash
sudo pip install -r requirements.txt
```

If using Python 3+

```bash
sudo pip3 install -r requirements.txt
```

Setup tf-pose-estimation

```bash
cd models/graph/cmu
bash download.sh
```



## Models

I have tried multiple variations of models to find optmized network architecture. Some of them are below and checkpoint files are provided for research purpose. 

- cmu 
  - the model based VGG pretrained network which described in the original paper.
  - I converted Weights in Caffe format to use in tensorflow.
  - [pretrained weight download](https://www.dropbox.com/s/xh5s7sb7remu8tx/openpose_coco.npy?dl=0)
  
- dsconv
  - Same architecture as the cmu version except for the **depthwise separable convolution** of mobilenet.
  - I trained it using 'transfer learning', but it provides not-enough speed and accuracy.
  
- mobilenet
  - Based on the mobilenet paper, 12 convolutional layers are used as feature-extraction layers.
  - To improve on small person, **minor modification** on the architecture have been made.
  - Three models were learned according to network size parameters.
    - mobilenet
      - 368x368 : [checkpoint weight download](https://www.dropbox.com/s/09xivpuboecge56/mobilenet_0.75_0.50_model-388003.zip?dl=0)
    - mobilenet_fast
    - mobilenet_accurate
  - I published models which is not the best ones, but you can test them before you trained a model from the scratch.

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files. You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

### Inference Time

| Dataset | Model              | Inference Time<br/>Macbook Pro i5 3.1G | Inference Time<br/>Jetson TX2  |
|---------|--------------------|----------------:|----------------:|
| Coco    | cmu                | 10.0s @ 368x368 | OOM   @ 368x368<br/> 5.5s  @ 320x240|
| Coco    | dsconv             | 1.10s @ 368x368 |
| Coco    | mobilenet_accurate | 0.40s @ 368x368 | 0.18s @ 368x368 |
| Coco    | mobilenet          | 0.24s @ 368x368 | 0.10s @ 368x368 |
| Coco    | mobilenet_fast     | 0.16s @ 368x368 | 0.07s @ 368x368 |

## Demo

### Test Inference

You can test the inference feature with a single image.

```
$ python3 run.py --model=mobilenet_thin --resolution=432x368 --image=...
```

The image flag MUST be relative to the src folder with no "~", i.e:
```
--image ../../Desktop
```

Then you will see the screen as below with pafmap, heatmap, result and etc.

![inferent_result](./etcs/inference_result2.png)

### Realtime Webcam

```
$ python3 run_webcam.py --model=mobilenet_thin --resolution=432x368 --camera=0
```

Then you will see the realtime webcam screen with estimated poses as below. This [Realtime Result](./etcs/openpose_macbook13_mobilenet2.gif) was recored on macbook pro 13" with 3.1Ghz Dual-Core CPU.

## Python Usage

This pose estimator provides simple python classes that you can use in your applications.

See [run.py](run.py) or [run_webcam.py](run_webcam.py) as references.

```python
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
humans = e.inference(image)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
```

## ROS Support

See : [etcs/ros.md](./etcs/ros.md)

## Training

See : [etcs/training.md](./etcs/training.md)

## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

[4] Keras Openpose : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation

### Lifting from the deep

[1] Arxiv Paper : https://arxiv.org/abs/1701.00295

[2] https://github.com/DenisTome/Lifting-from-the-Deep-release

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips

[1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

[2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2
