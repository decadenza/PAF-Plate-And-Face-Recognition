# PAF Plate and Face Recognition
A complete GUI tool to recognize and detect plates and faces in video streaming and files.

This software was tested on _Linux Debian 9_. If you want to use it on other OS, you may use a virtual machine.

## Installation
Please run this commands:
```
pip install numpy
sudo apt install cmake
pip install dlib # In case of problems compile from source...
pip install opencv-python
sudo apt-get install python3-pyqt5 # In case of problems compile from source...
```
Then you'll need to install OpenALPR, so execute these commands:
```
sudo apt-get install libopencv-dev libtesseract-dev git cmake build-essential libleptonica-dev
sudo apt-get install liblog4cplus-dev libcurl3-dev
```
Clone the latest code from GitHub
```
git clone https://github.com/openalpr/openalpr.git
cd openalpr/src
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc ..
make
sudo make install
```
To test the Alpr library you may try this:
```
alpr TEST_IMAGE.jpg
```
Finally, install python bindings:
```
cd ../bindings/python
sudo python3 setup.py install
```

## Usage and screenshots
From the home you can configure and controll up to 4 cameras.
![Home](/Screenshots/home.png?raw=true "Home")

Each camera can be configured clicking on the corresponding gear symbol. You must enter a URL (rtsp, http or other protocol supported by OpenCV) including credentials e.g. _username:myStrongPassword@12.34.56.78/video/live_.
You may want to detect also unknown faces/plates (i.e. faces and plates that are not a target).
You can also select a ROI (region of interest).
![Configure camera](/Screenshots/cameraconfig.png?raw=true "Camera configuration")

From the home, clicking on the rightmost button of each camera you can see all the events. At bottom left there is a button to delete all the events stored with that camera. 
![Events](/Screenshots/events.png?raw=true "Camera events")

You can process video files too. From menu, just select "Process files". The options are similar to the ones above, but you need to set a output destination.

![File process](/Screenshots/fileprocess.png?raw=true "File process")



## Considerations
1) When using live video the software is __not__ using a buffer. It takes the current frame from the camera. This means that you may lose a face or a plate, because the algorithm can not process 25 frames per seconds on a common machine.

2) Video file processing, instead, processes __every__ frame found in the video file(s). It will use all the CPUs available in parallel to speed up processing.

3) No software is free of bugs. Please report issues!


Enjoy,
Pasquale Lafiosca
