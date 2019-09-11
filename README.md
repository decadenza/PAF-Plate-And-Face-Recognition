## PAF Plate and Face Recognition
A complete GUI tool to recognize and detect plates and faces in video streaming and files.

This software was tested on Linux Debian 9. If you want to use it on other OS, you may use a virtual machine.

# Installation
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

