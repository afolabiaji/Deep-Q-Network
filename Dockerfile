FROM ubuntu
COPY . .
RUN sudo apt-get update
RUN sudo apt-get install ffmpeg libsm6 libxext6 xvfb  -y
RUN sudo apt-get install python-opengl
RUN sudo apt install cmake swig zlib1g-dev python3-tk -y
RUN sudo apt install ffmpeg
RUN sudo apt-get install ffmpeg libsm6 libxext6  -y
RUN sudo apt install python-opengl
RUN sudo apt install ffmpeg
RUN sudo apt install xvfb
RUN pip -r install requiremnents.txt