# base image
FROM python:2.7

# Download latest listing of available packages:
RUN apt-get -y update

# Install package required
RUN apt-get -y install \ 
	libportaudio2 \
	libasound-dev\
	libgtk2.0-dev\
	pkg-config\
	libopencv-dev
	
# Setting software directory
RUN mkdir /source
WORKDIR /source

# add naoqi OS
ADD https://community-static.aldebaran.com/resources/2.5.10/Python%20SDK/pynaoqi-python2.7-2.5.7.1-linux64.tar.gz .
RUN tar -xf pynaoqi-python2.7-2.5.7.1-linux64.tar.gz

# set environment variables for python to find naoqi api
ENV PYTHONPATH=/source/pynaoqi-python2.7-2.5.7.1-linux64/lib/python2.7/site-packages
ENV DYLD_LIBRARY_PATH=/source/pynaoqi-python2.7-2.5.7.1-linux64/lib

# Adding the main script
ADD main.py .

# Installing required libraries
RUN pip install requests wave numpy Pillow opencv-python==4.0.0.21 opencv-contrib-python==4.0.0.21

# Setting up ssh
RUN apt install openssh-client

# Generating hey pairs and setting the right permisions
RUN ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa &&  chmod 600 /root/.ssh/id_rsa &&  chmod 600 /root/.ssh/id_rsa.pub

# Adding the pepper as a known host   
RUN  echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan -H 172.18.48.249 >> /root/.ssh/known_hosts

RUN echo "QT_X11_NO_MITSHM=1">>/etc/environment

#ENTRYPOINT ["python", "main.py"]
