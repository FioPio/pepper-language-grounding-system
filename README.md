# About this repo
This repository holds the software implementation used to solve my master thesis **Language grounding for robotics** done in the University of Umeå. It tries to use the robot Pepper (from SoftBank Robotics) to perform language grounding offline. 

The main idea is that there are three different Docker containers, and each one of them performs a different task:

- **pepper** This docker is the one that interacts with the robot Pepper using the Naoqi software.
- **server-sr** This one contains a server that uses the [vosk-api](https://github.com/alphacep/vosk-api) to perform offline speech recognition.
- **server-lg** This is the server that tries to identify which object is the required from the image with the provided description using the software [zsgnet](https://github.com/TheShadow29/zsgnet-pytorch).

This is a small representation of the configuration for this project:

![alt text](https://github.com/FioPio/pepper-language-grounding-system/blob/main/resources/nodesSetup.png?raw=true)


# Installation
The requirements for this project are:

- Docker (To run the different parts)
- Cuda 10.0 (For the language grounding part)


First of all the repository must be cloned in order to use it:
```
git clone https://github.com/FioPio/pepper-language-grounding-system.git
```


## pepper node
For the image **pepper** the ip `172.18.48.249` should be changed to your **Pepper IP address** in `pepper/Dockerfile` line `42` and in `pepper/main.py` line `32`.

After that, the docker may be build opening a terminal inside `pepper/` and typing:
```
docker build -t pepper .
```


After that, the public key must be copied into the Pepper computer. To obtain the public key of the container you need to open a terminal in your computer and type:

```
docker run pepper cat /root/.ssh/id_rsa.pub
```

Then copy the output and connect by ssh to the robot Pepper. If `/home/nao/.ssh/authorized_keys` exist, just open it and paste the content of your `id_rsa.pub`.

If the directory `.ssh` does not exist create it
```
mkdir \home\nao\.ssh
```

if file `authorized_keys` does not exist create it (substitute `HERE YOUR PUBLIC KEY` by the content of your `id_rsa.pub`)

```
echo "HERE YOUR PUBLIC KEY">/home/nao/.ssh/authorized_keys
```

Finally change the permisions
```
chmod 700 /home/nao/.ssh
chmod 600 /home/nao/.ssh/authorized_keys
```

## server-sr node
To build this docker image first you need to download a model from [here](https://alphacephei.com/vosk/models) and extract it into `server-sr/`.  You should rename the directory that contains it to `model-en`.

After that, the image may be built by entering into `server-sr/` and typing 
```
docker build -t server-sr .
```

## server-lg  node

To be able to run this node, fist you need to train a model as explained [here](https://github.com/TheShadow29/zsgnet-pytorch) and place it in `server-lg/`. The Dockerfile expects the model `referit_try.pth`, but it may be changed in `server-lg/Dockerfile` line`82`.

Finally you may build the image  by entering into `server-lg/` and typing 
```
docker build -t server-lg .
```


# Running the software

As soon as everything is installed and configured, the software may run. First of all I advice to start the `server-lg` since it takes a while to load the models

```
docker run --gpus all -p 5001:5000 server-lg
```

After that, the `server-sr` may be started:
```
docker run -p 5002:5000 server-sr
```

And once both servers are available, the node `pepper`may be started. First allow the docker container to connect to your screen

```
xhost +local:docker
```
Then, in the same terminal run the container with:
```
docker run --network host -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 pepper
``` 
After execution, you may recovery your original configuration with:
```
xhost -
```

 
# Demo

There is a small demo using this project:

[![Pepper language grounding system](https://img.youtube.com/watch?v=LLB2ebpyLgA/0.jpg)](https://www.youtube.com/watch?v=LLB2ebpyLgA "Pepper language grounding system")
