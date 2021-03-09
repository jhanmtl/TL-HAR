sudo xhost +local:docker

sudo docker run \
--rm \
-ti \
--net=host \
--ipc=host \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/jay/school/winter2021/research/har/tl-har:/home/jay/school/winter2021/research/har/tl-har \
--env QT_X11_NO_MITSHM=1 \
--device=/dev/video0 \
--gpus all \
comp598/tensorflow:gpu

