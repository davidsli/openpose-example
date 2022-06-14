# openpose-example

What is OpenPose? See the github repository of OpenPose: [CMU-Perceptual-Computing-Lab/openpose][openpose]

I use the openpose model that is pretrained using MPII Human Pose Dataset.

## Getting started

Requirements:

- opencv-python
- numpy

If you use poetry, just run `poetry install`

To run this code, you need to download the pretrained model with the following command.

```sh
python src/get_model.py
```

And run the following command.

```sh
python src/stop-dumping.py --url rtsp://YourIpCameraAddress:554
```

[openpose]: https://github.com/CMU-Perceptual-Computing-Lab/openpose
