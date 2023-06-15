# Depth Completion of Transparent Objects using Augmented Unpaired Data
ICRA 2023 release of depth completion of transparent objects using augmented unpaired data.

To install, clone this repository and run `pip install -r requirements.txt`. Probably best to run it in a conda, virtual or docker environment :)

Models that convert from RGBD to RGBD can be found in the folder `RGBD2RGBD`. Models that convert from depth to depth can be found in the folder `Depth2Depth`.

Some notebooks for evaluating the results are also included, but these notebooks are quite bare.

For more details (and datasets), please check our [web site](https://florise.github.io/faking_depth_web/).

If you want to cite this work:

```
@inproceedings{erich2023depth,
    title={Learning Depth Completion of Transparent Objects using Augmented Unpaired Data},
    author={Erich, Floris and Leme, Bruno and Ando, Noriaki and Hanai, Ryo and Domae, Yukiyasu},
    year=2023,
    booktitle  = {2023 IEEE International Conference on Robotics and Automation (ICRA)},
    publisher = {IEEE}
}
```
