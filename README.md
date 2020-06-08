## Dependencies

Install the following dependencies to run the program. PIP is recommended

* opencv-python
* imageio
* numpy
* numba

## Running the program

Any video can be resized, but we have provided some in this repository (alongside example execution below).

Here are a list of the parameters needed for the program to run
| Arg     | Description | Default Value  |
| ------- | ----------- | -------------- |
| --video | Video to read in to program        | None: Required |
| --width | Width to retarget to        | None: Required |
| --height | Height to retarget to | None: Required |
| --out   | Output video name | None: Required |
| --window | Window that determines how far apart piecewise spatial seams can be | 10 |
| --saliencyW | Weight for saliency measure | 2 |
| --spatialW | Weight for spatial measure | 5 |
| --temporalW | Weight for temporal measure | 0.5 |

### Example execution
This will resize a provided video from 720x480 to 660x420
```
python disc_video_carving.py --video spacex_launch480.m4v --window 5 --width 660 --height 420 --out spacex_launch_retargeted_both.mp4
```

## Example output

All example output is provided in the EXAMPLES folder. The output will in .mp4 format
