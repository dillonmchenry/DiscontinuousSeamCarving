## Dependencies

Install the following dependencies to run the program. PIP is recommended

* opencv-python
* imageio
* numpy
* numba

## Running the program

Here are a list of the parameters needed for the program to run
| Arg     | Description | Default Value  |
| ------- | ----------- | -------------- |
| --video | Test        | None: Required |
 
    parser.add_argument('--video', type=str, help='The path to the video to retarget')
    parser.add_argument('--width', type=int, help='Width to retarget video to')
    parser.add_argument('--height', type=int, help='Height to retarget video to')
    parser.add_argument('--out', type=str, help='The path to store the output to')
    parser.add_argument('--window', type=int, help='Window for piecewise seams', default=10)
    parser.add_argument('--saliencyW', type=float, help='Saliency Weight in seam carving', default=2)
    parser.add_argument('--spatialW', type=float, help='Spatial Weight in seam carving', default=5)
    parser.add_argument('--temporalW', type=float, help='Temporal Weight in seaming carving', default=0.5)


The disc_video file is our project, the other PY is a reference to the image method implemented.

Right now I just have video I/O and a list of tasks for us.
