# KPP Object Detection
This was made to fullfill an intership assignment at Abinara-1 ITS Team, integrating ROS Noetic, OpenCV, and YOLO.

Each problem is a python module that exposes one function (`process(frame)`). A publisher node streams the frames of a video and a subscriber node runs the appropriate solver on every frame and displays the result.

## Problems Overview
 # | Problem | Module | Method |
|---|--------|--------|--------|
| 1 | Where Are The Shapes? | `shapes.py` | Normallized distance from the background color, contours, and polygon approximation |
| 2 | How Colorful? | `colorful.py` | 5 bit color quantization |
| 3 | Locate It, Please. | `locate.py` | Brightness threshold and contours |
| 4 | I Don't Think It's Reliable. | `reliable.py` | Threshold and morphological opening and closing |
| 5 | Does It Match? | `match.py` | Trained YOLO model exported to ONNX, run with ONNX Runtime |

## Requirements
 
- Ubuntu 20.04 with ROS Noetic, Python 3.8
- `ros-noetic-desktop-full`
- For problem 5 run: `pip3 install --user onnxruntime`

## Build and run
At the project root run:
```bash
catkin_make
source devel/setup.bash
```
 
The input videos are not stored in the repository. You need to pass the path of the video with the
`video:=` argument. Make sure to use an absolute path.
 
| # | Launch file | Example input |
|---|-------------|---------------|
| 1 | `shapes.launch` | `shape_detection_static.mp4`, `shape_detection_transition.mp4` |
| 2 | `colorful.launch` | `Lucky_Charms_Diamond_Dancers.mp4` |
| 3 | `locate.launch` | `diamond.mp4` |
| 4 | `reliable.launch` | `noisy_mask.mp4` |
| 5 | `match.launch` | `gameplay.mp4` |
 
### Example:
```bash
roslaunch kpp_object_detection shapes.launch video:=/absolute/path/to/shape_detection_static.mp4
```
 
## Project Structure
The video files and training data in `kpp_desc/` are not uploaded to the repo.
```
kpp-object-detection/
├── kpp_desc
│   ├── 1WhereAreTheShapes
│   │   ├── shape_detection_static.mp4
│   │   ├── shape_detection_transition.mp4
│   │   └── Where Are The Shapes.pdf
│   ├── 2HowColorful
│   │   ├── How Colorful.pdf
│   │   └── Lucky Charms Diamond Dancers.mp4
│   ├── 3LocateItPlease
│   │   ├── diamond.mp4
│   │   └── Locate It Please.pdf
│   ├── 4IDontThinkItsReliable
│   │   ├── I Don_t Think It_s Reliable.pdf
│   │   └── noisy_mask.mp4
│   ├── 5DoesItMatch
│   │   ├── Does it Match.pdf
│   │   ├── gameplay.mp4
│   │   └── Hollow Knight.yolo26.zip
│   └── KPP Object Detection.pdf
├── README.md
└── src
    └── kpp_object_detection
        ├── CMakeLists.txt
        ├── launch
        │   ├── colorful.launch
        │   ├── locate.launch
        │   ├── match.launch
        │   ├── reliable.launch
        │   └── shapes.launch
        ├── models
        │   └── best.onnx
        ├── package.xml
        ├── scripts
        │   ├── solver_node.py
        │   └── video_publisher.py
        ├── setup.py
        └── src
            └── kpp_object_detection
                ├── colorful.py
                ├── __init__.py
                ├── locate.py
                ├── match.py
                ├── __pycache__
                │   ├── colorful.cpython-38.pyc
                │   ├── locate.cpython-38.pyc
                │   ├── match.cpython-38.pyc
                │   ├── reliable.cpython-38.pyc
                │   └── shapes.cpython-38.pyc
                ├── reliable.py
                └── shapes.py
```
