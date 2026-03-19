# Hand Gesture Robot Control — ROS 2

Control a differential drive robot using hand gestures via a RealSense RGB-D camera and MediaPipe.

## Gesture Map

| Gesture | Action |
|---------|--------|
| Open palm | Forward (stops if obstacle within stop distance) |
| Fist | Backward |
| Thumb up | Stop |
| Palm X offset | Turn left / right |

## Dependencies

```bash
pip install mediapipe opencv-python numpy
sudo apt install ros-$ROS_DISTRO-cv-bridge
```

## Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | Subscribe |
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/Image` | Subscribe |
| `/cmd_vel` | `geometry_msgs/Twist` | Publish |

## Run

```bash
ros2 run <your_package> hand_gesture_control
```

## Controls (Window)

- `[1]` / `[2]` — select hand to track when multiple hands visible
- `[SPACE]` — auto-lock most centred hand
- `[R]` — re-register hand
- `[Q]` — quit
- Trackbars — tune linear speed, angular speed, stop distance live

## Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LINEAR_SPEED` | 0.12 m/s | Forward / backward speed |
| `ANGULAR_SPEED` | 0.45 rad/s | Turn rate |
| `STOP_DISTANCE` | 0.50 m | Depth threshold to halt forward motion |
| `DEBOUNCE_N` | 4 frames | Gesture stability before command fires |


######################################################################################################################################################
######################################################################################################################################################


# Red Ball Tracker — ROS 2

Autonomously follows a red ball using a RealSense RGB-D camera. Uses HSV colour segmentation for detection and depth for distance-based stopping.

## Behaviour

- Detects the largest circular red object in frame
- Rotates to keep the ball centred (proportional angular control)
- Drives forward until depth reading drops below `STOP_DISTANCE`
- Stops and waits if ball is lost

## Dependencies

```bash
pip install opencv-python numpy
sudo apt install ros-$ROS_DISTRO-cv-bridge
```

## Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | Subscribe |
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/Image` | Subscribe |
| `/cmd_vel` | `geometry_msgs/Twist` | Publish |

## Run

```bash
ros2 run <your_package> red_ball_tracker
```

## Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LINEAR_SPEED` | 0.12 m/s | Forward speed toward ball |
| `ANGULAR_SPEED` | 0.50 rad/s | Rotation rate to centre ball |
| `STOP_DISTANCE` | 0.60 m | Stop when ball closer than this |
| `MIN_RADIUS` | 12 px | Ignore detections smaller than this |
| `HSV_LOW1/HIGH1` | H 0–10 | Lower red hue band |
| `HSV_LOW2/HIGH2` | H 165–180 | Upper red hue band (red wraps at 180) |

> **Lighting tip:** If detection is unreliable, print `hsv[cy,cx]` at the ball centre and adjust the HSV ranges to match your environment.
