import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np


class RedBallTracker(Node):

    def __init__(self):
        super().__init__('red_ball_tracker')

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw',
            self.depth_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.depth_frame: np.ndarray | None = None

        # Motion params
        self.LINEAR_SPEED  = 0.07   # m/s
        self.ANGULAR_SPEED = 0.25   # rad/s
        self.STOP_DISTANCE = 0.60   # metres
        self.ANG_DEAD      = 0.08   # normalised X dead band
        self.ACC_STEP      = 0.04

        self.cur_lin = 0.0
        self.cur_ang = 0.0

        # HSV — two ranges because red wraps 0/180
        self.HSV_LOW1  = np.array([ 0, 120,  60])
        self.HSV_HIGH1 = np.array([10, 255, 255])
        self.HSV_LOW2  = np.array([165, 120,  60])
        self.HSV_HIGH2 = np.array([180, 255, 255])
        self.MIN_RADIUS = 12   # px

        self.get_logger().info("Red Ball Tracker started.")

    # ------------------------------------------------------------------ #
    def depth_callback(self, msg: Image) -> None:
        self.depth_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='16UC1')

    # ------------------------------------------------------------------ #
    def rgb_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, _ = frame.shape

        centre, radius = self._detect(frame)

        target_lin = 0.0
        target_ang = 0.0

        if centre is not None:
            cx, cy = centre

            # Angular — proportional to X error from frame centre
            error_x = (cx / w) - 0.5
            if abs(error_x) > self.ANG_DEAD:
                target_ang = -self.ANGULAR_SPEED * (error_x / 0.5)

            # Linear — forward unless depth says too close
            depth = self._sample_depth(cx, cy, w, h, radius)
            if depth is None or depth > self.STOP_DISTANCE:
                target_lin = self.LINEAR_SPEED

            # Annotate
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4,      (0, 255, 0), -1)
            d_str = f"{depth:.2f}m" if depth else "--"
            cv2.putText(frame, d_str,
                        (cx - 25, cy - radius - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "searching...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 200), 1, cv2.LINE_AA)

        # Velocity ramp
        self.cur_lin = self._smooth(target_lin, self.cur_lin)
        self.cur_ang = self._smooth(target_ang, self.cur_ang)
        self._publish(self.cur_lin, self.cur_ang)

        cv2.imshow("Red Ball Tracker", frame)
        cv2.waitKey(1)

    # ------------------------------------------------------------------ #
    def _detect(self, frame: np.ndarray) -> tuple:
        blurred = cv2.GaussianBlur(frame, (9, 9), 2)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.HSV_LOW1, self.HSV_HIGH1),
            cv2.inRange(hsv, self.HSV_LOW2, self.HSV_HIGH2),
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0

        c = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        r = int(r)

        if r < self.MIN_RADIUS:
            return None, 0

        # Circularity check — reject non-round blobs
        area         = cv2.contourArea(c)
        circularity  = (4 * np.pi * area) / (cv2.arcLength(c, True)**2 + 1e-6)
        if circularity < 0.55:
            return None, 0

        return (int(cx), int(cy)), r

    # ------------------------------------------------------------------ #
    def _sample_depth(self, cx: int, cy: int,
                      w: int, h: int, radius: int = 6) -> float | None:
        if self.depth_frame is None:
            return None

        dh, dw = self.depth_frame.shape[:2]

        # Scale RGB pixel → depth pixel (resolutions may differ)
        dcx = int(cx * dw / w)
        dcy = int(cy * dh / h)

        r = max(4, radius // 2)
        x0, x1 = max(0, dcx-r), min(dw, dcx+r+1)
        y0, y1 = max(0, dcy-r), min(dh, dcy+r+1)

        patch = self.depth_frame[y0:y1, x0:x1].flatten().astype(np.float32)
        valid = patch[patch > 0]

        return float(np.median(valid)) / 1000.0 if valid.size > 0 else None

    # ------------------------------------------------------------------ #
    def _smooth(self, target: float, current: float) -> float:
        diff = target - current
        if abs(diff) <= self.ACC_STEP:
            return target
        return current + self.ACC_STEP * float(np.sign(diff))

    # ------------------------------------------------------------------ #
    def _publish(self, lin: float, ang: float) -> None:
        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)


# ------------------------------------------------------------------ #
def main(args=None):
    rclpy.init(args=args)
    node = RedBallTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
