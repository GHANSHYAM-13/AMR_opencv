import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from enum import Enum


class Gesture(Enum):
    PALM     = "PALM"
    FIST     = "FIST"
    THUMB_UP = "THUMB_UP"
    UNKNOWN  = "UNKNOWN"
    NONE     = "NONE"


class HandGestureControl(Node):

    _WRIST            = 0
    _THUMB_TIP        = 4
    _THUMB_IP         = 3
    _THUMB_MCP        = 2
    _PALM_CENTRE      = 9
    _FINGER_TIPS_PIPS = [(8,6),(12,10),(16,14),(20,18)]

    # How many frames a hand must be consistently present before locking
    _LOCK_CONFIRM_N   = 10

    def __init__(self):
        super().__init__('hand_gesture_control')

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw',
            self.depth_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,          # need 2 so we can show both and let user pick
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

        # ── Depth ──────────────────────────────────────────────────────── #
        self.depth_frame: np.ndarray | None = None

        # ── Motion parameters ──────────────────────────────────────────── #
        self.LINEAR_SPEED  = 0.08
        self.ANGULAR_SPEED = 0.20
        self.ACC_STEP      = 0.05
        self.ANGULAR_DEAD  = 0.15
        self.STOP_DISTANCE = 0.5

        self.cur_lin: float = 0.0
        self.cur_ang: float = 0.0

        # ── Gesture debounce ───────────────────────────────────────────── #
        self._DEBOUNCE_N = 4
        self._gesture_buf: deque[Gesture] = deque(maxlen=self._DEBOUNCE_N)
        self._stable: Gesture = Gesture.NONE

        # ── Hand registration state ────────────────────────────────────── #
        # 'label' is MediaPipe handedness string: "Left" or "Right"
        self._registered_hand: str | None  = None   # None = not registered yet
        self._selection_mode:  bool        = True    # True = waiting for user to pick
        self._confirm_buf:     deque[str]  = deque(maxlen=self._LOCK_CONFIRM_N)
        # Store wrist pixel position of registered hand for identity continuity
        self._registered_wrist_px: tuple[int,int] | None = None

        self.get_logger().info(
            "\n────── Hand Gesture Control ──────\n"
            "  Show the hand you want to register\n"
            "  and press  [SPACE]  to lock it.\n"
            "  Press  [R]  anytime to re-register.\n"
            "──────────────────────────────────"
        )

    # ───────────────────────────────────────────────────────────────────── #
    def depth_callback(self, msg: Image) -> None:
        self.depth_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='16UC1')

    # ───────────────────────────────────────────────────────────────────── #
    def rgb_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        # Key handling — non-blocking
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') or key == ord('R'):
            self._reset_registration()
        elif key == ord(' ') and self._selection_mode:
            self._try_register(results, w, h)

        if self._selection_mode:
            self._draw_selection_screen(frame, h, w, results)
        else:
            self._run_control(frame, h, w, results)

        cv2.imshow("Hand Gesture Control", frame)

    # ───────────────────────────────────────────────────────────────────── #
    def _try_register(self, results, w: int, h: int) -> None:
        """Called on SPACE — lock whichever hand is currently most centred."""
        if not results.multi_handedness:
            self.get_logger().warn("No hand visible — cannot register.")
            return

        # If only one hand, pick it immediately
        # If two hands visible, pick the one whose wrist is closest to frame centre
        best_idx   = 0
        best_dist  = float('inf')
        cx, cy     = w // 2, h // 2

        for i, lm_proto in enumerate(results.multi_hand_landmarks):
            wx = int(lm_proto.landmark[self._WRIST].x * w)
            wy = int(lm_proto.landmark[self._WRIST].y * h)
            dist = abs(wx - cx) + abs(wy - cy)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        label = results.multi_handedness[best_idx].classification[0].label
        lm    = results.multi_hand_landmarks[best_idx].landmark

        self._registered_hand    = label
        self._selection_mode     = False
        self._registered_wrist_px = (
            int(lm[self._WRIST].x * w),
            int(lm[self._WRIST].y * h),
        )
        self._gesture_buf.clear()
        self._stable = Gesture.NONE

        self.get_logger().info(f"Hand registered: {label} — control active.")

    # ───────────────────────────────────────────────────────────────────── #
    def _reset_registration(self) -> None:
        self._registered_hand     = None
        self._selection_mode      = True
        self._registered_wrist_px = None
        self._confirm_buf.clear()
        self._gesture_buf.clear()
        self._stable  = Gesture.NONE
        self.cur_lin  = 0.0
        self.cur_ang  = 0.0
        self._publish(0.0, 0.0)   # safety stop
        self.get_logger().info("Registration reset — robot stopped.")

    # ───────────────────────────────────────────────────────────────────── #
    def _find_registered_hand(self, results, w: int, h: int):
        """
        Among all detected hands, return the landmark proto that matches
        the registered handedness label.
        If two hands share the same label (edge case), pick the one whose
        wrist is closest to the last known wrist position.
        Returns None if registered hand not found in this frame.
        """
        if not results.multi_handedness:
            return None

        candidates = []
        for i, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            if label == self._registered_hand:
                candidates.append(results.multi_hand_landmarks[i])

        if not candidates:
            return None

        if len(candidates) == 1 or self._registered_wrist_px is None:
            return candidates[0]

        # Pick closest to last known wrist position
        rx, ry = self._registered_wrist_px
        best, best_dist = candidates[0], float('inf')
        for c in candidates:
            wx = int(c.landmark[self._WRIST].x * w)
            wy = int(c.landmark[self._WRIST].y * h)
            d  = abs(wx - rx) + abs(wy - ry)
            if d < best_dist:
                best_dist = d
                best      = c
        return best

    # ───────────────────────────────────────────────────────────────────── #
    def _run_control(self, frame, h: int, w: int, results) -> None:
        gesture    = Gesture.NONE
        palm_x     = 0.5
        obstacle_m = None
        blocked    = False

        lm_proto = self._find_registered_hand(results, w, h)

        if lm_proto is not None:
            lm = lm_proto.landmark

            # Update last known wrist position
            self._registered_wrist_px = (
                int(lm[self._WRIST].x * w),
                int(lm[self._WRIST].y * h),
            )

            # Draw only the registered hand
            self.mp_draw.draw_landmarks(
                frame, lm_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_style.get_default_hand_landmarks_style(),
                self.mp_style.get_default_hand_connections_style(),
            )

            raw     = self._classify(lm)
            gesture = self._debounce(raw)
            palm_x  = lm[self._PALM_CENTRE].x

            if self.depth_frame is not None:
                cx = int(lm[self._PALM_CENTRE].x * w)
                cy = int(lm[self._PALM_CENTRE].y * h)
                obstacle_m = self._sample_depth(cx, cy)

        else:
            self._gesture_buf.clear()
            self._stable = Gesture.NONE

        target_lin, target_ang, blocked = self._resolve(
            gesture, palm_x, obstacle_m)

        self.cur_lin = self._smooth(target_lin, self.cur_lin)
        self.cur_ang = self._smooth(target_ang, self.cur_ang)
        self._publish(self.cur_lin, self.cur_ang)

        self._draw_control_hud(frame, h, w, gesture,
                               obstacle_m, blocked,
                               self.cur_lin, self.cur_ang)

    # ───────────────────────────────────────────────────────────────────── #
    def _classify(self, lm) -> Gesture:
        fingers_curled = all(
            lm[tip].y > lm[pip].y
            for tip, pip in self._FINGER_TIPS_PIPS
        )
        thumb_raised = (lm[self._WRIST].y - lm[self._THUMB_TIP].y) > 0.15

        if fingers_curled and thumb_raised:
            return Gesture.THUMB_UP

        thumb_curled = lm[self._THUMB_TIP].y > lm[self._THUMB_MCP].y
        if fingers_curled and thumb_curled:
            return Gesture.FIST

        fingers_up = sum(
            1 if lm[tip].y < lm[pip].y else 0
            for tip, pip in self._FINGER_TIPS_PIPS
        )
        if fingers_up >= 3:
            return Gesture.PALM

        return Gesture.UNKNOWN

    # ───────────────────────────────────────────────────────────────────── #
    def _debounce(self, raw: Gesture) -> Gesture:
        self._gesture_buf.append(raw)
        if (len(self._gesture_buf) == self._DEBOUNCE_N and
                self._gesture_buf.count(raw) == self._DEBOUNCE_N):
            self._stable = raw
        return self._stable

    # ───────────────────────────────────────────────────────────────────── #
    def _resolve(self, gesture: Gesture, palm_x: float,
                 obstacle_m: float | None) -> tuple[float, float, bool]:
        blocked = False

        if gesture == Gesture.PALM:
            if obstacle_m is not None and obstacle_m < self.STOP_DISTANCE:
                blocked    = True
                target_lin = 0.0
            else:
                target_lin = self.LINEAR_SPEED

            error_x    = palm_x - 0.5
            target_ang = (
                -self.ANGULAR_SPEED * (error_x / 0.5)
                if abs(error_x) > self.ANGULAR_DEAD else 0.0
            )

        elif gesture == Gesture.FIST:
            target_lin = -self.LINEAR_SPEED
            target_ang = 0.0

        else:
            target_lin = 0.0
            target_ang = 0.0

        return target_lin, target_ang, blocked

    # ───────────────────────────────────────────────────────────────────── #
    def _sample_depth(self, cx: int, cy: int, radius: int = 4) -> float | None:
        dh, dw = self.depth_frame.shape[:2]
        x0, x1 = max(0, cx-radius), min(dw, cx+radius+1)
        y0, y1 = max(0, cy-radius), min(dh, cy+radius+1)
        patch  = self.depth_frame[y0:y1, x0:x1].flatten().astype(np.float32)
        valid  = patch[patch > 0]
        return float(np.median(valid)) / 1000.0 if valid.size > 0 else None

    # ───────────────────────────────────────────────────────────────────── #
    def _smooth(self, target: float, current: float) -> float:
        diff = target - current
        if abs(diff) <= self.ACC_STEP:
            return target
        return current + self.ACC_STEP * float(np.sign(diff))

    # ───────────────────────────────────────────────────────────────────── #
    def _publish(self, lin: float, ang: float) -> None:
        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

    # ───────────────────────────────────────────────────────────────────── #
    def _draw_selection_screen(self, frame, h: int, w: int, results) -> None:
        """
        Overlay shown before hand is registered.
        Draws all detected hands with their handedness label + bounding box.
        Prompts user to press SPACE to lock.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, "HAND REGISTRATION MODE",
                    (w//2 - 195, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Show your hand  →  press SPACE to lock",
                    (w//2 - 230, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (200, 200, 200), 1)

        if results.multi_hand_landmarks:
            for i, (lm_proto, handedness) in enumerate(
                    zip(results.multi_hand_landmarks,
                        results.multi_handedness)):

                label = handedness.classification[0].label
                score = handedness.classification[0].score
                lm    = lm_proto.landmark

                # Bounding box from landmarks
                xs = [int(p.x * w) for p in lm]
                ys = [int(p.y * h) for p in lm]
                x1, y1 = max(0, min(xs)-20), max(0, min(ys)-20)
                x2, y2 = min(w, max(xs)+20), min(h, max(ys)+20)

                box_clr = (0, 255, 0) if label == "Right" else (255, 100, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_clr, 2)
                cv2.putText(frame, f"{label}  ({score:.0%})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_clr, 2)

                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame, lm_proto,
                    self.mp_hands.HAND_CONNECTIONS)

        else:
            cv2.putText(frame, "No hand detected",
                        (w//2 - 110, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, "[R] Re-register",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (160, 160, 160), 1)

    # ───────────────────────────────────────────────────────────────────── #
    def _draw_control_hud(self, frame, h: int, w: int,
                          gesture: Gesture, obstacle_m: float | None,
                          blocked: bool, lin: float, ang: float) -> None:

        clr_map = {
            Gesture.PALM:     (0,   255, 0  ),
            Gesture.FIST:     (0,   100, 255),
            Gesture.THUMB_UP: (160, 160, 160),
            Gesture.UNKNOWN:  (0,   165, 255),
            Gesture.NONE:     (100, 100, 100),
        }
        clr = clr_map[gesture]

        # Registered hand badge (top right)
        badge = f"Tracking: {self._registered_hand} hand"
        cv2.putText(frame, badge,
                    (w - 260, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2)

        if blocked:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 160), -1)
            cv2.putText(frame,
                        f"OBSTACLE — {obstacle_m:.2f}m  (limit {self.STOP_DISTANCE}m)",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Gesture : {gesture.value}",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clr, 2)

        depth_str = f"{obstacle_m:.2f}m" if obstacle_m is not None else "no depth"
        cv2.putText(frame, f"Depth   : {depth_str}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 200, 255) if blocked else (200, 200, 200), 2)
        cv2.putText(frame, f"Lin={lin:+.3f}  Ang={ang:+.3f}",
                    (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200, 200, 0), 2)

        # Legend
        legend = [
            (Gesture.PALM,     "PALM     → FORWARD"),
            (Gesture.FIST,     "FIST     → BACKWARD"),
            (Gesture.THUMB_UP, "THUMB UP → STOP"),
        ]
        base_y = h - len(legend) * 24 - 10
        for i, (g, text) in enumerate(legend):
            y   = base_y + i * 24
            col = clr_map[g]
            if gesture == g:
                cv2.rectangle(frame, (4, y-17), (235, y+6), col, -1)
                cv2.putText(frame, text, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
            else:
                cv2.putText(frame, text, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        # Distance bar
        bx, by_b, bw, bh = w-28, 20, 16, h-40
        cv2.rectangle(frame, (bx, by_b), (bx+bw, by_b+bh), (50,50,50), -1)
        max_range = 3.0
        stop_px   = int(by_b + bh * (1.0 - self.STOP_DISTANCE / max_range))
        cv2.rectangle(frame, (bx, by_b),   (bx+bw, stop_px),  (0,180,0), -1)
        cv2.rectangle(frame, (bx, stop_px),(bx+bw, by_b+bh),  (0,0,180), -1)
        if obstacle_m is not None:
            frac  = min(obstacle_m / max_range, 1.0)
            dot_y = int(by_b + bh * (1.0 - frac))
            cv2.circle(frame, (bx+bw//2, dot_y), 6, (0,0,0),   -1)
            cv2.circle(frame, (bx+bw//2, dot_y), 5,
                       (0,0,255) if blocked else (255,255,255), -1)
        cv2.putText(frame, f"{self.STOP_DISTANCE:.1f}m",
                    (bx-40, stop_px+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

        cv2.putText(frame, "[R] Re-register",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (160,160,160), 1)


# ─────────────────────────────────────────────────────────────────────── #
def main(args=None):
    rclpy.init(args=args)
    node = HandGestureControl()
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