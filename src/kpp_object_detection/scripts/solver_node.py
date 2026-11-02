#!/usr/bin/env python3
import importlib
import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class SolverNode:
    def __init__(self):
        name = rospy.get_param("~solver")
        topic = rospy.get_param("~topic", "/camera/image_raw")
        self.process = importlib.import_module(f"kpp_object_detection.{name}").process
        self.bridge = CvBridge()
        self.title = name
        self.last = time.time()
        rospy.Subscriber(topic, Image, self.callback, queue_size=1, buff_size=2**24)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        out = self.process(frame)

        now = time.time()
        fps = 1.0 / max(now - self.last, 1e-6)
        self.last = now
        cv2.putText(out, f"{fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(self.title, out)
        cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("solver_node")
    SolverNode()
    rospy.spin()