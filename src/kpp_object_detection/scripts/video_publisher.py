#!/usr/bin/env python3
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main():
    rospy.init_node("video_publisher")
    path = rospy.get_param("~video")
    topic = rospy.get_param("~topic", "/camera/image_raw")
    loop = rospy.get_param("~loop", False)

    pub = rospy.Publisher(topic, Image, queue_size=1)
    bridge = CvBridge()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        rospy.logfatal("Cannot open video: %s", path)
        return

    rate = rospy.Rate(cap.get(cv2.CAP_PROP_FPS) or 30)
    while not rospy.is_shutdown():
        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        rate.sleep()
    cap.release()

if __name__ == "__main__":
    main()