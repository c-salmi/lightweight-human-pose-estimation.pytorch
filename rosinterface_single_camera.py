import argparse
import time
from copy import copy

import cv2
import numpy as np
import torch
import rospy
import tf
from sensor_msgs.msg import Image, CameraInfo
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from geometry_msgs.msg import PoseStamped

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class RosOpenPoseLight:
    def __init__(self, net, camera_num, net_input_height_size=256, stride=8, upsample_ratio=4):
        rospy.init_node(f"openpose_light_camera{camera_num}")

        self.net = net
        self.net = self.net.eval()
        self.cpu = False
        if not self.cpu:
            self.net = self.net.cuda()
        self.net_input_height_size = net_input_height_size
        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.num_keypoints = Pose.num_kpts
        self.camera_num = camera_num
        self.delay = 1
        self.cv2viz = False

        self.color_msg = None
        self.depth_msg = None
        self.K = None

        # ROS subscribers
        rospy.Subscriber(f'/camera{camera_num}/color/image_raw', Image, self.color_cb, queue_size=1)
        rospy.Subscriber(f'/camera{camera_num}/aligned_depth_to_color/image_raw', Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(f'/camera{camera_num}/color/camera_info', CameraInfo, self.info_cb, queue_size=1)

        # ROS publishers
        # self.pub_persons = rospy.Publisher('/openpose/tracked_persons', TrackedPersons, queue_size=10)
        self.pub_persons = rospy.Publisher(f'/camera{camera_num}/openpose/detections', BoundingBoxArray, queue_size=10)

        self.listener = tf.TransformListener()

    def color_cb(self, msg):
        self.color_msg = msg
        # self.color_imgs[args['camera_num']] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # self.color_time = msg.header.stamp

    def depth_cb(self, msg):
        self.depth_msg = msg
        # self.depth_imgs[args['camera_num']] = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        # self.depth_time = msg.header.stamp

    def info_cb(self, msg):
        self.K = np.array(msg.K).reshape((3, 3))

    def run_demo(self, track=False, smooth=False):
        color_img = np.frombuffer(self.color_msg.data, dtype=np.uint8).reshape(self.color_msg.height, self.color_msg.width, -1)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        if self.cv2viz:
            orig_img = color_img.copy()
        depth_img = np.frombuffer(self.depth_msg.data, dtype=np.uint16).reshape(self.depth_msg.height, self.depth_msg.width, -1)
        # depth_imgs = np.array(self.depth_imgs)

        heatmap, paf, scale, pad = self.infer_fast(color_img)

        # for i, (heatmap, paf, depth_img) in enumerate(zip(heatmaps, pafs, depth_imgs)):
        current_poses = self.extract_poses_single_camera(heatmap, paf, scale, pad, depth_img)
        # TODO: handle duplicate poses

        # TODO: track previous poses

        if self.cv2viz:
            for pose in current_poses:
                pose.draw(color_img)
            color_img = cv2.addWeighted(orig_img, 0.6, color_img, 0.4, 0)
            for pose in current_poses:
                cv2.rectangle(color_img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', color_img)
            key = cv2.waitKey(self.delay)
            if key == 27:  # esc
                return
            elif key == 112:  # 'p'
                if self.delay == 1:
                    self.delay = 0
                else:
                    self.delay = 1

        self.viz_poses(current_poses)

    def infer_fast(self, img, pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, width, _ = img.shape
        scale = self.net_input_height_size / height

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [self.net_input_height_size, max(scaled_img.shape[1], self.net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, self.stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

        if not self.cpu:
            tensor_img = tensor_img.cuda()
        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        t = stage2_heatmaps.squeeze().cpu().data.numpy()
        heatmaps = np.transpose(t, (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def extract_poses_single_camera(self, heatmap, paf, scale, pad, depth_img):
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmap[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, paf)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        for pose in current_poses:
            # Extract median depth value of keypoints
            pose.skip = False
            keypoint_depths = [depth_img[pose_keypoint[1], pose_keypoint[0]] for pose_keypoint in pose.keypoints if not all(pose_keypoint == [-1, -1]) and not depth_img[pose_keypoint[1], pose_keypoint[0]] == 0]
            if len(keypoint_depths) == 0:
                pose.skip = True
                continue
            avg_depth = np.median(keypoint_depths)
            point = pose.keypoints[0] if not all(pose.keypoints[0] == [-1, -1]) else pose.keypoints[1]

            point3d_camera = self.calculate_depth(point, avg_depth/1000)
            pose.point3d_camera = point3d_camera

            # Create pose
            pose3d = PoseStamped()
            pose3d.header.frame_id = f'/camera{self.camera_num}_link'
            pose3d.pose.position.x = point3d_camera[0]
            pose3d.pose.position.y = point3d_camera[1]
            pose3d.pose.position.z = point3d_camera[2]
            pose.pose3d_camera = pose3d
            pose.pose3d_map = self.listener.transformPose('map', pose3d)

        return current_poses

    def calculate_depth(self, keypoint, z):
        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        x = z / fx * (keypoint[0] - cx)
        y = z / fy * (keypoint[1] - cy)

        return [z, -x, -y]

    def viz_poses(self, poses):
        persons = BoundingBoxArray()
        persons.header.frame_id = 'map'
        for p in poses:
            if p.skip: continue
            person = BoundingBox()
            person.header.frame_id = 'map'
            person.pose.position = p.pose3d_map.pose.position
            person.pose.position.z -= 0.5
            person.dimensions.x = 0.5
            person.dimensions.y = 0.5
            person.dimensions.z = 1.5
            persons.boxes.append(person)

        print(len(persons.boxes))
        self.pub_persons.publish(persons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--camera', type=int, default=1, help='camera id')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    node = RosOpenPoseLight(
        net=net,
        camera_num=args.camera
    )

    rospy.sleep(3)

    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        t1 = time.time()
        node.run_demo()
        r.sleep()
        # print(1/(time.time() - t1))

    # run_demo(net, RosReader(camera=4), args.height_size, args.cpu, args.track, args.smooth)
