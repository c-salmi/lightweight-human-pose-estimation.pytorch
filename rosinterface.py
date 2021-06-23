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


class RosReader:
    def __init__(self, camera=4):
        rospy.init_node("openpose_light")

        self.color_img = None
        self.depth_img = None
        self.K = None

        # ROS subscribers
        rospy.Subscriber(f'/camera{camera}/color/image_raw', Image, self.color_cb, queue_size=1)
        rospy.Subscriber(f'/camera{camera}/aligned_depth_to_color/image_raw', Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(f'/camera{camera}/color/camera_info', CameraInfo, self.info_cb, queue_size=1)

        # ROS publishers
        # self.pub_persons = rospy.Publisher('/openpose/tracked_persons', TrackedPersons, queue_size=10)
        self.pub_persons = rospy.Publisher(f'/camera{camera}/openpose/detections', BoundingBoxArray, queue_size=10)

        self.listener = tf.TransformListener()

    def color_cb(self, msg):
        self.color_msg = msg
        self.color_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.color_time = msg.header.stamp

    def depth_cb(self, msg):
        self.depth_msg = msg
        self.depth_img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        self.depth_time = msg.header.stamp

    def info_cb(self, msg):
        self.K = np.array(msg.K).reshape((3, 3))

    def __iter__(self):
        return self

    def __next__(self):
        return copy(self.color_img), copy(self.depth_img)

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

    # def project_to_image(self, msg):
    #     x_img, y_img = [], []
    #     depths = []
    #     self.depth_img = np.ones([480, 640])*3
    #     for point in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")):
    #         x = point[0]
    #         y = point[1]
    #         z = point[2]
    #         depths.append(z)
    #         px = x*self.K[0, 0] / z + self.K[0, 2]
    #         py = y*self.K[1, 1] / z + self.K[1, 2]
    #
    #         if px >= 0 and py >= 0 and py < 480 and px < 640:
    #             self.depth_img[int(py), int(px)] = z


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    t1 = time.time()
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    print(time.time() - t1)
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img, img_depth in image_provider:
        t1 = time.time()
        # print(f'depth time: {image_provider.depth_time.to_sec()}')
        # print(f'depth diff time: {rospy.get_time() - image_provider.depth_time.to_sec()}')
        # print(f'color time: {image_provider.color_time.to_sec()}')
        # print(f'color diff time: {rospy.get_time() - image_provider.color_time.to_sec()}')
        # print(f'diff time: {image_provider.depth_time.to_sec() - image_provider.color_time.to_sec()}')
        # print('\n')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        orig_img = img.copy()
        img_depth = img_depth/1000
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
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

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

        # TrackedPersons
        persons = BoundingBoxArray()
        persons.header.frame_id = 'map'
        for pose in current_poses:
            # Extract median depth value of keypoints
            keypoint_depths = [img_depth[pose_keypoint[1], pose_keypoint[0]] for pose_keypoint in pose.keypoints if not all(pose_keypoint == [-1, -1]) and not img_depth[pose_keypoint[1], pose_keypoint[0]] == 0]
            if len(keypoint_depths) == 0:
                continue
            avg_depth = np.median(keypoint_depths)
            point = pose.keypoints[0] if not all(pose.keypoints[0] == [-1, -1]) else pose.keypoints[1]

            # 3D conversion
            point3d = image_provider.calculate_depth(point, avg_depth)

            # Create pose
            pose3d = PoseStamped()
            pose3d.header.frame_id = '/camera4_link'
            pose3d.pose.position.x = point3d[0]
            pose3d.pose.position.y = point3d[1]
            pose3d.pose.position.z = point3d[2]

            # Covert to map frame
            pose3d_map = image_provider.listener.transformPose('map', pose3d)

            person = BoundingBox()
            person.header.frame_id = 'map'
            person.pose.position = pose3d_map.pose.position
            person.pose.position.z -= 0.5
            person.dimensions.x = 0.5
            person.dimensions.y = 0.5
            person.dimensions.z = 1.5

            persons.boxes.append(person)

            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

            cv2.rectangle(img_depth, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), 3)

            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        image_provider.pub_persons.publish(persons)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

        # print(1/(time.time() - t1))


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

    run_demo(net, RosReader(camera=4), args.height_size, args.cpu, args.track, args.smooth)
