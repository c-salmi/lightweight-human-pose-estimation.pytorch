import argparse
import time
from copy import copy

import cv2
import numpy as np
import torch
from torch2trt import torch2trt
import rospy
from ros_numpy import msgify
import tf
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker


from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class RosOpenPoseLight:
    def __init__(self, net, camera_ids, net_input_height_size=256, stride=8, upsample_ratio=4):
        rospy.init_node("openpose_light")

        # options
        self.cv2viz = True
        self.trt = False

        self.net = net.eval().cuda()

        if self.trt:
            # create example data
            x = torch.ones((5, 3, 256, 344)).cuda()

            # convert to TensorRT feeding sample data as input
            self.net = torch2trt(self.net, [x])

        # model post/pre processing parameters
        self.net_input_height_size = net_input_height_size
        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.num_keypoints = Pose.num_kpts

        # store camera data
        self.camera_ids = camera_ids
        self.color_imgs = [None]*len(camera_ids)
        self.depth_imgs = [None]*len(camera_ids)
        self.color_msgs = [None]*len(camera_ids)
        self.depth_msgs = [None]*len(camera_ids)
        self.Ks = [None]*len(camera_ids)

        # ROS subscribers
        for i, c_id in enumerate(camera_ids):
            # rospy.Subscriber(f'/camera{c_id}/color/image_raw_relay', Image, self.color_cb, ({'camera_num':i}), queue_size=1)
            rospy.Subscriber(f'/camera{c_id}/color/image_raw_relay/compressed', CompressedImage, self.color_compressed_cb, ({'camera_num':i}), queue_size=1)
            rospy.Subscriber(f'/camera{c_id}/aligned_depth_to_color/image_raw_relay', Image, self.depth_cb, ({'camera_num':i}), queue_size=1)
            rospy.Subscriber(f'/camera{c_id}/color/camera_info', CameraInfo, self.info_cb, ({'camera_num':i}), queue_size=1)

        # ROS publishers
        self.pub_persons = {}
        self.pub_camera_detections = {}
        for i, c_id in enumerate(camera_ids):
            self.pub_persons[c_id] = rospy.Publisher(f'/camera{c_id}/openpose/detections', BoundingBoxArray, queue_size=10)
            self.pub_camera_detections[c_id] = rospy.Publisher(f'/camera{c_id}/openpose/camera_detections', Image, queue_size=10)

        # self.pub_persons = rospy.Publisher('/openpose/tracked_persons', TrackedPersons, queue_size=10)
        self.pub_persons_all = rospy.Publisher('/openpose/detections', BoundingBoxArray, queue_size=10)
        self.pub_poses_viz = rospy.Publisher(f'/openpose/markers', MarkerArray, queue_size=10)

        self.listener = tf.TransformListener()

    def color_cb(self, msg, args):
        # self.color_time = msg.header.stamp
        self.color_msgs[args['camera_num']] = msg
        self.color_imgs[args['camera_num']] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        
    def color_compressed_cb(self, msg, args):
        self.color_msgs[args['camera_num']] = msg
        np_arr = np.fromstring(msg.data, np.uint8)
        self.color_imgs[args['camera_num']] = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def depth_cb(self, msg, args):
        # self.depth_time = msg.header.stamp
        self.depth_msgs[args['camera_num']] = msg
        self.depth_imgs[args['camera_num']] = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)

    def info_cb(self, msg, args):
        self.Ks[args['camera_num']] = np.array(msg.K).reshape((3, 3))

    def inference(self):
        color_imgs = np.array(copy(self.color_imgs))
        depth_imgs = np.array(copy(self.depth_imgs))

        if self.cv2viz:
            orig_imgs = color_imgs.copy()

        heatmaps, pafs, scale, pad = self.infer_fast(color_imgs)

        current_poses = []
        for i, (heatmap, paf, color_img, depth_img) in enumerate(zip(heatmaps, pafs, color_imgs, depth_imgs)):
            depth_img = depth_img/1000
            single_camera_poses = self.extract_poses_single_camera(i, heatmap, paf, scale, pad, depth_img)
            if self.cv2viz:
                for pose in single_camera_poses:
                    pose.draw(color_img)
                color_img = cv2.addWeighted(orig_imgs[i], 0.6, color_img, 0.4, 0)
                for pose in single_camera_poses:
                    cv2.rectangle(color_img, (pose.bbox[0], pose.bbox[1]),
                                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

                msg = msgify(Image, color_img, encoding='bgr8')
                self.pub_camera_detections[self.camera_ids[i]].publish(msg)

            current_poses += single_camera_poses

        self.viz_poses(current_poses)

    def infer_fast(self, imgs, pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        batch_size, height, width, _ = imgs.shape
        scale = self.net_input_height_size / height

        padded_imgs = []
        pad = None

        for i, img in enumerate(imgs):
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled_img = normalize(scaled_img, img_mean, img_scale)
            min_dims = [self.net_input_height_size, max(scaled_img.shape[1], self.net_input_height_size)]
            padded_img, pad = pad_width(scaled_img, self.stride, pad_value, min_dims)
            padded_imgs.append(np.transpose(padded_img, (2, 0, 1)))
        padded_imgs = np.array(padded_imgs)

        tensor_imgs = torch.from_numpy(padded_imgs).float().cuda()

        # t1 = time.perf_counter()
        # print(tensor_imgs.shape)
        stages_output = net(tensor_imgs)
        # torch.cuda.synchronize()
        # print(time.perf_counter() - t1)

        stage2_heatmaps = stages_output[-2].cpu().data.numpy()
        torch.cuda.synchronize()

        heatmaps = []
        for stage2_heatmap in stage2_heatmaps:
            heatmap = np.transpose(stage2_heatmap, (1, 2, 0))
            heatmaps.append(cv2.resize(heatmap, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC))
        heatmaps = np.array(heatmaps)

        stage2_pafs = stages_output[-1]

        pafs = []
        for stage2_paf in stage2_pafs:
            paf = np.transpose(stage2_paf.cpu().data.numpy(), (1, 2, 0))
            pafs.append(cv2.resize(paf, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC))
        pafs = np.array(pafs)

        return heatmaps, pafs, scale, pad

    def extract_poses_single_camera(self, camera_num, heatmap, paf, scale, pad, depth_img):
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmap[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, paf, min_paf_score=0.15)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        current_poses = []
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose[kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose[kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose[kpt_id]), 1])
            current_poses.append(Pose(pose_keypoints, pose[18]))

        # extract depth from every pose
        for pose in current_poses:
            pose.skip = False

            # TODO: clip keypoints to within frame
            kps = [np.clip(pose_keypoint, 0, 639) for pose_keypoint in pose.keypoints]

            # Extract all depth values and add to pose
            pose.keypoints3d = []
            pose.keypoints3dMap = []
            pose.keypoint_depths = []
            for kp in pose.keypoints:
                if all(kp == [-1, -1]):
                    pose.keypoints3d.append(None)
                    pose.keypoints3dMap.append(None)
                    pose.keypoint_depths.append(None)
                else:
                    d = depth_img[kp[1], kp[0]]
                    pose.keypoint_depths.append(d)
                    point = self.calculate_depth(kp, d, camera_num)
                    pose.keypoints3d.append(point)

                    p_temp = PoseStamped()
                    p_temp.header.frame_id = f'/camera{self.camera_ids[camera_num]}_link'
                    p_temp.pose.position.x = point[0]
                    p_temp.pose.position.y = point[1]
                    p_temp.pose.position.z = point[2]
                    pose.keypoints3dMap.append(self.listener.transformPose('map', p_temp))

            # Extract center based on ranked list of important keypoints, where depth is the median
            median_depth = np.median([d for d in pose.keypoint_depths if d is not None and d != 0])

            if pose.keypoints3d[0] is not None:  # Keypoint head
                point = self.calculate_depth(pose.keypoints[0], median_depth, camera_num)
                p_temp = PoseStamped()
                p_temp.header.frame_id = f'/camera{self.camera_ids[camera_num]}_link'
                p_temp.pose.position.x = point[0]
                p_temp.pose.position.y = point[1]
                p_temp.pose.position.z = point[2]
                pose.pose3dMap = self.listener.transformPose('map', p_temp)
            elif pose.keypoints3d[1] is not None:  # Keypoint neck
                point = self.calculate_depth(pose.keypoints[1], median_depth, camera_num)
                p_temp = PoseStamped()
                p_temp.header.frame_id = f'/camera{self.camera_ids[camera_num]}_link'
                p_temp.pose.position.x = point[0]
                p_temp.pose.position.y = point[1]
                p_temp.pose.position.z = point[2]
                pose.pose3dMap = self.listener.transformPose('map', p_temp)
            else:
                pose.skip = True

            # keypoint_depths = [depth_img[kp[1], kp[0]] for kp in kps if not all(kp == [-1, -1]) and not depth_img[kp[1], kp[0]] == 0]

            ############
            # keypoint_vals = [kp for kp in kps if not all(kp == [-1, -1]) and not depth_img[kp[1], kp[0]] == 0]
            # pose.keypoints3d = [self.calculate_depth(kp, d, camera_num) for kp, d in zip(keypoint_vals, keypoint_depths)]
            # pose.keypoints3dMap = []
            # for kp in pose.keypoints3d:
            #     p_temp = PoseStamped()
            #     p_temp.header.frame_id = f'/camera{self.camera_ids[camera_num]}_link'
            #     p_temp.pose.position.x = kp[0]
            #     p_temp.pose.position.y = kp[1]
            #     p_temp.pose.position.z = kp[2]
            #     pose.keypoints3dMap.append(self.listener.transformPose('map', p_temp))
            ############

            # if len(keypoint_depths) == 0:
            #     pose.skip = True
            #     continue
            #
            # avg_depth = np.median(keypoint_depths)
            # if all(pose.keypoints[0] == [-1, -1]) and all(pose.keypoints[1] == [-1, -1]):
            #     pose.skip = True
            #     continue
            # point = pose.keypoints[0] if not all(pose.keypoints[0] == [-1, -1]) else pose.keypoints[1]
            #
            # point3d_camera = self.calculate_depth(point, avg_depth, camera_num)
            # pose.point3d_camera = point3d_camera
            #
            # # Create pose
            # pose3d = PoseStamped()
            # pose3d.header.frame_id = f'/camera{self.camera_ids[camera_num]}_link'
            # pose3d.pose.position.x = point3d_camera[0]
            # pose3d.pose.position.y = point3d_camera[1]
            # pose3d.pose.position.z = point3d_camera[2]
            # pose.pose3d_camera = pose3d
            # pose.pose3d_map = self.listener.transformPose('map', pose3d)

        return [pose for pose in current_poses if not pose.skip]

    def calculate_depth(self, keypoint, z, camera_num):
        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        fx = self.Ks[camera_num][0, 0]
        fy = self.Ks[camera_num][1, 1]
        cx = self.Ks[camera_num][0, 2]
        cy = self.Ks[camera_num][1, 2]

        x = z / fx * (keypoint[0] - cx)
        y = z / fy * (keypoint[1] - cy)

        return [z, -x, -y]

    def viz_poses(self, poses):
        persons = BoundingBoxArray()
        persons.header.frame_id = 'map'
        marker_array = MarkerArray()
        m_id = 0
        for p in poses:
            person = BoundingBox()
            person.header.frame_id = 'map'
            person.pose.position = p.pose3dMap.pose.position
            person.pose.position.z -= 0.5
            person.dimensions.x = 0.5
            person.dimensions.y = 0.5
            person.dimensions.z = 1.5
            persons.boxes.append(person)

            m = Marker()
            m.id = m_id
            m_id += 1
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.a = 0.8
            m.color.r = 0
            m.color.g = 0
            m.color.b = 255
            m.action = Marker.ADD
            m.type = Marker.LINE_STRIP
            m.header.frame_id = 'map'
            for kp in p.keypoints3dMap:
                if kp is None: continue
                # m.pose.position = kp.pose.position
                m.points.append(kp.pose.position)
                # m.lifetime = 0.1
                marker_array.markers.append(m)

        # print(f"Found {len(persons.boxes)} persons")
        self.pub_persons_all.publish(persons)
        self.pub_poses_viz.publish(marker_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python, extended version that uses depth image to project 
                       keypoints to 3d space''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path)#, map_location='cpu')
    load_state(net, checkpoint)

    node = RosOpenPoseLight(
        net=net,
        camera_ids=[1, 2, 3, 4, 5],
        net_input_height_size=args.height_size
    )

    rospy.sleep(1)

    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        t1 = time.time()
        node.inference()
        print(time.time() - t1)
        r.sleep()

    # run_demo(net, RosReader(camera=4), args.height_size, args.cpu, args.track, args.smooth)
