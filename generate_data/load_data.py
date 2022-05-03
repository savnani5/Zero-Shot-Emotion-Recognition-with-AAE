import numpy as np
import re
import glob
from transforms3d.euler import euler2mat, mat2euler
from computeFeatures import *
import deepdish as dd
import h5py
import matplotlib.pyplot as plt


class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []

        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:10, 0], pos[:10, 2], pos[:10, 1],c='r')
        ax.scatter(pos[10:, 0], pos[10:, 2], pos[10:, 1],c='b')
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(-150, 150)
        plt.show()

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue

            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

        return index_offset

    def frame_pose(self, frame):
        # Extract position and rotation of every joint
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))

        return p, r

    def all_frame_poses(self):
        # Poses for all frames
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def _plot_pose(self, p, r, fig=None, ax=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(-150, 150)

        plt.draw()
        plt.pause(0.001)

    def plot_frame(self, frame, fig=None, ax=None):
        p, r = self.frame_pose(frame)
        self._plot_pose(p, r, fig, ax)

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def plot_all_frames(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(self.frames)
        for i in range(self.frames):
            print(i)
            self.plot_frame(i, fig, ax)

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"

def getFiles(filename):
    images = sorted(glob.glob(filename))
    return images

def getFeatureMatrix(frames):
    # Reducing the n joint space to 10 joints
    feature = np.zeros((frames.shape[0],10,frames.shape[2]))

    feature[:,0,:] = frames[:,0,:]
    feature[:,1,:] = frames[:,3,:]
    feature[:,2,:] = frames[:,5,:]
    feature[:,3,:] = frames[:,7,:]
    feature[:,4,:] = frames[:,9,:]
    feature[:,5,:] = frames[:,10,:]
    feature[:,6,:] = frames[:,12,:]
    feature[:,7,:] = frames[:,14,:]
    feature[:,8,:] = frames[:,15,:]
    feature[:,9,:] = frames[:,17,:]

    return feature



if __name__ == '__main__':

    filename = '../data/bvh/*.bvh'
    fname = getFiles(filename)
    f = fname[0]
    hf = h5py.File('../feature_data/featuresABHI.h5', 'w')
    max_tstep = 0
    vals = []
    for idx,f in enumerate(fname):
        if (idx==446) or (idx==666) or (idx==742) or (idx==749):
            continue
        anim = Bvh()
        posit = anim.parse_file(f)
        p, r = anim.frame_pose(0)
        name = anim.joint_names()
        timestep = 1 / float(anim.fps)
        all_p, all_r = anim.all_frame_poses()
        temp = getFeatureMatrix(all_p)
        temp = temp.reshape(temp.shape[0],temp.shape[1]*temp.shape[2])
        temp = np.asarray(temp)
        if temp.shape[0]<250:
            # print(idx,"Gayab")
            continue
        elif temp.shape[0]<500:
            temp_array = temp.copy()
            temp_array.resize(500,temp.shape[1])
            temp = temp_array[::5,:]
        elif temp.shape[0]<1000:
            temp = temp[::2,:]
            temp_array = temp.copy()
            temp_array.resize(500,temp.shape[1])
            temp = temp_array[::5,:]
        elif temp.shape[0]<1500:
            temp = temp[::3,:]
            temp_array = temp.copy()
            temp_array.resize(500,temp.shape[1])
            temp = temp_array[::5,:]
        else:
            temp = temp[::4,:]
            temp_array = temp.copy()
            temp_array.resize(500,temp.shape[1])
            temp = temp_array[::5,:]
        # vals.append(temp.shape[0])
        # temp = np.resize(temp, (2023,temp.shape[1]))
        # if temp.shape[0]>max_tstep:
        #     max_tstep = temp.shape[0]
        print(idx)
        # print(temp.shape[0])
        # features[(idx+1)] = temp
        hf.create_dataset('dataset_'+str(idx).zfill(5), data=temp)
    # vals = np.asarray(vals)
    # plt.hist(vals, bins=20)
    # plt.ylabel('No of times')
    # plt.show()
    ############################################################

    # print(list(features.keys()))
    # hf = h5py.File('data.h5', 'w')
    #############################################################
    label_dict = {'joy': 0,
                'relief': 1,
                'pride': 2,
                'shame': 3,
                'anger': 4,
                'surprise': 5,
                'amusement': 6, 
                'sadness': 7,
                'fear': 8,
                'neutral': 9,
                'disgust':10
                }
    hl = h5py.File('../feature_data/labelsABHI.h5', 'w')
    labelname = '../data/tags/*.txt'
    lname = getFiles(labelname)
    for idl,l in enumerate(lname):
        if (idl==446) or (idl==666) or (idl==742) or (idl==749) or (idl==757):
            continue
        file = open(l,'r').readlines()
        lab = file[1].strip()
        lab = label_dict[lab]
        hl.create_dataset('dataset_'+str(idl).zfill(5), data=lab)
    #############################################################