#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    spherical_cameras: list = None
    dist2_path: str = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            cx = intr.params[2]
            cy = intr.params[3]
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if width != image.width:
            scale_w = image.width / width
            scale_h = image.height / height
            cx = cx * scale_w
            cy = cy * scale_h

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, points_pcl_suffix: str = ""):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        ply_path = os.path.join(path, f"sparse/0/points3D{points_pcl_suffix}.ply")
        dist2_path = os.path.join(path, f"sparse/0/points3D{points_pcl_suffix}_dist2.pt")
        pcd = fetchPly(ply_path)
    except:
        print("Could not find a pointcloud at", ply_path)
        pcd = None
        dist2_path = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           dist2_path=dist2_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", overwrite_spherical_cams=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]

        if overwrite_spherical_cams:
            n = len(frames)
            radius = 4.0
            thetas = np.array([60] * n) / 180.0 * np.pi
            phis = np.linspace(start=0, stop=360, num=n) / 180.0 * np.pi

            centers = np.stack([
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.cos(thetas),
            ], axis=-1)

            targets = 0

            # lookat
            forward_vector = centers - targets
            forward_vector = forward_vector / np.linalg.norm(forward_vector, axis=-1)[..., None]

            up_vector = np.array([[0, 0, 1]])
            up_vector = np.repeat(up_vector, repeats=n, axis=0)

            right_vector = np.cross(forward_vector, up_vector)
            right_vector = right_vector / np.linalg.norm(right_vector, axis=-1)[..., None]

            up_vector = np.cross(right_vector, forward_vector)
            up_vector = up_vector / np.linalg.norm(up_vector, axis=-1)[..., None]

            poses = np.eye(4)[None]
            poses = np.repeat(poses, repeats=n, axis=0)
            poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
            poses[:, :3, 3] = centers
            poses[:, :3, 1:3] *= -1
            poses = np.linalg.inv(poses)

            # for frame in frames:
            #     # NeRF 'transform_matrix' is a camera-to-world transform
            #     c2w = np.array(frame["transform_matrix"])
            #     # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            #     c2w[:3, 1:3] *= -1
            #
            #     # get the world-to-camera transform and set R, T
            #     w2c = np.linalg.inv(c2w)
            #     R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            #     T = w2c[:3, 3]
            #
            #     poses = np.concatenate([poses, w2c[None]], axis=0)

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            if not overwrite_spherical_cams:
                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
            else:
                w2c = poses[idx]
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]

                # fx = fov2focal(fovx, 32)
                # K = np.array([[[fx, 0, 16], [0, fx, 16], [0, 0, 1]]])
                # K = np.repeat(K, repeats=len(poses), axis=0)
                #
                # def vis_cams(cams, intrs, h, w):
                #     import open3d as o3d
                #     viz = o3d.visualization.Visualizer()
                #     viz.create_window()
                #     viz.create_window(width=512, height=512)
                #
                #     # add coordinate axes
                #     meshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                #     viz.add_geometry(meshFrame)
                #
                #     # add cams
                #     for cam, K in zip(cams, intrs):
                #         cameraLines = o3d.geometry.LineSet.create_camera_visualization(
                #             view_width_px=w, view_height_px=h, intrinsic=K[:3, :3], extrinsic=cam, scale=0.3
                #         )
                #         viz.add_geometry(cameraLines)
                #
                #     # add object bbox -> [-1, 1]^3 cube
                #     lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                #         o3d.geometry.AxisAlignedBoundingBox(
                #             min_bound=np.array([-1, -1, -1]),
                #             max_bound=np.array([1, 1, 1]),
                #         )
                #     )
                #     lineset_colors = np.array(lineset.colors)
                #     lineset_colors[:, 0] = 1.0  # red
                #     lineset_colors[:, 1] = 0.0  # red
                #     lineset_colors[:, 2] = 0.0  # red
                #     lineset.colors = o3d.utility.Vector3dVector(lineset_colors)
                #     viz.add_geometry(lineset)
                #
                #     viz.add_geometry(o3d.io.read_point_cloud("output/32949332-3/point_cloud/iteration_500/point_cloud.ply"))
                #
                #     viz.run()

                # vis_cams(poses, K, 32, 32)

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=image.width/2, cy=image.height/2, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", points_pcl_suffix: str = ""):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    print("Creating Spherical Transforms")
    spherical_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, overwrite_spherical_cams=True)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"points3d{points_pcl_suffix}.ply")
    dist2_path = os.path.join(path, f"points3D{points_pcl_suffix}_dist2.pt")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        # num_pts = 5000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           spherical_cameras=spherical_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           dist2_path=dist2_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}