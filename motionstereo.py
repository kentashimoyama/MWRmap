#!/usr/bin/python
#! -*- encoding: utf-8 -*-

import os
import subprocess
import sys
import cv2
print(cv2.getBuildInformation())
exit()

# Indicate the openMVG binary directory
OPENMVG_BIN_DIR = "~/workspace/reps/openmvg/install/bin"
OPENMVS_BIN_DIR = "~/workspace/reps/openmvs/install/bin/OpenMVS"
OPENCV_SO_DIR = "~/workspace/reps/opencv-3.1.0/install/lib"

def SetEnvVars():
  if 'LD_LIBRARY_PATH' not in os.environ:
    # LD_LIBRARY_PATH_ORG = os.environ.get('LD_LIBRARY_PATH', '')  # デフォルトで空文字を設定 #自分で追加した行
    # print(os.environ)
    # LD_LIBRARY_PATH_ORG = os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = OPENCV_SO_DIR
    try:
      os.execv(sys.argv[0], sys.argv)
    except Exception as exc:
      print(f"Failed re-exec:{exc}")
      sys.exit(1)

  print(f'Success:{os.environ['LD_LIBRARY_PATH']}')

class SfMPipeLine(object):

  def __init__(self, datadir, camera_file_path):
    self.__datadir = datadir
    self.__image_dir = os.path.join(self.__datadir, "images")
    self.__output_dir = os.path.join(self.__datadir, "result")
    self.__matches_dir = os.path.join(self.__output_dir, "matches")
    self.__reconstruction_dir = os.path.join(self.__output_dir,"reconstruction_global")
    self.__mvs_dir = os.path.join(self.__output_dir,"mvs_dir")
    self.__camera_file_params = camera_file_path
    self.__create_folder_structure(self.__output_dir)

  def start_sfm_pipeline(self):
    self.__intrinsics_analysis(self.__camera_file_params)
    self.__compute_features()
    self.__compute_matches()
    self.__reconstruct_incremental()
    self.__colorize_structure()
    self.__triangulate_robust()
    # self.__convert_openmvg2openmvs()
    # self.__densify_pointcloud()
    # self.__reconstruct_mesh()
    # self.__refine_mesh()
    # self.__texture_mesh()


  def __create_folder_structure(self, output_dir):
    print("Create Folder Structure")
    
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)  

    if not os.path.exists(self.__matches_dir):
      os.mkdir(self.__matches_dir)

    if not os.path.exists(self.__reconstruction_dir):
      os.mkdir(self.__reconstruction_dir)

    if not os.path.exists(self.__mvs_dir):
      os.mkdir(self.__mvs_dir)


  def __intrinsics_analysis(self, camera_file_params):
    print("Intrinsics Analysis")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_SfMInit_ImageListing")
    pIntrisics = \
            subprocess.Popen( [exepath, "-i", self.__image_dir, "-o", self.__matches_dir, \
                              "-d", camera_file_params, "-c", "3"] )
    pIntrisics.wait()

  def __compute_features(self):
    print("Compute Features")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_ComputeFeatures")
    jsonpath = self.__matches_dir+"/sfm_data.json"
    pFeatures = \
            subprocess.Popen( [exepath, "-i", jsonpath, "-o", self.__matches_dir, \
                              "-m", "SIFT", "-f" , "1"] )
    pFeatures.wait()


  def __compute_matches(self):
    print("Compute Matches for Incremental SfM PipeLine")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_ComputeMatches")
    jsonpath = self.__matches_dir+"/sfm_data.json"
    pMatches = \
            subprocess.Popen( [exepath, "-i", jsonpath, "-o", self.__matches_dir, \
                               "-f", "1", "-n", "ANNL2"] )
    pMatches.wait()

  def __reconstruct_incremental(self):
    print("Incremental Reconstruction")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_IncrementalSfM")
    jsonpath = self.__matches_dir+"/sfm_data.json"
    pRecons = subprocess.Popen( [exepath, "-i", jsonpath, "-m", self.__matches_dir, \
                                "-o", self.__reconstruction_dir] )
    pRecons.wait()


  def __colorize_structure(self):
    print("Colorize Structure")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_ComputeSfM_DataColor")
    plypath = os.path.join(self.__reconstruction_dir,"colorized.ply")
    binpath = self.__reconstruction_dir+"/sfm_data.bin"
    pRecons = subprocess.Popen( [exepath, "-i", binpath, "-o", plypath] )
    pRecons.wait()

  def __triangulate_robust(self):
    print("Robust Triangulation")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_ComputeStructureFromKnownPoses")
    plypath = os.path.join(self.__reconstruction_dir,"robust.ply")
    binpath = self.__reconstruction_dir+"/sfm_data.bin"
    pRecons = \
          subprocess.Popen( [exepath, "-i", binpath, "-m", self.__matches_dir, "-o", plypath] )
    pRecons.wait()

  def __convert_openmvg2openmvs(self):
    print("OpenMVG 2 OpenMVS")
    exepath = os.path.join(OPENMVG_BIN_DIR, "openMVG_main_openMVG2openMVS")
    binpath = self.__reconstruction_dir+"/sfm_data.bin"
    mvspath = self.__mvs_dir+"/scene.mvs"
    pRecons = \
          subprocess.Popen( [exepath, "-i", binpath, "-o", mvspath, "-d", self.__mvs_dir] )
    pRecons.wait()

  def __densify_pointcloud(self):
    print("Densify Point Cloud")
    exepath = os.path.join(OPENMVS_BIN_DIR, "DensifyPointCloud")
    mvspath = self.__mvs_dir + "/scene.mvs"
    pRecons = \
          subprocess.Popen( [exepath, mvspath, "-w", self.__mvs_dir] )
    pRecons.wait()


  def __reconstruct_mesh(self):
    print("Reconstruct Mesh")
    exepath = os.path.join(OPENMVS_BIN_DIR, "ReconstructMesh")
    mvspath = self.__mvs_dir + "/scene_dense.mvs"
    pRecons = \
          subprocess.Popen( [exepath, mvspath, "-w", self.__mvs_dir] )
    pRecons.wait()

  def __refine_mesh(self):
    print("Refine Mesh")
    exepath = os.path.join(OPENMVS_BIN_DIR, "RefineMesh")
    mvspath = self.__mvs_dir + "/scene_dense_mesh.mvs"
    pRefin = \
          subprocess.Popen( [exepath, \
                            mvspath, "-w", self.__mvs_dir] )
    pRefin.wait()

  def __texture_mesh(self):
    print("Texture Mesh")
    exepath = os.path.join(OPENMVS_BIN_DIR, "TextureMesh")
    mvspath = self.__mvs_dir + "/scene_dense_mesh_refine.mvs"
    pRecons = \
          subprocess.Popen( [exepath, \
                            mvspath, "-w", self.__mvs_dir] )
    pRecons.wait()

if __name__=='__main__':

  SetEnvVars()

  camera_file_params = \
        "~/workspace/reps/openmvg/openMVG/src/openMVG/exif/" + \
        "sensor_width_database/sensor_width_camera_database.txt"
  data_dir = "~/Desktop/SFM/SFM_Sample/ImageDataset_SceauxCastle"
  #data_dir = "~/Desktop/SFM/SFM_Sample/Sendai_Station"
  #data_dir = "~/Desktop/SFM/SFM_Sample/Mansion"
  #data_dir = "~/Desktop/SFM/SFM_Sample/ET"

  print("SfM Pipeline Started")
  pl = SfMPipeLine(data_dir, camera_file_params)
  pl.start_sfm_pipeline()