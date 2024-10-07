import zmq
import cv2
import struct
import numpy as np
import time
from datetime import datetime

# Connection String
TIME_FORMAT = "%y%m%d%H%M%S%f"
class MMSProbeFrame:
    def __init__(self):
        self.image = None
        self.gps_timestamp = None
        self.gps_latitude = None
        self.gps_altitude = None
        self.image_address = None
        self.frame_number = None
        self.mwr_symbol = None  # 暫定はdictでやる
        self.flag = 0  # 1 means last frame


class DataSocket:
    def __init__(self, conn_str="tcp://localhost:4999", channel=''):
        self.conn_str = conn_str
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.channel = channel
        channel_name = self.channel.encode('ascii')
        self.sock.setsockopt(zmq.SUBSCRIBE, channel_name)
        self.sock.connect(self.conn_str)


    def get_frame(self):
        # Receive Data from C++ Program
        # byte_rows, byte_cols, byte_mat_type, byte_timestamp, byte_latitude, byte_longitude, byte_altitude, \
        #                                                         byte_frame_number, data = self.sock.recv_multipart()
        # ########### from python############
        byte_rows, byte_cols, byte_mat_type, byte_timestamp, byte_latitude, byte_longitude, byte_altitude, \
                            byte_frame_number, byte_time, byte_flag, send_flag = self.sock.recv_multipart()

        MWR_flag = struct.unpack('i', send_flag)[0]
        if MWR_flag:
            MWR_dict = self.sock.recv_pyobj()
        image = self.sock.recv_pyobj()
        ##################################

        # Convert byte to integer
        rows = struct.unpack('i', byte_rows)[0]
        # print(rows)
        cols = struct.unpack('i', byte_cols)[0]
        mat_type = struct.unpack('i', byte_mat_type)[0]

        # if mat_type[0] == 0:
        # #     # Gray Scale
        #     image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0], cols[0]))
        # else:
        #     # BGR Color
        #     image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0], cols[0], 3))

        image = image.reshape(720, 1280, 3)
        frame = MMSProbeFrame()
        frame.image = image
        frame.gps_timestamp = struct.unpack('Q', byte_timestamp)[0]
        frame.gps_latitude = struct.unpack('d', byte_latitude)[0]
        frame.gps_longitude = struct.unpack('d', byte_longitude)[0]
        frame.gps_altitude = struct.unpack('d', byte_altitude)[0]
        frame.frame_number = struct.unpack('Q', byte_frame_number)[0]
        if MWR_flag:
            frame.mwr_symbol = MWR_dict

        return frame

        # t = struct.unpack('d', byte_time)[0]
        # print('time used: ', time.time() - t)
        # for batch process
        # frame.flag = struct.unpack('i', byte_flag)[0]




if __name__ == '__main__':
    print('Start')
    old = 0
    DS = DataSocket()
    Flag = 0
    while Flag == 0: # Flag?
        frame = DS.get_frame()
        print('timestamp:', datetime.fromtimestamp(frame.gps_timestamp/1000).strftime(TIME_FORMAT))
        print('Frame Number: ', frame.frame_number)
        print('Millimeter wave symbol: ', frame.mwr_symbol)
        cv2.imshow('MMS-Probe', frame.image)
        cv2.waitKey(1)
        FPS = 1/((time.time() - old))
        print('FPS: ', FPS)
        old = time.time()
        Flag = frame.flag