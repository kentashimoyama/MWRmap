#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import threading
import time
import datetime
import sys
import math
import struct
import numpy as np

import csv
import pprint
import array

import termios
import tty
import pandas as pd

#import io

class MWRdata():
    def __init__(self):
        self.MWR_dict = dict() #{"17089...(unixtime)":ID UnixTime M_Xrange[m] M_Yrange[m]
                               #    1                  1   17089...  -0.78...   1.184...
                               #    2                  1   17089...  -0.79...   1.186... } という辞書のなかにdataframeが格納されている

# unixtime用JST定義
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

# フレーム時間係数
numUpDnFD_Val = 75 #[ms]

# 変換係数
invQformat = 1.0 / (1 << 7)
    
# ファイル名用UNIXTIME
filename_Unixtime = datetime.datetime.now(JST).strftime('%Y%m%d') + '_' + str(round(time.time(),0))[:-2]

# フレームID（受信数を保持する必要があるためグローバル化
Frameid = 0

# フレーム取得数(Time測定用)
fst_frame_num = -1

# csv書き込み用ログデータ用リスト初期化
log_data = list()
log_data2 = list()

# D1データ受信フラグ
recv_data_sta = 0

# D1データ部残量
recv_data_cnt = 0

# 受信応答フラグ(0:受信待ち、1:受信確認)
recv_flg = 0

# データカウント(Symbolid用変数)
data_cnt = 0

def serial_open(v_port, v_baudrate, v_timeout):
    global ser #グローバルオブジェクト定義
    
    ser = serial.Serial(port = v_port, baudrate = v_baudrate, timeout = v_timeout)
    
def serial_close():
    global ser #グローバルオブジェクト定義
    ser.close()
        
def serial_recieve():
    # 受信割り込み処理
    global ser #グローバルオブジェクト定義
    global recv_flg

    try:
        recieved_data = ""
        while True:
            if ser.in_waiting > 0:
                recieved_data += ser.read(ser.in_waiting).decode('utf-8', errors='replace')
                while '\r' in recieved_data:
                    if recieved_data == '\r':
                        recieved_data = ""
                        recv_flg = 1
                        pass
                    elif recieved_data == 'z':
                        recieved_data = ""
                    elif recieved_data == b'x07':
                        recieved_data = ""
                        print('err_bell')
                    else:
                        line, recieved_data = recieved_data.split('\r', 1)
                        res_data = on_recieve(line)

                        _mwrdata = MWRdata()
                        MWR_time_curr = res_data[2]
                        if MWR_time_pre == False:
                            df_milliwave = pd.DataFrame(data=[res_data], columns=["ID", "UnixTime", "M_Xrange[m]", "M_Yrange[m]"])
                        elif MWR_time_curr == MWR_time_pre:
                            df_milliwave.loc = [res_data] #データフレームに行追加
                        else:
                            _mwrdata.MWR_dict[MWR_time_curr] = df_milliwave
                            df_milliwave =  pd.DataFrame(data=[res_data], columns=["ID", "UnixTime", "M_Xrange[m]", "M_Yrange[m]"])
                        
                        MWR_time_pre = MWR_time_curr

                        recv_flg = 1
    except:
        pass


def on_recieve(data):
    global Frameid
    global fst_frame_num
    global recv_data_sta
    global recv_data_cnt
    global data_c
    global f_recv_date

    res_data = list()

    if data != '' :
        if data[0] == 't':
            idcode = str(data[2]) + str(data[3])
            recv_frame = str(data[7]) + str(data[8]) + str(data[5]) + str(data[6]) + str(data[11]) + str(data[12]) + str(data[9]) + str(data[10])
            if idcode == 'D1':


                #0x0D1コード受信動作
                if recv_data_sta == 0: #データ部受信完了、ヘッダ部受信待ち
                    if recv_frame == '12345678':
                        #.dat出力処理
                        Frameid = struct.unpack('>H', bytes.fromhex(str(data[15]) + str(data[16]) + str(data[13]) + str(data[14])))[0]
                        data_write(data)
                        f_recv_date = datetime.datetime.now(JST)
                        if fst_frame_num < 0:
                            fst_frame_num = Frameid

                        recv_dsata_cnt = int(str(data[19]) + str(data[20]),16)
                        if recv_data_cnt > 0:
                            recv_data_sta = 1
                            data_c = 1
                elif recv_data_sta == 1: #データ部受信待ち
                    if recv_data_cnt > 0:
                        #.dat出力処理
                        data_write(data)
                        #res_dataをcsvに出力
                        res_data = D1logcsvWrite(Frameid,data_c,f_recv_date,data)
                        recv_data_cnt -= 1
                        data_c += 1
                        if recv_data_cnt == 0:
                            recv_data_sta = 0
                        return res_data
                    
            elif idcode == 'C1':
                #0x0C1コード受信動作
                print('command! 0xC1')
                print(data)
                pass
            else:
                pass
        else:
            #該当データではないため捨てる
            print(data)
            pass


def recvwait():
    global recv_flg
    
    while recv_flg == 0:
        pass
    
    recv_flg = 0

    
def data_write(add_data):
    global filename_Unixtime
    
    #datファイル定義
    fw = open(filename_Unixtime + '.dat','ab')
    writebyte = bytes.fromhex(add_data[5:])
    fw.write(writebyte)
    fw.close()

def D1logcsvWrite(FrameidN, Symbolid, DTime, data):
    global log_data
    global log_data2
    global fst_frame_num
    
    res_data = list()
    
    if fst_frame_num < 0:
        fst_frame_num = FrameidN
        
    fnum = FrameidN - fst_frame_num + 1
    
    # CANデータから生値取得
    Xr_data = str(data[7]) + str(data[8]) + str(data[5]) + str(data[6]) # Xレンジ
    Yr_data = str(data[11]) + str(data[12]) + str(data[9]) + str(data[10]) # Yレンジ
    P_data = str(data[19]) + str(data[20]) + str(data[17]) + str(data[18]) # 振幅
    V_data = str(data[15]) + str(data[16]) + str(data[13]) + str(data[14]) # ドップラー速度
    
    # XレンジをFloat変換
    fx_raw = struct.unpack('>h', bytes.fromhex(Xr_data))[0]
    fx = float(fx_raw * invQformat)

    # YレンジをFloat変換
    fy_raw = struct.unpack('>H', bytes.fromhex(Yr_data))[0]
    fy = float(fy_raw * invQformat)

    # ドップラー速度をFloat変換
    dopp_raw = struct.unpack('>h', bytes.fromhex(V_data))[0]
    doppler = float(dopp_raw * invQformat)

    # 振幅をFloat変換
    peak_raw = struct.unpack('>H', bytes.fromhex(P_data))[0]
    peak = float(peak_raw * 0.1)
    
    # 距離の算出
    tmp = (fx * fy) + (fx * fy)
    range_ans = math.sqrt(abs(tmp))

    # .CSV出力行に追加するデータ作成
    Nrep = FrameidN + 1
    Time = float((fnum*(numUpDnFD_Val / 1000.0)))
    NoOnNrep = 1
    Range = "{:.6E}".format(range_ans)
    Velocity = "{:.6E}".format(doppler)
    Azimuth = "{:.6E}".format(math.atan2(fx,fy) * 180 / math.pi)
    Elevation = "{:.6E}".format(0.0)
    Amplitudel10 = "{:.6E}".format(10 * math.log10(peak))
    Amplitude = "{:.6E}".format(peak)
    DataNo = FrameidN + 1
    Unixtime = "{:.3f}".format(DTime.timestamp())
    Date = DTime.strftime('%Y-%m-%d')
    Hour = str(DTime.hour) + '-' + str(DTime.minute) + '-' + str(DTime.second) + '.' + str(round((DTime.microsecond / 1000),0))[:-2]
    
    res_data += [str(Nrep)] + [str(Time)] + [str(NoOnNrep)] + [str(Range)] + [str(Velocity)] + [str(Azimuth)] + [str(Elevation)] + [str(Amplitudel10)] + [str(Amplitude)] + [str(DataNo)] + [str(Unixtime)] + [str(Date)] + [str(Hour)]

    # 追加するデータ出力
    print(res_data)
   
    # .CSV出力行に追加
    log_data.append(res_data)
    
    # バッファクリア
    res_data = list()

    # _adv.CSV出力行に追加するデータ作成
    #FrameID = Nrep
    SymID = Symbolid
    #Unixtime = Unixtime(流用)
    Xrange = fx
    Yrange = fy
    #反射強度 = Amplitude
    #速度 = Velocity
    
    res_data += [str(Nrep)] + [str(SymID)] + [str(Unixtime)] + [str(Xrange)] + [str(Yrange)] + [str(Amplitude)] + [str(Velocity)]
    # _adv.CSV出力行に追加
    log_data2.append(res_data)

    

    # .CSVのタイトル行を定義
    Title_line = [['Nrep', 'Time[s]', 'No in Nrep', 'Range[m]', 'Velocity[m/s]', 'Azimuth[deg]', 'Elevation[deg]', 'Amplitude[dB]', 'amplitude', 'Data No', 'UNIXTIME', 'date', 'time(h-m-s)']]
    csv_write(Title_line, log_data, '')

    # _adv.CSVのタイトル行を定義
    Title_line = [['FrameID', 'SymbolID', 'UnixTime', 'Xrange', 'Yrange', 'Amplitude', 'Velocity[m/s]']]
    csv_write(Title_line, log_data2, '_adv')

    return res_data


def csv_write(Title_Line, Write_line, optname):
    #csvファイル定義
    csv_file_path = filename_Unixtime + optname + '.csv'

    #ファイルオープン
    with open(csv_file_path, mode='w', newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        
        #タイトル行を書き込み
        writer.writerows(Title_Line)

        #引数データを以降の行に追記
        writer.writerows(Write_line)


def serial_send(data):
    # 送信処理
    global ser #グローバルオブジェクト定義
    
    data = data.encode('utf-8') #送信データのエンコード
    ser.write(data) #送信実行
    ser.flush() #バッファクリア


def sendCpuCommand(cmd):
    if cmd[0] == '%':
        exit

    buffer = bytearray(8)
    dummy = '########'
    data = cmd
    len_data = len(data)
    loop_cnt = int(len_data / 8)
    mod_cnt = int(len_data % 8)
    pos = 0
    
    for i in range(loop_cnt):
        buffer = data[pos:pos+8]
        canSendData(buffer)
        pos += 8

        if mod_cnt != 0:
            buffer = data[pos:len_data]
            buffer += dummy[mod_cnt:8]
            canSendData(buffer)


def canSendData(data_buff):
    canFrameData = 't'
    canFrameData += '0a1'
    canFrameData += '8'
    canFrameData += str(hex(ord(data_buff[0]))[2:])
    canFrameData += str(hex(ord(data_buff[1]))[2:])
    canFrameData += str(hex(ord(data_buff[2]))[2:])
    canFrameData += str(hex(ord(data_buff[3]))[2:])
    canFrameData += str(hex(ord(data_buff[4]))[2:])
    canFrameData += str(hex(ord(data_buff[5]))[2:])
    canFrameData += str(hex(ord(data_buff[6]))[2:])
    canFrameData += str(hex(ord(data_buff[7]))[2:])
    canFrameData += '\r'
    
    serial_send(canFrameData)
    recvwait()
    

# キーボード入力を非ブロッキングで取得する関数
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> int:
    global recv_data_sta
    
    # シリアルポート設定
    serial_port = "/dev/ttyUSB2"
    serial_baudrate = 115200*8 # 115200*8
    serial_timeout = 0.5 #[s]
    
    # シリアルポートオープン
    serial_open(serial_port, serial_baudrate, serial_timeout)

    # 受信スレッド起動
    receive_thread = threading.Thread(target = serial_recieve)
    receive_thread.start()

    #バッファクリア処理
    serial_send('\r')
    serial_send('\r')
    serial_send('\r')
    time.sleep(1)

    #バージョン表示
    serial_send('V\r')
    recvwait()

    #クローズ
    serial_send('C\r')
    recvwait()

    #CANボーレート設定
    serial_send('S8\r')
    recvwait()

    #CAN基本設定
    serial_send('M00000000\r')
    recvwait()
    serial_send('mFFFFFFFF\r')
    recvwait()

    #オープン
    serial_send('O\r')
    recvwait()
    
    #sensorStart送信
    sendCpuCommand("sensorStart")

    try:
        # メインスレッド処理実行
        while True:
            # キーボード入力を監視 (非ブロッキング)
            key = get_key()
            
            if key.lower() == 'q':
                # D1データ書き込み処理が完了するまで待機
                while recv_data_sta == 1:
                    pass

                # センサ停止コマンド送信
                sendCpuCommand('sensorStop')
                
                # CANUSBクローズ
                serial_send('C\r')
                recvwait()
                
                # シリアルポートクローズ
                serial_close()
                break
                
    except KeyboardInterrupt:
        pass
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
