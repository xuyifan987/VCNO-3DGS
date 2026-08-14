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

import torch
import traceback
import socket
import json
import struct
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def send_json_data(conn, data):
    # Serialize the list of strings to JSON
    serialized_data = json.dumps(data)
    # Convert the serialized data to bytes
    bytes_data = serialized_data.encode('utf-8')
    # Send the length of the serialized data first
    conn.sendall(struct.pack('I', len(bytes_data)))
    # Send the actual serialized data
    conn.sendall(bytes_data)

def try_connect(render_items):
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        # print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
        # raise inst
            
def recvall(length):
    global conn
    chunks = []
    received = 0
    while received < length:
        chunk = conn.recv(length - received)
        if chunk == b"":
            raise ConnectionError("Viewer socket closed while receiving data.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)

def read():
    messageLength = int.from_bytes(recvall(4), 'little')
    message = recvall(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify, metrics=None):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))
    if metrics is not None:
        send_json_data(conn, metrics)

def receive():
    message = read()
    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        custom_cam = None
        do_training = False
        keep_alive = True
        scaling_modifier = 1.0
        render_mode = "RGB"
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            render_mode = message.get("render_mode", "RGB")
        except Exception as e:
            print("")
            traceback.print_exc()
            # raise e
        return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
    else:
        return None, None, None, None, None
