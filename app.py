import socket
import struct
import numpy as np
import open3d as o3d
import time
import os
from threading import Thread, Lock
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import colorsys

# === ПАРАМЕТРЫ LiDAR ===
LIDAR_IP = "192.168.1.200"
HOST_IP = "192.168.1.102"
MSOP_PORT = 6699

# Глобальные данные
latest_points = np.empty((0, 3))
points_lock = Lock()
latest_temp = 25.0
latest_sync = "Unknown"

# Папка для сохранённых PCD
PCD_SAVE_DIR = Path("saved_pcds")
PCD_SAVE_DIR.mkdir(exist_ok=True)

# Flask + SocketIO
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'lidar_secret'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    engineio_options={"max_decode_packets": 256}
)

# ==========================================================
# Функции парсинга
# ==========================================================
def get_pkt_cnt(data):
    if len(data) < 6:
        return None
    sync = data[0:4]
    if sync != b'\x55\xaa\x5a\xa5':
        return None
    return struct.unpack('>H', data[4:6])[0]

def parse_msop_header(data):
    if len(data) < 32 or data[0:4] != b'\x55\xaa\x5a\xa5':
        return None
    lidar_tmp = struct.unpack('B', data[31:32])[0]
    temp_c = lidar_tmp - 80
    time_mode = struct.unpack('B', data[7:8])[0]
    time_modes = {0: "Internal", 2: "PTP E2E", 3: "gPTP"}
    sync_mode = time_modes.get(time_mode, "Unknown")
    return {"temperature": temp_c, "sync_mode": sync_mode}

def parse_points_only(data):
    if len(data) != 1200 or data[0:4] != b'\x55\xaa\x5a\xa5':
        return None
    points = []
    for i in range(96):
        base = 32 + i * 12
        radius_raw = struct.unpack('>H', data[base + 2:base + 4])[0]
        if radius_raw == 0:
            continue
        radius_m = radius_raw * 0.005
        if radius_m > 35.0:
            continue
        dx_raw = struct.unpack('>h', data[base + 4:base + 6])[0]
        dy_raw = struct.unpack('>h', data[base + 6:base + 8])[0]
        dz_raw = struct.unpack('>h', data[base + 8:base + 10])[0]
        dx = dx_raw / 32768.0
        dy = dy_raw / 32768.0
        dz = dz_raw / 32768.0
        intensity = data[base + 10]
        if intensity < 1:
            continue
        x = dx * radius_m
        y = dy * radius_m
        z = dz * radius_m
        points.append([x, y, z])
    return np.array(points, dtype=np.float32) if points else None

# ==========================================================
# Захват кадра
# ==========================================================
def capture_one_frame(sock):
    frame_points = []
    frame_started = False
    packet_count = 0
    MAX_PACKETS = 350

    while True:
        try:
            data, addr = sock.recvfrom(1500)
            header_info = parse_msop_header(data)
            if header_info:
                global latest_temp, latest_sync
                latest_temp = header_info["temperature"]
                latest_sync = header_info["sync_mode"]
        except socket.timeout:
            return None

        pkt_cnt = get_pkt_cnt(data)
        if pkt_cnt is None:
            continue

        if pkt_cnt == 0:
            if frame_started and packet_count > 20:
                break
            frame_points = []
            frame_started = True
            packet_count = 0

        if not frame_started:
            continue

        pts = parse_points_only(data)
        if pts is not None and pts.size > 0:
            frame_points.append(pts)
        packet_count += 1

        if packet_count >= MAX_PACKETS:
            break

    return np.vstack(frame_points) if frame_points else None

# ==========================================================
# Фоновый поток: читает LiDAR
# ==========================================================
def lidar_reader():
    global latest_points
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind((HOST_IP, MSOP_PORT))
    sock.settimeout(1.0)
    print("📡 Запущен фоновый поток чтения LiDAR...")

    while True:
        cloud = capture_one_frame(sock)
        if cloud is not None and len(cloud) > 0:
            with points_lock:
                latest_points = cloud.copy()

# ==========================================================
# Кластеризация DBSCAN
# ==========================================================
def cluster_pointcloud(points, eps=0.5, min_points=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    return labels

# ==========================================================
# Кластеризация RANSAC
# ==========================================================
def remove_ground(points, distance_threshold=0.1, max_iterations=100):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Сегментация плоскости (пол/стена)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=max_iterations
    )
    
    # Удаляем "inliers" — точки на плоскости
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return np.asarray(outlier_cloud.points)
# ==========================================================
# Маршруты
# ==========================================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/save_pcd', methods=['POST'])
def save_pcd():
    with points_lock:
        pts = latest_points.copy()
    if len(pts) == 0:
        return jsonify({"error": "No points to save"}), 400
    timestamp = int(time.time())
    filename = f"cloud_{timestamp}.pcd"
    filepath = PCD_SAVE_DIR / filename
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(filepath), pcd)
    return jsonify({"success": True, "filename": filename})

@app.route('/list_pcds')
def list_pcds():
    files = sorted([f.name for f in PCD_SAVE_DIR.glob("*.pcd")], reverse=True)
    return jsonify({"files": files})

@app.route('/load_pcd/<filename>')
def load_pcd(filename):
    filepath = PCD_SAVE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    pcd = o3d.io.read_point_cloud(str(filepath))
    points = np.asarray(pcd.points)
    flat = points.flatten().tolist()
    return jsonify({"points": flat, "count": len(points)})

@app.route('/cluster_pcd', methods=['POST'])
def cluster_pcd():
    data = request.get_json()
    filename = data.get("filename")
    eps = float(data.get("eps", 0.5))
    min_points = int(data.get("min_points", 10))

    filepath = PCD_SAVE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    pcd = o3d.io.read_point_cloud(str(filepath))
    #points = np.asarray(pcd.points) #Без RANSAC 
    points = remove_ground(np.asarray(pcd.points), distance_threshold=0.08) #С RANSAC 
    labels = cluster_pointcloud(points, eps=eps, min_points=min_points)

    max_label = labels.max()
    if max_label == -1:
        colors = np.zeros((len(points), 3))  # чёрный — шум
    else:
        hue = np.linspace(0, 1, max_label + 1)
        colors = np.zeros((len(points), 3))
        for i, h in enumerate(hue):
            rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            colors[labels == i] = rgb
        colors[labels == -1] = [0, 0, 0]

    # Генерация AABB боксов
    boxes = []
    for i in range(max_label + 1):
        cluster_points = points[labels == i]
        if len(cluster_points) == 0:
            continue
        min_bound = cluster_points.min(axis=0).tolist()
        max_bound = cluster_points.max(axis=0).tolist()
        boxes.append({
            "min": min_bound,
            "max": max_bound,
            "color": list(colorsys.hsv_to_rgb(hue[i], 1.0, 1.0))
        })

    return jsonify({
        "points": points.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "boxes": boxes,
        "count": len(points)
    })

# ==========================================================
# WebSocket: отправка облака точек
# ==========================================================
@socketio.on('request_pointcloud')
def handle_request():
    global latest_points, latest_temp, latest_sync
    with points_lock:
        pts = latest_points.copy()
    if len(pts) > 0:
        max_pts = 30000
        if len(pts) > max_pts:
            indices = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[indices]
        flat = pts.flatten().tolist()
        emit('pointcloud_update', {
            'points': flat,
            'count': len(pts),
            'temperature': latest_temp,
            'sync_mode': latest_sync
        })

# ==========================================================
# Запуск
# ==========================================================
if __name__ == '__main__':
    lidar_thread = Thread(target=lidar_reader, daemon=True)
    lidar_thread.start()
    print("Запуск веб-сервера на http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)