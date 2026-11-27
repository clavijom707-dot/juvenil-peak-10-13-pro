import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile
import time
import random
import os
import math
import json
from datetime import datetime, timedelta
import google.generativeai as genai
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Juvenil Peak 10-13 Pro - Scouting IA", page_icon="⚽", layout="wide")

# --- CANDADO DE PRIVACIDAD (Cambia "futbol2026" por tu clave secreta) ---
clave_acceso = st.sidebar.text_input("🔒 Contraseña de Acceso:", type="password")
if clave_acceso != "futbol2026":
    st.warning("🔒 Ingresa la contraseña en la barra lateral izquierda para acceder al sistema.")
    st.stop()

DB_FILE = "historial_progreso_futbol.csv"
if not os.path.exists(DB_FILE):
    df_init = pd.DataFrame(columns=list((
        "Fecha", "Jugador", "Posicion", "Prueba", "Valor", "Unidad", "Pierna_Lado", "Notas"
    )))
    df_init.to_csv(DB_FILE, index=False)

@st.cache_resource
def obtener_modelos_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt"), YOLO("yolov8n-pose.pt")

CONEXIONES_ESQUELETO = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
)

def dibujar_esqueleto(frame, keypoints):
    puntos = {}
    for idx in range(len(keypoints)):
        x, y = int(keypoints.item(idx, 0)), int(keypoints.item(idx, 1))
        if x > 0 and y > 0:
            puntos.update({idx: (x, y)})
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for p1, p2 in CONEXIONES_ESQUELETO:
        if p1 in puntos and p2 in puntos:
            cv2.line(frame, puntos.get(p1), puntos.get(p2), (255, 50, 50), 2)

def calcular_angulo(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    if (ax == 0 and ay == 0) or (bx == 0 and by == 0) or (cx == 0 and cy == 0):
        return 0.0
    radianes = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angulo = abs(radianes * 180.0 / math.pi)
    if angulo > 180.0:
        angulo = 360.0 - angulo
    return round(float(angulo), 1)

# --- 1. ANALIZADOR INSTANTÁNEO DE CÁMARA ---
def analizar_captura_camara(image_pil, perfil, conf_yolo=0.25):
    yolo_balon, yolo_pose = obtener_modelos_yolo()
    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    res_balon = yolo_balon(frame, classes=list((32,)), conf=conf_yolo, verbose=False)
    balon_detectado = False
    for r in res_balon:
        for box in r.boxes:
            balon_detectado = True
            coords = box.xyxy.cpu().numpy().flatten()
            x1, y1, x2, y2 = int(coords.item(0)), int(coords.item(1)), int(coords.item(2)), int(coords.item(3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)
            cv2.putText(frame, "BALON DETECTADO", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

    ang_i, ang_d, ang_t = 0.0, 0.0, 0.0
    res_pose = yolo_pose(frame, conf=0.4, verbose=False)
    for r in res_pose:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kp = r.keypoints.xy.cpu().numpy()
            if len(kp) > 0:
                persona = kp.reshape(-1, 2)[:17]
                dibujar_esqueleto(frame, persona)
                
                hombro_i = (persona.item(5, 0), persona.item(5, 1))
                cadera_i = (persona.item(11, 0), persona.item(11, 1))
                rodilla_i = (persona.item(13, 0), persona.item(13, 1))
                tobillo_i = (persona.item(15, 0), persona.item(15, 1))
                cadera_d = (persona.item(12, 0), persona.item(12, 1))
                rodilla_d = (persona.item(14, 0), persona.item(14, 1))
                tobillo_d = (persona.item(16, 0), persona.item(16, 1))

                ang_i = calcular_angulo(cadera_i, rodilla_i, tobillo_i)
                ang_d = calcular_angulo(cadera_d, rodilla_d, tobillo_d)
                ang_t = calcular_angulo(hombro_i, cadera_i, rodilla_i)

                cv2.rectangle(frame, (10, 10), (430, 130), (20, 20, 20), -1)
                cv2.putText(frame, f"Rodilla Izq: {ang_i} deg | Der: {ang_d} deg", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Inclinacion Torso: {ang_t} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Perfil: {perfil[:24]}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                break

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {"Rodilla_Izq": ang_i, "Rodilla_Der": ang_d, "Torso": ang_t, "Balon": balon_detectado}

# --- 2. ANALIZADOR COMPLETO DE VIDEO ---
def analizar_video_completo(video_path, perfil, conf_yolo=0.25, diametro_balon_m=0.22, estatura_jugador_m=1.45):
    yolo_balon, yolo_pose = obtener_modelos_yolo()
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    angulos_rodilla_izq, angulos_rodilla_der, angulos_torso = list(), list(), list()
    historial_cadera = list()
    velocidades_jugador_ms = list()
    historial_balon = list()
    velocidades_balon_kmh = list()
    toques_detectados = 0
    ultimo_frame_toque = -15
    giros_cabeza = 0
    estado_cabeza_previo = "CENTRO"
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        res_balon = yolo_balon(frame, classes=list((32,)), conf=conf_yolo, verbose=False)
        mejor_caja = None
        mayor_conf = 0.0
        for r in res_balon:
            for box in r.boxes:
                conf = float(box.conf.item(0))
                if conf > mayor_conf:
                    mayor_conf = conf
                    mejor_caja = box.xyxy.cpu().numpy().flatten()

        velocidad_actual_kmh = 0.0
        if mejor_caja is not None:
            x1, y1, x2, y2 = int(mejor_caja.item(0)), int(mejor_caja.item(1)), int(mejor_caja.item(2)), int(mejor_caja.item(3))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ancho_px = max(x2 - x1, y2 - y1)
            historial_balon.append((frame_idx, cx, cy, ancho_px))

            if len(historial_balon) >= 2:
                f_prev, px_prev, py_prev, ancho_prev = historial_balon.__getitem__(-2)
                diff_frames = frame_idx - f_prev
                if 1 <= diff_frames <= 4:
                    dist_px = math.hypot(cx - px_prev, cy - py_prev)
                    tamano_ref_px = max((ancho_px + ancho_prev) / 2.0, 12.0)
                    metros_por_pixel = diametro_balon_m / tamano_ref_px
                    vel_ms = (dist_px * metros_por_pixel) / (diff_frames / fps)
                    velocidad_actual_kmh = round(vel_ms * 3.6, 1)
                    if velocidad_actual_kmh < 110:
                        velocidades_balon_kmh.append(velocidad_actual_kmh)

            if len(historial_balon) >= 3 and (frame_idx - ultimo_frame_toque) > int(fps * 0.25):
                _, _, y_a, _ = historial_balon.__getitem__(-3)
                _, _, y_b, _ = historial_balon.__getitem__(-2)
                _, _, y_c, _ = historial_balon.__getitem__(-1)
                vy1, vy2 = y_b - y_a, y_c - y_b
                if (vy1 > 2 and vy2 < -2) or (abs(vy2 - vy1) > 18):
                    toques_detectados += 1
                    ultimo_frame_toque = frame_idx
                    cv2.circle(frame, (cx, cy), 35, (0, 255, 0), 4)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)
            cv2.putText(frame, f"Balon {velocidad_actual_kmh} km/h", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

            for i in range(max(1, len(historial_balon) - 10), len(historial_balon)):
                pt1 = (historial_balon.__getitem__(i-1).__getitem__(1), historial_balon.__getitem__(i-1).__getitem__(2))
                pt2 = (historial_balon.__getitem__(i).__getitem__(1), historial_balon.__getitem__(i).__getitem__(2))
                cv2.line(frame, pt1, pt2, (0, 165, 255), 2)

        res_pose = yolo_pose(frame, conf=0.4, verbose=False)
        for r in res_pose:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                kp = r.keypoints.xy.cpu().numpy()
                if len(kp) > 0:
                    persona = kp.reshape(-1, 2)[:17]
                    dibujar_esqueleto(frame, persona)

                    nariz_x, nariz_y = persona.item(0, 0), persona.item(0, 1)
                    hombro_i = (persona.item(5, 0), persona.item(5, 1))
                    hombro_d = (persona.item(6, 0), persona.item(6, 1))
                    cadera_i = (persona.item(11, 0), persona.item(11, 1))
                    rodilla_i = (persona.item(13, 0), persona.item(13, 1))
                    tobillo_i = (persona.item(15, 0), persona.item(15, 1))
                    cadera_d = (persona.item(12, 0), persona.item(12, 1))
                    rodilla_d = (persona.item(14, 0), persona.item(14, 1))
                    tobillo_d = (persona.item(16, 0), persona.item(16, 1))

                    ang_i = calcular_angulo(cadera_i, rodilla_i, tobillo_i)
                    ang_d = calcular_angulo(cadera_d, rodilla_d, tobillo_d)
                    ang_t = calcular_angulo(hombro_i, cadera_i, rodilla_i)
                    if ang_i > 0: angulos_rodilla_izq.append(ang_i)
                    if ang_d > 0: angulos_rodilla_der.append(ang_d)
                    if ang_t > 0: angulos_torso.append(ang_t)

                    centro_cadera_x = (cadera_i.__getitem__(0) + cadera_d.__getitem__(0)) / 2.0
                    centro_cadera_y = (cadera_i.__getitem__(1) + cadera_d.__getitem__(1)) / 2.0
