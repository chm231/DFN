import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List
from .data_models import Face, FractureHypothesis

def plot_reconstruction_results(faces: List[Face], hypotheses: List[FractureHypothesis]):
    """복원 결과를 3D로 시각화 (Matplotlib)"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 막장면 표시 (회색 평면 가이드)
    for face in faces:
        x = face.plane_point[0]
        yy, zz = np.meshgrid(np.linspace(-10, 10, 2), np.linspace(-10, 10, 2))
        ax.plot_surface(np.full_like(yy, x), yy, zz, alpha=0.1, color='gray')
        
        # Trace 표시 (검은색 선)
        for t in face.traces:
            pts = t.endpoints_3d
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'k-', linewidth=2)

    # 2. 복원된 원판 표시
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(hypotheses))))
    for i, h in enumerate(hypotheses):
        c = h.center
        n = h.normal
        r = h.radius
        
        # 원판 그리드 생성
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, r, 5)
        T, P = np.meshgrid(theta, phi)
        
        # 로컬 좌표계 산출
        v1 = np.array([0, 1, 0]) if np.abs(n[0]) > 0.9 else np.array([1, 0, 0])
        u1 = np.cross(n, v1)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(n, u1)
        
        # 카테시안 변환
        X = c[0] + P * (np.cos(T) * u1[0] + np.sin(T) * u2[0])
        Y = c[1] + P * (np.cos(T) * u1[1] + np.sin(T) * u2[1])
        Z = c[2] + P * (np.cos(T) * u1[2] + np.sin(T) * u2[2])
        
        ax.plot_surface(X, Y, Z, color=colors[i % 10], alpha=0.4)
        ax.text(c[0], c[1], c[2], f"ID:{i}\nConf:{h.confidence:.2f}", fontsize=8)

    ax.set_xlabel('X (Advance)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SC-PMFR Reconstruction Prototype')
    plt.show()

def plot_traces_2d(face: Face):
    """특정 막장의 Trace 분포를 2D로 확인"""
    plt.figure(figsize=(6, 6))
    for t in face.traces:
        pts = t.endpoints_3d
        plt.plot(pts[:, 1], pts[:, 2], 'b-')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(f"Face {face.face_id} Trace Map")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
