import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

def export_blocks_csv(block_info: list, outdir: str):
    """
    blocks.csv 형식으로 블록 정보를 출력합니다.
    (요청된 스키마에 따라 누락된 필드는 NaN 처리)
    """
    if not block_info:
        print("[Export] 추출할 블록이 없습니다. blocks.csv 저장을 건너뜁니다.")
        return

    data = []
    for b in block_info:
        cx, cy, cz = b['centroid']
        data.append({
            'Block_ID': int(b['label']),
            'Volume': float(b['volume_m3']),
            'Centroid_X': float(cx),
            'Centroid_Y': float(cy),
            'Centroid_Z': float(cz),
            'Material_ID': np.nan,  # 규칙에 따라 현재는 NaN
        })

    df = pd.DataFrame(data)
    if outdir.endswith('.csv'):
        csv_path = outdir
    else:
        csv_path = os.path.join(outdir, 'blocks.csv')
        
    # ensure parent dir exists
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False, na_rep='NaN')
    
    nan_cols = df.columns[df.isna().any()].tolist()
    print(f"[Export] {csv_path} 저장 성공")
    print(f"  - Exported block count: {len(data)}")
    if nan_cols:
        print(f"  - NaN 포함 블록 필드 요약: {', '.join(nan_cols)}")


def export_interfaces_csv(labels: np.ndarray, block_info: list, grid_info: dict, outdir: str):
    """
    interfaces.csv 형식으로 블록-블록 및 블록-기반암 간의 접촉면을 추출하여 저장합니다.
    """
    if not block_info:
        print("[Export] 추출할 블록이 없습니다. interfaces.csv 저장을 건너뜁니다.")
        return

    vs = float(grid_info['voxel_size'])
    voxel_area = vs ** 2
    struct = ndi.generate_binary_structure(3, 1)  # 6-connectivity for strict surface touch

    valid_labels = set([int(b['label']) for b in block_info])
    faces = []
    face_id = 1

    print("[Export] 접촉면 교집합 공간분석(Interface Extraction)을 시작합니다...")
    
    # Bounding Box 활용 거대 배열 고속 접근
    try:
        slices = ndi.find_objects(labels)
    except Exception:
        slices = None

    # 중복 기록 방지를 위해 집계 (A, B)
    # A < B (Block A -> Block B) 또는 B=0 (Block A -> Bedrock)
    interface_map = {}

    for b in block_info:
        lbl_A = int(b['label'])
        
        # Fast local processing using slice padding
        if slices is not None and lbl_A <= len(slices) and slices[lbl_A-1] is not None:
            s = slices[lbl_A-1]
            padded_bbox = tuple(slice(max(0, sl.start - 1), min(labels.shape[dim], sl.stop + 1)) 
                                for dim, sl in enumerate(s))
            sub_labels = labels[padded_bbox]
        else:
            # Fallback
            sub_labels = labels

        mask_A = (sub_labels == lbl_A)
        
        dilated_A = ndi.binary_dilation(mask_A, structure=struct)
        shell_A = dilated_A ^ mask_A  # XOR (순수 껍질 영역)
        
        touching_labels = sub_labels[shell_A]
        unique_labels, counts = np.unique(touching_labels, return_counts=True)
        
        for lbl_adj, count in zip(unique_labels, counts):
            if lbl_adj == lbl_A:
                continue
                
            # 기반암(Bedrock)이거나 삭제/비유효 라벨인 경우 모두 0(Bedrock)로 처리
            lbl_B = int(lbl_adj)
            if lbl_B not in valid_labels:
                lbl_B = 0
                
            # 항상 작은 ID를 A, 큰 ID를 B로 두어 양항 중복 검출을 취합
            # (단, 기반암 0인경우는 항상 B에 할당, 즉 (A, 0))
            if lbl_B == 0:
                pair = (lbl_A, 0)
            else:
                pair = (min(lbl_A, lbl_B), max(lbl_A, lbl_B))
                
            if pair not in interface_map:
                interface_map[pair] = 0.0
            interface_map[pair] += float(count * voxel_area)
            
    # Dictionary 내 데이터를 List로 조립하면서 Area를 양방향 오차 평균 보완
    # 복셀 경계에서는 A->B와 B->A 측정 면적이 서로 약간 다를 수 있으므로 합산 후 2로 나눔
    # 단, 기반암 접속(0)은 단방향에서만 계산되었으므로 나누지 않음.
    for pair, area_sum in interface_map.items():
        if pair[1] != 0:
            area = area_sum / 2.0
        else:
            area = area_sum
            
        faces.append({
            'Face_ID': face_id,
            'Block_A': pair[0],
            'Block_B': pair[1],
            'Area': area,
            'Nx': np.nan,  # 현재 법선 도출 불가로 np.nan 할당
            'Ny': np.nan,
            'Nz': np.nan,
            'Cohesion': np.nan,
            'Friction': np.nan,
        })
        face_id += 1

    df = pd.DataFrame(faces)
    csv_path = os.path.join(outdir, 'interfaces.csv')
    df.to_csv(csv_path, index=False, na_rep='NaN')
    
    nan_cols = df.columns[df.isna().any()].tolist()
    print(f"[Export] {csv_path} 저장 성공")
    print(f"  - Exported interface count: {len(faces)}")
    if nan_cols:
        print(f"  - NaN 포함 인터페이스 필드 요약: {', '.join(nan_cols)}")
