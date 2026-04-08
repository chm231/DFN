import os
import argparse
import pandas as pd

def export_to_cad(input_csv, out_dir):
    """
    Trace 정보를 읽어들여 CAD에서 시각화할 수 있도록 3D 선분(Lines) 데이터를 추출 및 포맷 변환합니다.
    - 1. CSV 포맷: CAD의 LISP/스크립트에서 쉽게 읽을 수 있도록 시작점/끝점 3차원 절대 좌표 제공
    - 2. SCR 포맷: AutoCAD 스크립트 파일 (드래그앤드롭으로 바로 3D 공간상에 선분을 그림)
    - 3. DXF 포맷: ezdxf 패키지가 있을 경우 DXF (R2010) 직접 생성
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 2D Trace 데이터 로드
    if not os.path.exists(input_csv):
        print(f"[Error] 입력 파일을 찾을 수 없습니다: {input_csv}")
        return
        
    df = pd.read_csv(input_csv)
    print(f"[Info] 로드된 Trace 개수: {len(df)}")
    
    # 3D 좌표 변환
    # 터널 진행 방향이 X축이므로, x_face가 곧 X 좌표가 됩니다.
    df_cad = pd.DataFrame()
    df_cad['Face_ID'] = df['face_id']
    df_cad['Trace_ID'] = df['trace_id']
    
    # Start Point
    df_cad['X1'] = df['x_face']
    df_cad['Y1'] = df['y0']
    df_cad['Z1'] = df['z0']
    
    # End Point
    df_cad['X2'] = df['x_face']
    df_cad['Y2'] = df['y1']
    df_cad['Z2'] = df['z1']
    
    # --- Output 1. 3D Coordinates CSV ---
    csv_out = os.path.join(out_dir, "traces_3d_coordinates.csv")
    df_cad.to_csv(csv_out, index=False)
    print(f" -> [Export] 3D 좌표 CSV 생성: {csv_out}")
    
    # --- Output 2. AutoCAD Script (SCR) ---
    scr_out = os.path.join(out_dir, "draw_traces_autocad.scr")
    with open(scr_out, 'w') as f:
        # Layer 생성 (선택 사항)
        f.write("-LAYER\nMAKE\nTrace_Lines\nCOLOR\n1\nTrace_Lines\n\n")
        
        for _, row in df_cad.iterrows():
            # LINE 명령어 호출
            f.write("LINE\n")
            # 시작점
            f.write(f"{row['X1']},{row['Y1']},{row['Z1']}\n")
            # 끝점
            f.write(f"{row['X2']},{row['Y2']},{row['Z2']}\n")
            # 종료 (엔터)
            f.write("\n")
            
        f.write("ZOOM\nEXTENTS\n")
    print(f" -> [Export] AutoCAD SCR 생성: {scr_out} (AutoCAD 창에 드래그 앤 드롭하시면 선분이 그려집니다.)")
    
    # --- Output 3. DXF File (Optional) ---
    try:
        import ezdxf
        dxf_out = os.path.join(out_dir, "traces_3d.dxf")
        doc = ezdxf.new('R2010')
        doc.layers.new(name='Trace_Lines', dxfattribs={'color': 1}) # Color 1 is Red
        msp = doc.modelspace()
        
        for _, row in df_cad.iterrows():
            start_pt = (row['X1'], row['Y1'], row['Z1'])
            end_pt = (row['X2'], row['Y2'], row['Z2'])
            msp.add_line(start_pt, end_pt, dxfattribs={'layer': 'Trace_Lines'})
            
        doc.saveas(dxf_out)
        print(f" -> [Export] DXF 파일 생성 성공: {dxf_out}")
        
    except ImportError:
        print(" -> [Info] 'ezdxf' 패키지가 설치되어 있지 않아 DXF 파일은 생성하지 않습니다. (pip install ezdxf)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tunnel Face 2D Traces -> 3D CAD Data Exporter")
    parser.add_argument("--input", default="inverse_results/synthetic_face_traces.csv", help="분석 결과로 나온 CSV 경로")
    parser.add_argument("--outdir", default="cad_export", help="저장할 디렉토리 경로")
    
    args = parser.parse_args()
    print("="*60)
    print(" [Trace to CAD Converter] ")
    print("="*60)
    export_to_cad(args.input, args.outdir)
    print("="*60)
