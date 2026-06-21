import numpy as np
from typing import List, Dict, Tuple
from .slab_types import LocalCandidate, SlabLink, ReconstructedPlane
from .plane_fitting import fit_plane_svd

def merge_links_into_chains(links: List[SlabLink]) -> List[List[Tuple[int, int]]]:
    """
    SlabLink 리스트를 분석하여 연속된 체인(List of (slab_idx, cand_id))들을 생성
    """
    if not links:
        return []
        
    # (slab_idx, cand_id) -> next (slab_idx, cand_id) mapping
    adj = {}
    nodes = set()
    for l in links:
        node_A = (l.slab_idx_A, l.id_A)
        node_B = (l.slab_idx_B, l.id_B)
        adj[node_A] = node_B
        nodes.add(node_A)
        nodes.add(node_B)
        
    # 이미 다음 노드로 지목된 노드들 (시작점이 될 수 없음)
    has_in_edge = set(adj.values())
    starts = [n for n in nodes if n not in has_in_edge]
    
    chains = []
    for s in starts:
        curr_chain = [s]
        curr = s
        while curr in adj:
            curr = adj[curr]
            curr_chain.append(curr)
        chains.append(curr_chain)
        
    return chains

def reconstruct_global_planes(
    all_candidates: Dict[int, List[LocalCandidate]], 
    chains: List[List[Tuple[int, int]]]
) -> List[ReconstructedPlane]:
    """
    매칭된 체인들을 하나의 ReconstructedPlane으로 통합
    """
    reconstructed = []
    
    for i, chain in enumerate(chains):
        all_pts_list = []
        slab_indices = []
        
        # 각 마디(Slab)의 포인트들 수집
        for slab_idx, cand_id in chain:
            cand = all_candidates[slab_idx][cand_id]
            all_pts_list.append(cand.points)
            slab_indices.append(slab_idx)
            
        if not all_pts_list:
            continue
            
        all_pts = np.vstack(all_pts_list)
        
        # 전역 평면 피팅
        normal, centroid, residual = fit_plane_svd(all_pts)
        
        reconstructed.append(ReconstructedPlane(
            plane_id=i+1,
            points=all_pts,
            normal=normal,
            centroid=centroid,
            residual=residual,
            source_slab_indices=slab_indices
        ))
        
    return reconstructed
