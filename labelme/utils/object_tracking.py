import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import trackpy as tp


class ObjectTracking:
    def __init__(self, list_of_labels: list[list[tuple[float, float]]], 
                 max_distance: float = 50.0, search_range: float = None):
        self.list_of_labels = list_of_labels
        self.global_id = {}  # {(frame_idx, obj_idx): global_id}
        self.max_distance = max_distance
        self.search_range = search_range if search_range is not None else max_distance

    def _prepare_dataframe(self) -> pd.DataFrame:
        data = []
        for frame_idx, frame_objects in enumerate(self.list_of_labels):
            for obj_idx, (x, y) in enumerate(frame_objects):
                data.append({
                    'frame': frame_idx,
                    'x': float(x),
                    'y': float(y),
                    'obj_idx': obj_idx
                })
        return pd.DataFrame(data)

    def assign_global_id(self) -> Dict[Tuple[int, int], int]:
        if not self.list_of_labels:
            return {}
        
        df = self._prepare_dataframe()
        
        if df.empty:
            return {}
        
        try:
            linked = tp.link(df, search_range=self.search_range, memory=0)
        except Exception as e:
            print(f"Error in trackpy.link: {e}")
            linked = tp.link_df(df, search_range=self.search_range, memory=0)
        
        self.global_id = {}
        particle_to_global = {}
        next_global_id = 0
        
        for _, row in linked.iterrows():
            frame_idx = int(row['frame'])
            obj_idx = int(row['obj_idx'])
            particle_id = int(row['particle'])
            
            if particle_id not in particle_to_global:
                particle_to_global[particle_id] = next_global_id
                next_global_id += 1
            
            global_id = particle_to_global[particle_id]
            self.global_id[(frame_idx, obj_idx)] = global_id
        
        return self.global_id

    def get_global_id(self, frame_idx: int, obj_idx: int) -> int:
        return self.global_id.get((frame_idx, obj_idx), -1)