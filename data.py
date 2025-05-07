# -*- coding: utf-8 -*-
import sys
#from module.log_module import log_module

import pandas as pd
import numpy as np
import networkx as nx
import math
import geopandas as gpd
from shapely.geometry import Point, LineString

def read_df(datapath, header):

    if datapath.endswith("csv"):
        df = pd.read_csv(datapath, header=header)
    else:
        df = pd.read_csv(datapath, sep='\t', header=header)

    return df

class _Trajectory:
    def __init__(self): 

        self.trajectory = self._TrajectoryData()
 

    class _TrajectoryData:
        pass

    def set_observation(self, datapath, id_col=0, x_col=1, y_col=2, header=None):
 

        df = read_df(datapath, header)
        if id_col == None:
            df["id"] = df.index
        if header == None:
            df = df.rename(columns={id_col: "id", x_col: 'x', y_col: 'y'})
        self.trajectory.observation_df = df  # [["id","x","y"]]
 

    def set_truth(self, datapath, header=None):
 

        df = read_df(datapath, header)
        self.trajectory.truth_array = np.ravel(df.values)
 

class _Network:
    def __init__(self):
 

        self.network = self._NetworkData()
 

    class _NetworkData:
        pass

    def set_node(self, datapath, id_col=0, x_col=1, y_col=2, header=None):
 

        df = read_df(datapath, header)
        if id_col == None:
            df["id"] = df.index
        if header == None:
            df = df.rename(columns={id_col: "id", x_col: 'x', y_col: 'y'})
        self.network.node_df = df  # [["id","x","y"]]
 

    def set_link(self, datapath, id_col=0, source_col=1, target_col=2, header=None):
 

        df = read_df(datapath, header)
        if id_col == None:
            df["id"] = df.index
        if header == None:
            df = df.rename(columns={id_col: "id", source_col: 'source', target_col: 'target'})
        self.network.link_df = df  # [["id", 'source','target']]
 

    def _calc_linklength(self, n1_x, n1_y, n2_x, n2_y):
 
 
        return math.sqrt((n1_x - n2_x) ** 2 + (n1_y - n2_y) ** 2)

    def _set_linkweight(self, G):
 

        for e in G.edges:
            n1, n2 = G.nodes[e[0]], G.nodes[e[-1]]
            G.edges[e]["weight"] = self._calc_linklength(n1["x"], n1["y"], n2["x"], n2["y"])
 
        return G

    def _setup_graph(self, G):
 

        self.network.link_id_dict = nx.get_edge_attributes(G, "id")
        self.network.link_id_dict_inv = {v: k for k, v in self.network.link_id_dict.items()}

        node_dict = self.network.node_df.to_dict(orient='index')
        nx.set_node_attributes(G, node_dict)

        self.network.node_id_dict = nx.get_node_attributes(G, "id")
        self.network.node_id_dict_inv = {v: k for k, v in self.network.node_id_dict.items()}

        G = self._set_linkweight(G)
        self.network.G = G
 

    def get_node_df_of_G(self):
 

        df = self.network.node_df[
            self.network.node_df["id"].isin(list(nx.get_node_attributes(self.network.G, "id").values()))]
 
        return df

    def get_node_df_of_argG(self, G):
        df = self.network.node_df[
            self.network.node_df["id"].isin(list(nx.get_node_attributes(G, "id").values()))]
        return df

    def get_link_df_of_G(self):
 

        df = self.network.link_df[
            self.network.link_df["id"].isin(list(nx.get_edge_attributes(self.network.G, "id").values()))]
 
        return df

    def get_link_df_of_argG(self, G):
        df = self.network.link_df[
            self.network.link_df["id"].isin(list(nx.get_edge_attributes(G, "id").values()))]
        return df

    def transform_nodes_to_links(self, nodes):
 

        edge_id_list = []
        if len(nodes) >= 1:
            for i in range(len(nodes) - 1):
                s = self.get_node_index_of_G(nodes[i])
                t = self.get_node_index_of_G(nodes[i + 1])
                if self.network.G.has_edge(s, t):
                    edge_id = self.network.G.edges[s, t]["id"]
                    edge_id_list.append(edge_id)
 
        return edge_id_list

    def get_link_index_of_G(self, link_id):
 

        if type(link_id) == list:
            link_index = [self.network.link_id_dict_inv[_link_id] for _link_id in link_id]
        else:
            link_index = self.network.link_id_dict_inv[link_id]
 
        return link_index

    def get_node_index_of_G(self, node_id):
 

        if type(node_id) == list:
            node_index = [self.network.node_id_dict_inv[_node_id] for _node_id in node_id]
        else:
            node_index = self.network.node_id_dict_inv[node_id]
 
        return node_index
    
    def get_length_by_id(self,link_id):
        G = self.network.G_total
        for u, v, data in G.edges(data = True):
            if data.get("id") == link_id:
                return data.get("weight", None)
            elif str(data.get("id")) == str(link_id):
                return data.get("weight", None)
        return None  # 見つからない場合

    def make_link_geom(self):
 

        if not hasattr(self.network, 'link_gdf'):
            self.network.link_gdf=gpd.GeoDataFrame(self.network.link_df.copy())
            self.network.link_gdf["geometry"]=None
            for i in self.network.link_gdf.index:
                s, t = self.network.link_gdf["source"][i], self.network.link_gdf["target"][i]
                s_x, s_y, *_ = self.network.node_df.loc[self.network.node_df["id"]==s,["x","y"]].values.tolist()[0]
                t_x, t_y, *_ = self.network.node_df.loc[self.network.node_df["id"] == t, ["x", "y"]].values.tolist()[0]
                self.network.link_gdf["geometry"][i] = LineString([(s_x, s_y), (t_x, t_y)])
            self.network.link_gdf["length"]=self.network.link_gdf["geometry"].apply(lambda x:x.length)
 

class Data(_Trajectory, _Network):
    def __init__(self):
        
 

        _Trajectory.__init__(self)
        _Network.__init__(self)
 

    def reproject_crs(self, to_crs, from_crs="EPSG:4326"):
 

        self.trajectory.observation_df = self._func_reproject_crs(self.trajectory.observation_df, to_crs, from_crs)
        self.network.node_df = self._func_reproject_crs(self.network.node_df, to_crs, from_crs)
 

    def _func_reproject_crs(self, df, to_crs, from_crs):
 

        df = df.rename(columns={"x": f"x_{from_crs}", "y": f"y_{from_crs}"})
        df["geometry"] = df.apply(lambda x: Point(x[f"x_{from_crs}"], x[f"y_{from_crs}"]), axis=1)
        gdf = gpd.GeoDataFrame(df, crs=from_crs).to_crs(to_crs)
        df["x"] = gdf["geometry"].apply(lambda x: x.x)
        df["y"] = gdf["geometry"].apply(lambda x: x.y)
 
        return df

    def create_graph(self):
 

        G = nx.from_pandas_edgelist(self.network.link_df, "source", "target", "id", create_using=nx.DiGraph)
        self._setup_graph(G)
        self.network.G_total = self.network.G
        
        G = G.subgraph(max(nx.strongly_connected_components(G), key=len))  # delete unconnected subgraph
        self._setup_graph(G)
 