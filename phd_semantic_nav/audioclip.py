import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import threading
import queue
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
import torch
import os
import sys
from ament_index_python.packages import get_package_share_directory
from phd_msgs.msg import OdasOut
# import soundfile as sf
from phd_msgs.msg import WorldObject, WorldObjectArray, VisualObservation, VisualObservationArray

# audioclip_path = os.path.join(get_package_share_directory('phd_semantic_nav'), 'AudioCLIP') # for production
# audioclip_path = '/home/matthew/dev/ros_ws/src/phd_semantic_nav/AudioCLIP'
audioclip_path = '/ros2_ws/src/phd_semantic_nav/AudioCLIP'
if audioclip_path not in sys.path:
    sys.path.append(audioclip_path)

from model import AudioCLIP
from utils.transforms import ToTensor1D

from sklearn.cluster import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import glob
from collections import defaultdict, deque
from visualization_msgs.msg import Marker, MarkerArray
from itertools import combinations
from geometry_msgs.msg import Point

torch.set_grad_enabled(False)
# torch.set_num_threads(2)  # Give PyTorch 2 dedicated cores for efficient matrix batching
# torch.set_num_interop_threads(1)

class AudioCLIPSub(Node):
    def __init__(self):
        super().__init__('audioclip')

        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        world_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        observation_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.batch_size = 1
        self.audioclip_q = queue.Queue(maxsize=100)

        self.aclp = AudioCLIP(pretrained=f'{audioclip_path}/assets/AudioCLIP-Full-Training.pt')
        self.aclp.eval()
        self.audio_transforms = ToTensor1D()

        # self.tf_buffer = tf2_ros.Buffer()
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

        # self.position_clustering = HDBSCAN(min_cluster_size=2, metric='euclidean')
        # self.semantic_clustering = HDBSCAN(min_cluster_size=2, metric='cosine')

        self.semantic_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.1,
            metric='cosine',
            linkage='average' # or 'complete'
        )

        self.position_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0,    # in meters, i should tune with real scenario examples
            metric='euclidean',
            linkage='ward'
        )

        # self.LABELS = [
        #     'sound of human speech and talking',
        #     'sound of footsteps on the floor',
        #     'sound of a door opening or closing',
        #     # 'sound of a person walking',
        #     # 'sound of a person screaming in fear',
        #     # 'sound of a person shouting loudly',
        #     'sound of a car engine running',
        #     'sound of an animal like a dog or cat',
        #     'sound of an alarm or siren beeping',
        #     'sound of a crowd of people',
        #     'sound of silence or background noise'
        # ]
        self.LABELS = [
            'human talking',
            # 'footsteps',
            # 'door open or close',
            # 'car engine',
            # 'cat',
            'alarm clock'
        ]

        self.visual_graph_nodes = {}  # 3DSG-like structure (only nodes for now)
        self.visual_graph_edges = {}
        self.visual_area_nodes = {}
        self.visual_area_edges = {}
        self.object_to_area = {}
        self.next_area_id = 0

        self.audio_graph_nodes = {}
        self.audio_node_id = 0
        self.audio_obs_id = 0

        self.obs_cache = defaultdict(lambda: deque(maxlen=20)) # cache memory for recent visual observation messages
        self.wo_lock = threading.Lock()

        text = [[label] for label in self.LABELS]
        with torch.no_grad():
            ((_, _, self.text_features), _), _ = self.aclp(text=text)
        self.text_features = self.text_features / torch.linalg.norm(self.text_features, dim=-1, keepdim=True)
        self.scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(), min=1.0, max=100.0)

        # test audio
        # self.test_audio = list()
        # for path_to_audio in glob.glob('/ros2_ws/src/phd_semantic_nav/AudioCLIP/demo/audio/*.wav'):
        #     track, _ = librosa.load(path_to_audio, sr=44100, dtype=np.float32)
        #     self.test_audio.append(track)
        
        # self.test_audio = torch.stack([self.audio_transforms(track.reshape(1, -1)) for track in self.test_audio])
        # ((self.test_audio_features, _, _), _), _ = self.aclp(audio=self.test_audio)
        # self.test_audio_features = self.test_audio_features / torch.linalg.norm(self.test_audio_features, dim=-1, keepdim=True)
        # test audio end

        self.sss_pub = self.create_publisher(MarkerArray, '/audio_graph/audioclip_cls', audio_qos)
        self.audio_graph_viz_pub = self.create_publisher(MarkerArray, "/audio_graph/node_markers", 10)

        self.sss_sub = self.create_subscription(OdasOut, '/audioclip_sss', self.sss_callback, audio_qos)
        self.sss_sub  # prevent unused variable warning

        self.world_objects_sub = self.create_subscription(WorldObjectArray, '/world_objects', self.wo_callback, world_qos)
        self.world_objects_sub  # prevent unused variable warning

        self.obs_sub = self.create_subscription(VisualObservationArray, '/object_observations', self.vo_callback, observation_qos)
        self.obs_sub  # prevent unused variable warning

        self.visual_graph_edge_viz_pub = self.create_publisher(Marker, '/camera_pipeline/edge_markers', 10)

        self.thread = threading.Thread(target=self.inference_thread, daemon=True)
        self.thread.start()

    def wo_callback(self, msg):
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.wo_lock:
            for world_object in msg.objects:
                object_id = int(world_object.world_object_id)
                if object_id not in self.visual_graph_nodes.keys():
                    self.visual_graph_nodes[object_id] = {
                        'world_object_id': object_id,
                        'first_seen': world_object.first_seen,
                        'state_history': deque(maxlen=100)
                    }

                self.visual_graph_nodes[object_id]['timestamp_ns'] = timestamp_ns
                self.visual_graph_nodes[object_id]['label'] = str(world_object.label)
                self.visual_graph_nodes[object_id]['label_confidence'] = world_object.label_confidence
                self.visual_graph_nodes[object_id]['position_map'] = np.array([world_object.position.x, world_object.position.y, world_object.position.z])
                self.visual_graph_nodes[object_id]['velocity_map'] = np.array([world_object.velocity.x, world_object.velocity.y, world_object.velocity.z])
                self.visual_graph_nodes[object_id]['radius_m'] = world_object.radius_m
                self.visual_graph_nodes[object_id]['position_uncertainty_m'] = world_object.position_uncertainty_m
                self.visual_graph_nodes[object_id]['last_seen'] = world_object.last_seen
                self.visual_graph_nodes[object_id]['active'] = world_object.active
                self.visual_graph_nodes[object_id]['observation_count'] = world_object.observation_count
                self.visual_graph_nodes[object_id]['vision_track_ids'] = sorted(int(t_id) for t_id in world_object.vision_track_ids)

                last_seen_ns = world_object.last_seen.sec * 1_000_000_000 + world_object.last_seen.nanosec
                if not self.visual_graph_nodes[object_id]['state_history'] or self.visual_graph_nodes[object_id]['state_history'][-1]['timestamp_ns'] != last_seen_ns:
                    state = {
                        'timestamp_ns': last_seen_ns,
                        'position_map': np.array([world_object.position.x, world_object.position.y, world_object.position.z], dtype=np.float64),
                        'velocity_map': np.array([world_object.velocity.x, world_object.velocity.y, world_object.velocity.z], dtype=np.float64),
                        'radius_m': float(world_object.radius_m),
                        'position_uncertainty_m': float(world_object.position_uncertainty_m),
                        'active': bool(world_object.active),
                    }

                    self.visual_graph_nodes[object_id]['state_history'].append(state)

            active_objects = {object_id: node for object_id, node in self.visual_graph_nodes.items() if node.get("active", False)}

            # create edges
            current_near_edges = set()
            for (id_a, object_a), (id_b, object_b) in combinations(active_objects.items(), 2):
                position_a = np.asarray(object_a["position_map"], dtype=np.float64)
                position_b = np.asarray(object_b["position_map"], dtype=np.float64)

                distance_m = float(np.linalg.norm(position_a - position_b))
                proximity_threshold = 1.0 + float(object_a["radius_m"]) + float(object_b["radius_m"]) + float(object_a["position_uncertainty_m"]) + float(object_b["position_uncertainty_m"])
                edge_key = (min(id_a, id_b), max(id_a, id_b), 'nearest')

                if distance_m <= proximity_threshold:
                    current_near_edges.add(edge_key)

                    if edge_key not in self.visual_graph_edges:
                        self.visual_graph_edges[edge_key] = {
                            "source_id": edge_key[0],
                            "target_id": edge_key[1],
                            "type": "nearest",
                            "distance_m": distance_m,
                            "first_seen_ns": timestamp_ns,
                            "last_seen_ns": timestamp_ns,
                            "observation_count": 1,
                            "active": True
                        }
                    else:
                        edge = self.visual_graph_edges[edge_key]
                        edge["distance_m"] = distance_m
                        edge["last_seen_ns"] = timestamp_ns
                        edge["observation_count"] += 1
                        edge["active"] = True

            # Deactivate nearest edges that are no longer present.
            for edge_key, edge in self.visual_graph_edges.items():
                if edge["type"] != "nearest":
                    continue

                if edge_key not in current_near_edges:
                    edge["active"] = False

            # create area edges
            del active_objects
            active_object_ids = {object_id for object_id, node in self.visual_graph_nodes.items() if node.get("active", False)}

            # Every active object starts as an isolated graph node.
            adjacency = {object_id: set() for object_id in active_object_ids}

            # Add currently active nearest relationships.
            for edge in self.visual_graph_edges.values():
                if edge.get("type") != "nearest": continue
                if not edge.get("active", False): continue

                source_id = edge["source_id"]
                target_id = edge["target_id"]

                if source_id not in active_object_ids or target_id not in active_object_ids:
                    continue

                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

            # Find connected components.
            components = []
            unvisited = set(active_object_ids)

            while unvisited:
                start_id = unvisited.pop()
                component = {start_id}
                stack = [start_id]

                while stack:
                    current_id = stack.pop()

                    for neighbour_id in adjacency[current_id]:
                        if neighbour_id not in unvisited:
                            continue

                        unvisited.remove(neighbour_id)
                        component.add(neighbour_id)
                        stack.append(neighbour_id)

                components.append(component)

            # Larger components are matched first. This helps preserve
            # stable area IDs when an area splits.
            components.sort(key=len, reverse=True)
            previous_areas = {area_id: set(area["member_ids"]) for area_id, area in self.visual_area_nodes.items() if area.get("active", False)}

            # Match new components to old areas according to how many
            # object IDs they have in common.
            match_candidates = []

            for component_index, component in enumerate(components):
                for area_id, old_members in previous_areas.items():
                    overlap = len(component.intersection(old_members))

                    if overlap > 0:
                        match_candidates.append((overlap, component_index, area_id))

            match_candidates.sort(reverse=True)

            component_to_area = {}
            used_area_ids = set()

            for overlap, component_index, area_id in match_candidates:
                if component_index in component_to_area:
                    continue

                if area_id in used_area_ids:
                    continue

                component_to_area[component_index] = area_id
                used_area_ids.add(area_id)

            # Deactivate old areas. Matched areas are reactivated below.
            for area in self.visual_area_nodes.values():
                area["active"] = False

            self.object_to_area.clear()

            for area_edge in self.visual_area_edges.values():
                area_edge["active"] = False

            for component_index, member_ids in enumerate(components):
                if component_index in component_to_area:
                    area_id = component_to_area[component_index]
                else:
                    area_id = self.next_area_id
                    self.next_area_id += 1

                positions = np.array([self.visual_graph_nodes[object_id]["position_map"] for object_id in member_ids], dtype=np.float64)
                centroid_map = np.mean(positions, axis=0)

                if area_id not in self.visual_area_nodes:
                    self.visual_area_nodes[area_id] = {
                        "area_id": area_id,
                        "member_ids": set(),
                        "centroid_map": centroid_map,
                        "first_seen_ns": timestamp_ns,
                        "last_updated_ns": timestamp_ns,
                        "active": True,
                    }

                area = self.visual_area_nodes[area_id]

                area["member_ids"] = set(member_ids)
                area["centroid_map"] = centroid_map
                area["last_updated_ns"] = timestamp_ns
                area["active"] = True

                for object_id in member_ids:
                    self.object_to_area[object_id] = area_id
                    self.visual_graph_nodes[object_id]["area_id"] = area_id

                    area_edge_key = (area_id, object_id, "area")

                    if area_edge_key not in self.visual_area_edges:
                        self.visual_area_edges[area_edge_key] = {
                            "area_id": area_id,
                            "object_id": object_id,
                            "type": "area",
                            "first_seen_ns": timestamp_ns,
                            "last_updated_ns": timestamp_ns,
                            "active": True,
                        }
                    else:
                        area_edge = self.visual_area_edges[area_edge_key]
                        area_edge["last_updated_ns"] = timestamp_ns
                        area_edge["active"] = True

            # visualize nearest object edges
            marker = Marker()

            marker.header.stamp = msg.header.stamp
            marker.header.frame_id = msg.header.frame_id

            marker.ns = "visual_graph_edges"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            # Width of each line in metres.
            marker.scale.x = 0.025

            # Light cyan edge colour.
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 0.9

            marker.pose.orientation.w = 1.0

            for edge in self.visual_graph_edges.values():
                if edge["type"] != "nearest":
                    continue

                if not edge["active"]:
                    continue

                source_id = edge["source_id"]
                target_id = edge["target_id"]

                source_node = self.visual_graph_nodes.get(source_id)
                target_node = self.visual_graph_nodes.get(target_id)

                if source_node is None or target_node is None: continue
                if not source_node.get("active", False): continue
                if not target_node.get("active", False): continue

                source_position = source_node["position_map"]
                target_position = target_node["position_map"]

                source_point = Point()
                source_point.x = float(source_position[0])
                source_point.y = float(source_position[1])
                source_point.z = float(source_position[2])

                target_point = Point()
                target_point.x = float(target_position[0])
                target_point.y = float(target_position[1])
                target_point.z = float(target_position[2])

                marker.points.append(source_point)
                marker.points.append(target_point)

            if not marker.points:
                # Remove the previously displayed edge marker when no active edges remain
                marker.action = Marker.DELETE

            # visualize area nodes
            area_node_marker = Marker()

            area_node_marker.header = msg.header
            area_node_marker.ns = "visual_area_nodes"
            area_node_marker.id = 0
            area_node_marker.type = Marker.SPHERE_LIST
            area_node_marker.action = Marker.ADD
            area_node_marker.pose.orientation.w = 1.0

            area_node_marker.scale.x = 0.35
            area_node_marker.scale.y = 0.35
            area_node_marker.scale.z = 0.35

            area_node_marker.color.r = 1.0
            area_node_marker.color.g = 0.2
            area_node_marker.color.b = 0.2
            area_node_marker.color.a = 1.0

            for area in self.visual_area_nodes.values():
                if not area["active"]:
                    continue

                position = area["centroid_map"]

                point = Point()
                point.x = float(position[0])
                point.y = float(position[1])
                point.z = float(position[2] + 1.5)

                area_node_marker.points.append(point)

            if not area_node_marker.points:
                area_node_marker.action = Marker.DELETE

            # visualize area egdes
            area_edge_marker = Marker()

            area_edge_marker.header = msg.header
            area_edge_marker.ns = "visual_area_edges"
            area_edge_marker.id = 0
            area_edge_marker.type = Marker.LINE_LIST
            area_edge_marker.action = Marker.ADD
            area_edge_marker.pose.orientation.w = 1.0

            area_edge_marker.scale.x = 0.04

            # Orange area-membership edges.
            area_edge_marker.color.r = 1.0
            area_edge_marker.color.g = 0.5
            area_edge_marker.color.b = 0.1
            area_edge_marker.color.a = 0.9

            for area_edge in self.visual_area_edges.values():
                if not area_edge["active"]:
                    continue

                area = self.visual_area_nodes.get(area_edge["area_id"])
                object_node = self.visual_graph_nodes.get(area_edge["object_id"])

                if area is None or object_node is None: continue
                if not area["active"]: continue
                if not object_node.get("active", False): continue

                area_position = area["centroid_map"]
                object_position = object_node["position_map"]

                area_point = Point()
                area_point.x = float(area_position[0])
                area_point.y = float(area_position[1])
                area_point.z = float(area_position[2] + 1.5)

                object_point = Point()
                object_point.x = float(object_position[0])
                object_point.y = float(object_position[1])
                object_point.z = float(object_position[2])

                area_edge_marker.points.append(area_point)
                area_edge_marker.points.append(object_point)

            if not area_edge_marker.points:
                area_edge_marker.action = Marker.DELETE

        self.visual_graph_edge_viz_pub.publish(marker)
        self.visual_graph_edge_viz_pub.publish(area_edge_marker)
        self.visual_graph_edge_viz_pub.publish(area_node_marker)

    def vo_callback(self, msg):
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        for obs in msg.observations:
            object_id = int(obs.world_object_id)

            self.obs_cache[object_id].append({
                'timestamp_ns': timestamp_ns,
                'has_track_id': obs.has_track_id,
                'track_id': int(obs.track_id),
                'label': obs.label,
                'class_id': obs.class_id,
                'confidence': obs.confidence,
                'bbox_xyxy': list(obs.bbox_xyxy),
                'image_width': obs.image_width,
                'image_height': obs.image_height,
                'crop': obs.crop
            })
    
    def sss_callback(self, msg):
        self.get_logger().info(f'Placing in queue..')
        # data = np.array(msg.data.data, dtype=np.float32).reshape((msg.data.layout.dim[0].size, msg.data.layout.dim[1].size))
        data_list = []
        metadata_list = []
        audio_observations = []

        # associate world object (vision) with sound source direction (audio)
        for odas_sss_msg in msg.sources:
            audio_clip_candidates = {}
            audio = np.array(odas_sss_msg.data, dtype=np.float32)
            data_list.append(audio)
            # metadata_list.append((odas_sss_msg.start, odas_sss_msg.end, odas_sss_msg.id, \
            #                       odas_sss_msg.channel_idx, odas_sss_msg.sample_rate, odas_sss_msg.frame_sample_count, odas_sss_msg.sst))

            source_directions = []
            for sst in odas_sss_msg.sst:
                direction = np.array([sst.x, sst.y, sst.z], dtype=np.float64)
                norm = np.linalg.norm(direction)
                if norm < 1e-9: continue
                direction /= norm

                source_directions.append({
                    'timestamp_ns': int(sst.stamp.sec) * 1_000_000_000 + sst.stamp.nanosec,
                    'direction_mic': direction,
                    'activity': float(sst.activity)
                })

            audio_observations.append({
                'audio_obs_id': self.audio_obs_id,
                'odas_source_id': int(odas_sss_msg.id),
                'channel_idx': int(odas_sss_msg.channel_idx),
                'start_ns': int(odas_sss_msg.start.sec) * 1_000_000_000 + int(odas_sss_msg.start.nanosec),
                'end_ns': int(odas_sss_msg.end.sec) * 1_000_000_000 + int(odas_sss_msg.end.nanosec),
                'sample_rate': int(odas_sss_msg.sample_rate),
                'duration_sec': float(audio.size) / float(odas_sss_msg.sample_rate),
                'audio': audio.copy(),
                'odas_directions': source_directions,
                'audioclip_embeddings': None,
                'label': '',
                'label_confidence': None
            })

            self.audio_obs_id += 1

            # for direction in odas_sss_msg.sst:
            #     try:
            #         mic_to_map = self.tf_buffer.lookup_transform(
            #             'walp/map',
            #             'box_mic',
            #             Time.from_msg(direction.stamp),
            #             timeout=Duration(seconds=0.1)
            #         )
            #     except tf2_ros.TransformException as e:
            #         self.get_logger().warning(f"Cannot transform box_mic into walp/map: {e}")
            #         continue

            #     translation = mic_to_map.transform.translation
            #     rotation_matrix = self.quaternion_to_rotation_matrix(mic_to_map.transform.rotation)

            #     mic_direction = np.array([direction.x, direction.y, direction.z], dtype=np.float64)
            #     direction_norm = np.linalg.norm(mic_direction)
            #     if direction_norm < 1e-9: continue
            #     mic_direction /= direction_norm

            #     mic_position_map = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
            #     direction_map = rotation_matrix @ mic_direction
            #     direction_map /= np.linalg.norm(direction_map)

            #     tstamp_ns = direction.stamp.sec * 1_000_000_000 + direction.stamp.nanosec
            #     with self.wo_lock:
            #         world_object_ids = list(self.visual_graph_nodes.keys())

            #     sst_candidate_objs = []

            #     for object_id in world_object_ids:
            #         state_history = None
            #         with self.wo_lock:
            #             node = self.visual_graph_nodes[object_id]
            #             if node is not None: state_history = node['state_history']

            #         state = None
            #         if state_history:
            #             previous_states = [pstate for pstate in state_history if pstate['timestamp_ns'] <= tstamp_ns]
            #             if previous_states:
            #                 state = max(previous_states, key=lambda item: item['timestamp_ns'])
            #             else:
            #                 nearest_state = min(state_history, key=lambda item: item['timestamp_ns'])
            #                 time_diff = nearest_state['timestamp_ns'] - tstamp_ns

            #                 if time_diff >= 0 and time_diff <= 100_000_000:
            #                     state = nearest_state

            #         if state is None: continue
                    
            #         mic_to_object = state['position_map'] - mic_position_map
            #         forward_distance = float(np.dot(mic_to_object, direction_map))    # projection of sound source to graph object along the sst direction
                    
            #         if forward_distance <= 0.0: continue  # if the dot product is <= 0, it lies in the opposite direction of the sst line
                    
            #         closest_point_map = mic_position_map + forward_distance * direction_map
            #         perpendicular_distance = float(np.linalg.norm(state['position_map'] - closest_point_map))
            #         cone_radius = forward_distance * np.tan(np.deg2rad(15.0)) # 15 degrees angles of cone search radius
            #         allowed_distance = cone_radius + float(state["radius_m"]) + float(state["uncertainty_m"])  # include visual position uncertainty

            #         # cone search
            #         if perpendicular_distance <= allowed_distance:
            #             sst_candidate_objs.append({
            #                 'id': direction.id,
            #                 'timestamp_ns': tstamp_ns,
            #                 'world_object_id': object_id,
            #                 'forward_distance_m': forward_distance,
            #                 'perpendicular_distance_m': perpendicular_distance,
            #                 'cone_radius_m': cone_radius,
            #                 'allowed_distance_m': allowed_distance
            #             })

        # proceed with semantic similarity
        data = np.stack(data_list, axis=0, dtype=np.float32)
        rms = np.sqrt(np.mean(data**2))
        if rms < 1e-6:
            self.get_logger().info(f"\033[91mDiscarding...\033[0m")
            return

        # per-chunk normalization
        # data = data / rms * 0.1
        # self.get_logger().info(f"\033[95mReceived min: {data.min():.6f}, max: {data.max():.6f}\033[0m")
        # self.get_logger().info(f"\033[95mReceived mean: {data.mean():.6f}\033[0m")
        # self.get_logger().info(f"\033[95mReceived std: {data.std():.6f}\033[0m")
        # self.get_logger().info(f"\033[95mReceived RMS: {np.sqrt(np.mean(data**2)):.6f}\033[0m")
        # self.get_logger().info(f"\033[95mReceived vals: {data[:, :5]}\033[0m")
        # self.audioclip_q.put((msg.header, data, metadata_list))
        self.audioclip_q.put((msg.header, data, audio_observations))
        # for i in range(data.shape[0]):
        #     source = data[i].reshape(1, -1)
        #     self.audioclip_q.put((source, msg.sst, msg.i, msg.stride))
    
    def _evaluate_node(self, node, observation):
        # calculate time difference between observation and node
        elapsed_time = (int(observation["start_ns"]) - int(node["last_seen_ns"])) / 1e9 # in seconds

        # ignore observations that hugely overlap the node or much older observations
        if elapsed_time < -0.1:
            self.get_logger().info(
                f"Reject node {node['audio_node_id']}: "
                f"overlap={elapsed_time:.3f}s"
            )
            return None
        if elapsed_time > 10.0:
            self.get_logger().info(
                f"Reject node {node['audio_node_id']}: "
                f"gap={elapsed_time:.3f}s"
            )
            return None

        elapsed_time = max(0.0, elapsed_time)
        time_cost = min(elapsed_time / 10.0, 1.0)   # time cost inside [0.0, 1.0]

        # calculate cosine similarity between node and observation to identify graph existence
        observation_norm = float(np.linalg.norm(observation['audioclip_embeddings']))
        node_norm = float(np.linalg.norm(node["audioclip_embeddings"]))

        if observation_norm < 1e-9 or node_norm < 1e-9:
            return None

        embedding_similarity = float(np.dot(observation['audioclip_embeddings'], node["audioclip_embeddings"]) / (observation_norm * node_norm))
        embedding_similarity = float(np.clip(embedding_similarity, -1.0, 1.0))  # cosine similarity inside [-1.0, 1.0]

        embedding_cost = (1.0 - embedding_similarity) / 2.0 # embedding cost inside [0.0, 1.0]

        # label match
        labels_match = observation["label"] == node["label"]
        label_cost = 0.0 if labels_match else 1.0

        # weighted total cost
        cost = 0.7 * embedding_cost + 0.20 * time_cost + 0.10 * label_cost

        return max(0.0, float(cost))

    def create_new_node(self, observation):
        accumulated_observations = deque(maxlen=5)
        accumulated_observations.append(observation)

        self.audio_graph_nodes[self.audio_node_id] = {
            'audio_node_id': self.audio_node_id,
            'audio_obs_id': observation['audio_obs_id'],
            'odas_source_ids': {observation['odas_source_id']},
            'latest_odas_source_id': observation['odas_source_id'],
            'latest_channel_idx': observation['channel_idx'],
            'first_seen_ns': observation['start_ns'],
            'last_seen_ns': observation['end_ns'],
            'sample_rate': observation['sample_rate'],
            'duration_sec': observation['duration_sec'],
            'odas_directions': observation['odas_directions'].copy(),
            'audio': observation['audio'].copy(),
            'audioclip_embeddings': observation['audioclip_embeddings'].copy(),
            'label': str(observation['label']),
            'label_confidence': observation['label_confidence'],
            'active': True,
            'past_observations': accumulated_observations
        }

        self.audio_node_id += 1

    def mark_audio_node_activity(self, current_audio_time_ns):
        INACTIVITY_TIMEOUT = 6.0
        DELETION_TIMEOUT = 15.0

        inactive_timeout_ns = int(INACTIVITY_TIMEOUT * 1e9)
        deletion_timeout_ns = int(DELETION_TIMEOUT * 1e9)
        nodes_to_delete = []

        for audio_node_id, node in self.audio_graph_nodes.items():
            time_since_last_seen_ns = max(0, int(current_audio_time_ns) - int(node["last_seen_ns"]))
            node["active"] = time_since_last_seen_ns <= inactive_timeout_ns

            # if time_since_last_seen_ns > deletion_timeout_ns:
            #     nodes_to_delete.append(audio_node_id)

        for audio_node_id in nodes_to_delete:
            del self.audio_graph_nodes[audio_node_id]
    
    def inference_thread(self):
        while rclpy.ok():
            minibatch = []
            sstlist = []

            self.get_logger().info(f'Trying to get from queue...')
            sss_header, data, audio_observations = self.audioclip_q.get()
            audio = self.audio_transforms(data)
            
            try:
                self.get_logger().info(f'Starting inference...')
                # audio = torch.stack(minibatch)
                # audio = audio.squeeze(dim=0)
                audio = audio.unsqueeze(dim=1)
                self.get_logger().info(f'Size: {audio.size()}')
                # compare with test audio
                # if audio.size(0) == 1:
                #     test_similarity = cosine_similarity(self.test_audio.numpy()[0,:,:44100], audio.numpy()[0])
                #     self.get_logger().info(f"\033[91mTest similarity input: {test_similarity}\033[0m")
                with torch.no_grad():
                    ((audio_features, _, _), _), _ = self.aclp(audio=audio)
                audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
                logits_audio_text = self.scale_audio_text * audio_features @ self.text_features.T

                # compare with test audio embeddings
                # if audio_features.size(0) == 1:
                #     test_similarity = cosine_similarity(self.test_audio_features.numpy(), audio_features.numpy())
                #     self.get_logger().info(f"\033[91mTest similarity embeddings: {test_similarity}\033[0m")

                # self.get_logger().info(f"Audio embeddings shape: {audio_features.size()}")
                # self.get_logger().info(f"\033[95mEmbedding vals: {audio_features.numpy()[:,:5]}\033[0m")
                # self.get_logger().info(f"\033[95mEmbedding std: {audio_features.numpy().std():.6f}\033[0m")
                # self.get_logger().info(f"\033[95mEmbedding min: {audio_features.numpy().min():.6f}, max: {audio_features.numpy().max():.6f}\033[0m")

                # Zero-shot classification
                raw_cosine = audio_features @ self.text_features.T
                self.get_logger().info(f"Raw cosine similarities: {raw_cosine}")
                confidence = logits_audio_text.softmax(dim=1)
                source_counter = 0
                marker_array = MarkerArray()
                offset_text = 0.0
                assigned_node_ids = []
                batch_visited_nodes = set()
                for audio_idx in range(audio_features.size(0)):
                    # get top most similar results
                    conf_values, ids = confidence[audio_idx].topk(audio_features.size(0))

                    # format output strings
                    query = f'{audio_idx} ->\t'
                    results = ', '.join([f'{self.LABELS[i]:>5s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

                    # update audio info
                    audio_observations[audio_idx]['audioclip_embeddings'] = audio_features[audio_idx].detach().cpu().numpy().astype(np.float32)
                    audio_observations[audio_idx]['label'] = str(self.LABELS[ids[0]])
                    audio_observations[audio_idx]['label_confidence'] = float(conf_values[0])

                    # create or update audio graph node
                    best_node_id = None
                    # best_cost = float('inf')
                    best_cost = 0.3

                    candidate_costs = {}
                    rejected_node_ids = []
                    excluded_node_ids = []
                    for audio_node_id, node in self.audio_graph_nodes.items():
                        if audio_node_id in batch_visited_nodes:
                            excluded_node_ids.append(audio_node_id)
                            continue

                        # check if the audio observation matches any saved node with certain criteria
                        cost = self._evaluate_node(node, audio_observations[audio_idx])

                        if cost is None:
                            rejected_node_ids.append(audio_node_id)
                            continue
                        candidate_costs[audio_node_id] = cost
                        if cost < best_cost:
                            best_cost = cost
                            best_node_id = audio_node_id

                    audio_node_id = -1
                    if best_node_id is None:
                        # create new node
                        audio_node_id = int(self.audio_node_id)
                        self.create_new_node(audio_observations[audio_idx])

                        self.get_logger().warning(
                            f"Creating new audio node: "
                            f"obs={audio_observations[audio_idx]['audio_obs_id']}, "
                            f"label={audio_observations[audio_idx]['label']}, "
                            f"ODAS={audio_observations[audio_idx]['odas_source_id']}, "
                            f"candidate_costs={candidate_costs}, "
                            f"rejected={rejected_node_ids}, "
                            f"batch_excluded={excluded_node_ids}, "
                            f"new_node_cost={best_cost:.3f}"
                        )
                    else:
                        # update existing node
                        audio_node_id = int(best_node_id)
                        node = self.audio_graph_nodes[audio_node_id]

                        node['audio_obs_id'] = int(audio_observations[audio_idx]['audio_obs_id'])
                        node['odas_source_ids'].add(audio_observations[audio_idx]['odas_source_id'])
                        node['latest_odas_source_id'] = audio_observations[audio_idx]['odas_source_id']
                        node['latest_channel_idx'] = audio_observations[audio_idx]['channel_idx']
                        node['last_seen_ns'] = audio_observations[audio_idx]['end_ns']
                        node['sample_rate'] = audio_observations[audio_idx]['sample_rate']
                        node['duration_sec'] = audio_observations[audio_idx]['duration_sec']
                        node['odas_directions'] = audio_observations[audio_idx]['odas_directions'].copy()
                        node['audio'] = audio_observations[audio_idx]['audio'].copy()
                        node['audioclip_embeddings'] = audio_observations[audio_idx]['audioclip_embeddings'].copy()
                        node['label'] = str(audio_observations[audio_idx]['label'])
                        node['label_confidence'] = audio_observations[audio_idx]['label_confidence']
                        node['active'] = True
                        node['past_observations'].append(audio_observations[audio_idx])

                    audio_observations[audio_idx]['audio_node_id'] = int(audio_node_id)
                    assigned_node_ids.append(audio_node_id)
                    batch_visited_nodes.add(audio_node_id)

                    # TODO: associate audio nodes to visual nodes

                    # visualize classification results (temporarily before creating and visualizing audio graph nodes)
                    # text_marker = Marker()
                    # text_marker.header = sss_header
                    # text_marker.ns = "audioclip_class_text"
                    # text_marker.id = audio_idx
                    # text_marker.type = Marker.TEXT_VIEW_FACING
                    # text_marker.action = Marker.ADD

                    # text_marker.pose.position.x = float(offset_text)
                    # text_marker.pose.position.y = float(offset_text)
                    # text_marker.pose.position.z = float(offset_text)
                    # text_marker.pose.orientation.w = 1.0
                    # offset_text -= 0.1

                    # text_marker.scale.z = 0.11

                    # text_marker.color.r = 0.318
                    # text_marker.color.g = 0.749
                    # text_marker.color.b = 0.212
                    # text_marker.color.a = 1.0
                    # text_marker.text = f'{query + results}'
                    # marker_array.markers.append(text_marker)

                    # sf.write(f'/ros2_ws/src/phd_semantic_nav/phd_semantic_nav/audio_out_samples/out_{nanosec}_{source_counter}.wav', audio[audio_idx, 0].numpy(), 44100)
                    # avg_activity = 0.0
                    # for element in sstlist[0][0][source_counter:source_counter + sstlist[0][2][audio_idx]]:
                    #     avg_activity += element.activity
                    # avg_activity /= len(sstlist[0][0][source_counter:source_counter + sstlist[0][2][audio_idx]])
                    # self.get_logger().info(f"Average activity: {avg_activity}")
                    # self.get_logger().info(f"Sstlist len: {len(sstlist[0][0])}")
                    # self.get_logger().info(f"Sstlist i: {sstlist[0][1]}")
                    # self.get_logger().info(f"Sstlist stride: {sstlist[0][2]}")

                    self.get_logger().info(f"\033[95m{query + results}\033[0m")
                    source_counter += 1
                    # source_counter += sstlist[0][2][audio_idx]
                
                # check if sound sources are still active
                if audio_observations:
                    current_audio_time_ns = max(int(observation["end_ns"]) for observation in audio_observations)
                    self.mark_audio_node_activity(current_audio_time_ns)

                # visualize audio graph nodes (for now, only for a fixed position in map along the sst direction)
                self.get_logger().info(f"Audio nodes: {len(list(self.audio_graph_nodes.values()))}")
                audio_graph_marker_array = MarkerArray()

                # clear the previous audio graph visualization
                delete_marker = Marker()
                delete_marker.header.stamp = sss_header.stamp
                delete_marker.header.frame_id = "walp/map"
                delete_marker.action = Marker.DELETEALL
                audio_graph_marker_array.markers.append(delete_marker)

                for node in self.audio_graph_nodes.values():
                    if not node.get("active", False): continue

                    directions = node.get('odas_directions', [])
                    if not directions: continue

                    # use midpoint of the direction list
                    direction_state = directions[len(directions) // 2]

                    timestamp_ns = int(direction_state["timestamp_ns"])
                    direction_mic = np.asarray(direction_state["direction_mic"], dtype=np.float64).copy()
                    direction_norm = float(np.linalg.norm(direction_mic))

                    if direction_norm < 1e-9: continue
                    direction_mic /= direction_norm

                    try:
                        mic_to_map = self.tf_buffer.lookup_transform(
                            "walp/map",
                            "box_mic",
                            Time(nanoseconds=timestamp_ns),
                            timeout=Duration(seconds=0.1)
                        )
                    except tf2_ros.TransformException as error:
                        self.get_logger().warning(f"Cannot visualize audio node {node['audio_node_id']}: {error}")
                        continue

                    translation = mic_to_map.transform.translation
                    mic_position_map = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
                    rotation_matrix = self.quaternion_to_rotation_matrix(mic_to_map.transform.rotation)

                    # rotate to map frame
                    direction_map = rotation_matrix @ direction_mic
                    map_direction_norm = float(np.linalg.norm(direction_map))

                    if map_direction_norm < 1e-9: continue

                    direction_map /= map_direction_norm

                    # fixed marker position since audio not yet matched with a visual object
                    display_position_map = mic_position_map + direction_map
                    audio_node_id = int(node["audio_node_id"])

                    # audio node marker
                    node_marker = Marker()
                    node_marker.header.stamp = sss_header.stamp
                    node_marker.header.frame_id = "walp/map"
                    node_marker.ns = "audio_graph_nodes"
                    node_marker.id = audio_node_id
                    node_marker.type = Marker.SPHERE
                    node_marker.action = Marker.ADD

                    node_marker.pose.position.x = float(display_position_map[0])
                    node_marker.pose.position.y = float(display_position_map[1])
                    node_marker.pose.position.z = float(display_position_map[2] + 1.5)
                    node_marker.pose.orientation.w = 1.0

                    node_marker.scale.x = 0.30
                    node_marker.scale.y = 0.30
                    node_marker.scale.z = 0.30

                    node_marker.color.r = 0.95
                    node_marker.color.g = 0.20
                    node_marker.color.b = 0.75
                    node_marker.color.a = 1.0

                    audio_graph_marker_array.markers.append(node_marker)

                    # audio node text
                    text_marker = Marker()
                    text_marker.header.stamp = sss_header.stamp
                    text_marker.header.frame_id = "walp/map"
                    text_marker.ns = "audio_graph_node_text"
                    text_marker.id = audio_node_id
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD

                    text_marker.pose.position.x = float(display_position_map[0])
                    text_marker.pose.position.y = float(display_position_map[1])
                    text_marker.pose.position.z = float(display_position_map[2] + 1.5 + 0.3)
                    text_marker.pose.orientation.w = 1.0

                    text_marker.scale.z = 0.16

                    text_marker.color.r = 0.318
                    text_marker.color.g = 0.749
                    text_marker.color.b = 0.212
                    text_marker.color.a = 1.0

                    source_ids_text = ",".join(str(source_id) for source_id in sorted(node["odas_source_ids"]))

                    text_marker.text = (
                        f"{node['label']}\n"
                        f"{float(node['label_confidence']):.2f}\n"
                        f"ODAS IDs={source_ids_text}"
                    )

                    audio_graph_marker_array.markers.append(text_marker)

                self.audio_graph_viz_pub.publish(audio_graph_marker_array)
                # self.sss_pub.publish(marker_array)

                # if audio_features.size(0) > 1:
                #     distances = cosine_similarity(audio_features.numpy())
                #     self.get_logger().info(f"\033[95mDistances: {distances}\033[0m")
                    # Clustering
                    # semantic_clusters = self.semantic_clustering.fit_predict(audio_features.numpy())
                    # if not np.all(semantic_clusters == -1):
                    #     self.get_logger().info(f"\033[92mSemantic clusters: {semantic_clusters}\033[0m")
                    # else:
                    #     self.get_logger().info(f"\033[91mSemantic clusters: {semantic_clusters}\033[0m")
            except Exception as e:
                self.get_logger().error(f"Inference error: {str(e)}")
            finally:
                for _ in range(self.batch_size):
                    self.audioclip_q.task_done()

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        x = float(quaternion.x)
        y = float(quaternion.y)
        z = float(quaternion.z)
        w = float(quaternion.w)

        norm = np.sqrt(x * x + y * y + z * z + w * w)

        if norm < 1e-12:
            raise ValueError("Transform contains an invalid zero quaternion")

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

def main(args=None):
    rclpy.init(args=args)

    subscriber = AudioCLIPSub()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        if rclpy.ok():
            subscriber.get_logger().info('Shutting down...')
    finally:
        if rclpy.ok():
            subscriber.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()