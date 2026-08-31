import rclpy
from rclpy.node import Node
import numpy as np
import threading
import queue
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch
import os
import sys
from ament_index_python.packages import get_package_share_directory

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

        self.batch_size = 1
        self.audioclip_q = queue.Queue(maxsize=100)

        self.aclp = AudioCLIP(pretrained=f'{audioclip_path}/assets/AudioCLIP-Full-Training.pt')
        self.aclp.eval()
        self.audio_transforms = ToTensor1D()

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
                    'footsteps',
                    'door open or close',
                    'car engine',
                    'cat',
                    'alarm clock'
                ]

        text = [[label] for label in self.LABELS]
        with torch.no_grad():
            ((_, _, self.text_features), _), _ = self.aclp(text=text)
        self.text_features = self.text_features / torch.linalg.norm(self.text_features, dim=-1, keepdim=True)
        self.scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(), min=1.0, max=100.0)

        # test audio
        self.test_audio = list()
        for path_to_audio in glob.glob('/ros2_ws/src/phd_semantic_nav/AudioCLIP/demo/audio/*.wav'):
            track, _ = librosa.load(path_to_audio, sr=44100, dtype=np.float32)
            self.test_audio.append(track)

        audio_transforms = ToTensor1D()
        self.test_audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track in self.test_audio])
        ((self.test_audio_features, _, _), _), _ = self.aclp(audio=self.test_audio)
        self.test_audio_features = self.test_audio_features / torch.linalg.norm(self.test_audio_features, dim=-1, keepdim=True)
        # test audio end

        self.sss_sub = self.create_subscription(Float32MultiArray, '/audioclip_sss', self.sss_callback, audio_qos)
        self.sss_sub  # prevent unused variable warning

        self.thread = threading.Thread(target=self.inference_thread, daemon=True)
        self.thread.start()
    
    def sss_callback(self, msg):
        self.get_logger().info(f'Placing in queue..')
        data = np.array(msg.data, dtype=np.float32).reshape((msg.layout.dim[0].size, msg.layout.dim[1].size))
        rms = np.sqrt(np.mean(data**2))
        if rms < 1e-6:
            self.get_logger().info(f"\033[91mDiscarding...\033[0m")
            return

        # per-chunk normalization
        # data = data / rms * 0.1
        # self.get_logger().info(f"\033[95mReceived min: {data.min():.6f}, max: {data.max():.6f}\033[0m")
        # self.get_logger().info(f"\033[95mReceived RMS: {np.sqrt(np.mean(data**2)):.6f}\033[0m")
        # self.get_logger().info(f"\033[95mReceived vals: {data[:, :5]}\033[0m")
        self.audioclip_q.put(data)
    
    def inference_thread(self):
        while rclpy.ok():
            minibatch = []

            for _ in range(self.batch_size):
                self.get_logger().info(f'Trying to get from queue...')
                data = self.audioclip_q.get()
                self.get_logger().info(f'Got!!')
                # data = self.audio_transforms(np.array(data, dtype=np.float32).reshape(1, -1))
                data = torch.from_numpy(data)
                minibatch.append(data)
            
            try:
                self.get_logger().info(f'Starting inference...')
                audio = torch.stack(minibatch)
                audio = audio.squeeze(dim=0)
                audio = audio.unsqueeze(dim=1)
                self.get_logger().info(f'Size: {audio.size()}')
                # compare with test audio
                if audio.size(0) == 1:
                    test_similarity = cosine_similarity(self.test_audio.numpy()[0,:,:44100], audio.numpy()[0])
                    self.get_logger().info(f"\033[91mTest similarity input: {test_similarity}\033[0m")
                with torch.no_grad():
                    ((audio_features, _, _), _), _ = self.aclp(audio=audio)
                audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
                logits_audio_text = self.scale_audio_text * audio_features @ self.text_features.T

                # compare with test audio embeddings
                if audio_features.size(0) == 1:
                    test_similarity = cosine_similarity(self.test_audio_features.numpy(), audio_features.numpy())
                    self.get_logger().info(f"\033[91mTest similarity embeddings: {test_similarity}\033[0m")

                # self.get_logger().info(f"Audio embeddings shape: {audio_features.size()}")
                # self.get_logger().info(f"\033[95mEmbedding vals: {audio_features.numpy()[:,:5]}\033[0m")
                # self.get_logger().info(f"\033[95mEmbedding std: {audio_features.numpy().std():.6f}\033[0m")
                # self.get_logger().info(f"\033[95mEmbedding min: {audio_features.numpy().min():.6f}, max: {audio_features.numpy().max():.6f}\033[0m")

                # Zero-shot classification
                confidence = logits_audio_text.softmax(dim=1)
                for audio_idx in range(audio_features.size(0)):
                    # acquire Top-3 most similar results
                    conf_values, ids = confidence[audio_idx].topk(3)

                    # format output strings
                    query = f'{audio_idx} ->\t'
                    results = ', '.join([f'{self.LABELS[i]:>5s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

                    self.get_logger().info(f"\033[95m{query + results}\033[0m")

                if audio_features.size(0) > 1:
                    distances = cosine_similarity(audio_features.numpy())
                    self.get_logger().info(f"\033[95mDistances: {distances}\033[0m")
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