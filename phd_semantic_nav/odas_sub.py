import rclpy
from rclpy.node import Node
import numpy as np
from odas_ros_msgs.msg import OdasSstArrayStamped
from audio_utils_msgs.msg import AudioFrame
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading

class OdasSub(Node):
    def __init__(self):
        super().__init__('odas_sub')
        self.lock = threading.Lock()
        self.active_sources = {}
        self.npots = 4
        self.threshold = 0.6
        self.sample_rate = 16000
        self.example_duration = 1.0
        self.sample_batch = int(self.sample_rate * self.example_duration)  # Samples number of the example for AudioCLIP
        self.audio_buffers = [[] for _ in range(self.npots)]

        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        self.sst_sub = self.create_subscription(OdasSstArrayStamped, '/sst', self.sst_callback, audio_qos)
        self.sss_sub = self.create_subscription(AudioFrame, '/sss', self.sss_callback, audio_qos)
        self.sss_pub = self.create_publisher(Float32MultiArray, '/audioclip_sss', audio_qos)
        self.sst_sub  # prevent unused variable warning
        self.sss_sub  # prevent unused variable warning
    
    def sst_callback(self, msg):
        self.get_logger().info(f"SST: {msg.sources}")
        with self.lock:
            self.active_sources = {src.id: (src.x, src.y, src.z, src.activity) for src in msg.sources}
    
    def sss_callback(self, msg):
        with self.lock:
            has_active = bool(self.active_sources)
            active_sources = dict(self.active_sources)

        if not has_active:
            return

        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        for i, (key, val) in enumerate(active_sources.items()):
            if i >= min(msg.channel_count, self.npots): continue
            if val[3] < self.threshold or key == 0: continue
            self.get_logger().info(f"SST: {val}")
            audio = samples[i::msg.channel_count]
            self.audio_buffers[i].extend(audio)

            if len(self.audio_buffers[i]) >= self.sample_batch:
                example = np.array(self.audio_buffers[i][:self.sample_batch], dtype=np.float32) / 32767.0
                self.audio_buffers[i] = self.audio_buffers[i][self.sample_batch:]
                self.get_logger().info(f"{example.shape}")
                example_msg = Float32MultiArray()
                example_msg.data = example.flatten().tolist()
                self.sss_pub.publish(example_msg)

def main(args=None):
    rclpy.init(args=args)

    subscriber = OdasSub()

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