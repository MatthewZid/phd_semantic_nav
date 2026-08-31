import rclpy
from rclpy.node import Node
import numpy as np
from hydra_msgs.msg import DsgUpdate
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import sys
sys.path.insert(0, '/home/matthew/dev/hydra_stuff/hydra_env/hydra/lib/python3.12/site-packages')
import spark_dsg

class HydraSub(Node):
    def __init__(self):
        super().__init__('hydra_sub')

        self.graph = None

        # hydra_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=100
        # )

        self.hydra_sub = self.create_subscription(DsgUpdate, '/hydra/backend/dsg', self.hydra_callback, 50)
        self.hydra_sub  # prevent unused variable warning
    
    def hydra_callback(self, msg):
        # deserialize the graph binary message
        # (notice that this way of getting the object centroid positions in the map frame, it keeps two graph instances in memory,
        # so it's not optimized for real time and intensive execution. The best way is to implement the position querying mechanism inside
        # the C++ Hydra code)
        if msg.full_update or self.graph is None:
            binary_data = bytes(msg.layer_contents)
            self.graph = spark_dsg.DynamicSceneGraph.from_binary(binary_data)
        else:
            self.graph.update_from_binary(binary_data)

        object_layer = self.graph.get_layer(spark_dsg.DsgLayers.OBJECTS)
        for node in object_layer.nodes:
            position = self.graph.get_position(node.id.value)
            self.get_logger().info(f"{position}")
            break


def main(args=None):
    rclpy.init(args=args)

    subscriber = HydraSub()

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