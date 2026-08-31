from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, LogInfo, TimerAction
from launch.event_handlers import OnProcessStart
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # venv_path = os.path.join(get_package_share_directory('phd_semantic_nav'), '.venv/audioclip/bin/python') # for production
    # venv_path = '/home/matthew/dev/ros_ws/src/phd_semantic_nav/.venv/audioclip/bin/python'
    venv_path = '/ros2_ws/src/phd_semantic_nav/.venv/audioclip/bin/python'

    rviz_config = PathJoinSubstitution([
        FindPackageShare('phd_semantic_nav'),
        'rviz',
        'odas.rviz'
    ])

    odas_config = PathJoinSubstitution([
        FindPackageShare('odas_ros'),
        'config',
        'configuration_uma16v2.cfg'
    ])

    odas_vis_node = Node(
        package='odas_ros',
        namespace='odas',
        executable='odas_visualization_node.py',
        name='odas_visualization_node',
        parameters=[{'configuration_path': odas_config}],
        output='screen'
    )
    
    odas_server_node = Node(
        package='odas_ros',
        namespace='odas',
        executable='odas_server_node.py',
        name='odas_server_node',
        parameters=[{'configuration_path': odas_config}],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    ssl_sub_node = Node(
        package='phd_semantic_nav',
        namespace='phd_semantic',
        executable='odas_sub',
        name='odas_sub',
        output='screen'
    )

    audioclip_node = Node(
        package='phd_semantic_nav',
        namespace='phd_semantic',
        executable='audioclip',
        name='audioclip',
        output='screen',
        prefix=venv_path
    )
    
    return LaunchDescription([
        audioclip_node,
        # TimerAction(
        #     period=1.0,
        #     actions=[ssl_sub_node]
        # ),
        # rviz_node,
        # TimerAction(
        #     period=1.0,
        #     actions=[audioclip_node]
        # ),
        # TimerAction(
        #     period=2.0,
        #     actions=[odas_vis_node]
        # ),
        # TimerAction(
        #     period=3.0,
        #     actions=[ssl_sub_node]
        # ),
        # TimerAction(
        #     period=4.0,
        #     actions=[odas_server_node]
        # )
    ])