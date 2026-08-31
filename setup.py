from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'phd_semantic_nav'

def production_manifests(*dirs):
    manifests = []
    ignore = ['demo', '__pycache__', '.git', '.vscode', 'images', 'test']

    for dir in dirs:
        if not os.path.exists(dir): continue
        for root, _, filenames in os.walk(dir):
            if any(el in root for el in ignore): continue
            install_target = os.path.join('share', package_name, root)
            source_list = []
            for filename in filenames:
                source_list.append(os.path.join(root, filename))
            manifests.append((install_target, source_list))
    
    return manifests

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],  # + production_manifests('.venv', 'AudioCLIP')
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matthew',
    maintainer_email='matthew@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 'odas_sub = phd_semantic_nav.odas_sub:main',
            'audioclip = phd_semantic_nav.audioclip:main',
            # 'hydra_sub = phd_semantic_nav.hydra_sub:main'
        ],
    },
    scripts=[
        'scripts/uma16v2_pos_setup.py',
    ],
)
