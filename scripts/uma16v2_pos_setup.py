#!/usr/bin/env python3
import sys
import os
import numpy as np

def main():
    pos = []
    mapping = (8, 7, 10, 9, 6, 5, 12, 11, 4, 3, 14, 13, 2, 1, 16, 15)
    midpoint = sum(list(range(4))) / 4
    mic_dist = 0.042
    for x in range(4):
        for y in range(4):
            pos.append(((midpoint - x) * mic_dist, (y - midpoint) * mic_dist, 0.0))
    if len(mapping) != len(pos):
        print("Lengths don't match!")
        return

    cfg = ""
    for i, mu in zip(mapping, pos):
        cfg += f"# Microphone {i} #TODO\n{{\n\tmu = ( {', '.join([str(num).upper() for num in mu])} );\n"
        # cfg += "\tsigma2 = ( +1E-6, 0.0, 0.0, 0.0, +1E-6, 0.0, 0.0, 0.0, +1E-6 );\n"
        cfg += "\tsigma2 = ( +0.0, 0.0, 0.0, 0.0, +0.0, 0.0, 0.0, 0.0, +0.0 );\n"
        cfg += "\tdirection = ( 0.0, 0.0, +1.0 );\n"
        cfg += "\tangle = ( 80.0, 100.0 );\n},\n\n"
    
    curr_path = os.path.abspath(__file__)
    if '/install/' in curr_path:
        ros_ws = curr_path.split('/install/')[0]
        pkg_name = curr_path.split('/install/')[1].split('/')[0]
        output_file = os.path.join(ros_ws, 'src', pkg_name, 'scripts', 'pos.cfg')
    else:
        output_file = os.path.join(os.path.dirname(curr_path), 'pos.cfg')

    with open(output_file, 'w') as f:
        f.write(cfg)

if __name__ == "__main__":
    main()