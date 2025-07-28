# NewScenes dev-kit.
# Code written by Holger Caesar, 2018.

# ---------------------------------------------
# Modified by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

from typing import List, Optional

#------------------映射成检测的标签名，不在评估范围的类别滤掉，这里是从原始标映射并选择需要的标签-------
#-----------------目前直接过滤掉了一些类别，后期可以根据需要修改-----------------
def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from NewScenes to NewScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic NewScenes class.
    :return: NewScenes detection class.
    """
 
    detection_mapping = {
            "suv":"car",
            "van":"car",
            "truck":"large_vehicle",
            "rider":"rider",
            # 'cyclist':'rider',
            "pedestrian":"pedestrian",
            "car":"car",
            "tricyclist":"car",
            "light_truck":"large_vehicle",
            "bus":"large_vehicle",
            "engineering_vehicle":"large_vehicle",
            "handcart":"car",
            "trailer":"large_vehicle",
            } 

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None


def detection_name_to_rel_attributes(detection_name: str) -> List[str]:
    """
    Returns a list of relevant attributes for a given detection class.
    :param detection_name: The detection class.
    :return: List of relevant attributes.
    """
    if detection_name in ['pedestrian']:
        rel_attributes = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing']
    elif detection_name in ['bicycle', 'motorcycle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif detection_name in ['car', 'bus', 'construction_vehicle', 'trailer', 'truck']:
        rel_attributes = ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    elif detection_name in ['barrier', 'traffic_cone']:
        # Classes without attributes: barrier, traffic_cone.
        rel_attributes = []
    else:
        raise ValueError('Error: %s is not a valid detection class.' % detection_name)

    return rel_attributes

