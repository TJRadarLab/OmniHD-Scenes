# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.

# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------



DETECTION_NAMES = ['car', 'pedestrian', 'rider', 'large_vehicle'] 
TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err'] #--无attr

PRETTY_DETECTION_NAMES = {"car":"Car",
            "pedestrian":"Pedestrian",
            "rider":"Rider",
            "large_vehicle":"Large_Vehicle",
            }

DETECTION_COLORS = {'car': 'C0',
                    'pedestrian': 'C1',
                    'rider': 'C2',
                    'large_vehicle': 'C3'}

ATTRIBUTE_NAMES = ['']




PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     }

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    }
