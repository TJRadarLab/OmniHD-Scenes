# newScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# ---------------------------------------------
# Modified by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
import json
from typing import Dict, Tuple

import numpy as np
import tqdm
from pyquaternion import Quaternion

from newscenes_devkit.newscenes import NewScenes
from newscenes_devkit.eval.common.data_classes import EvalBoxes
from newscenes_devkit.eval.detection.data_classes import DetectionBox
from newscenes_devkit.eval.detection.utils import category_to_detection_name
from newscenes_devkit.eval.tracking.data_classes import TrackingBox
from newscenes_devkit.data_classes import Box
from newscenes_devkit.geometry_utils import points_in_box

# from newscenes_devkit.splits import create_splits_scenes


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' 
                              

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta


def load_gt(newsc: NewScenes, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param newsc: A NewScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """

    if verbose:
        print('Loading annotations for {} split from newScenes version: {}'.format(eval_split, newsc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in newsc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = newsc.scene_split

    # Check compatibility of split with newsc_version.
    version = newsc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NewScenes version {}'.format(eval_split, version)
    elif eval_split in {'train_mini', 'val_mini'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NewScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NewScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NewScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(newsc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'
    
    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = newsc.get('sample', sample_token)['scene_token']
        if scene_token in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):
        
        gt_boxes = newsc.get_annotation_box(sample_token)

        sample_boxes = []
        for i, box in enumerate(gt_boxes):

            if box_cls == DetectionBox:
                
                detection_name = category_to_detection_name(box.name)
                if detection_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=tuple(box.center.tolist()),
                        size=tuple(box.wlh.tolist()),
                        rotation=tuple(box.orientation.elements.tolist()),
                        velocity=tuple(box.velocity[:2].tolist()),
                        ego_translation = tuple(box.center.tolist()),
                        num_pts=-1, 
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name='', 
                        visibility=box.visibility,
                    )
                )

            #TODO#-----------------------------------------------------------------
            elif box_cls == TrackingBox:
                # Use newScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from newscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue
            
                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=newsc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            #--------------------------------------------------------------------
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations



#-----------
def filter_eval_boxes(newsc: NewScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False,
                      bad_conditions: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param newsc: An instance of the NewScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field =_get_box_class_field(eval_boxes) #---detection_name

    # Accumulators for number of filtered boxes.
    total, dist_filter, visibility_filter = 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        # eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
        #                                   box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          abs(box.ego_translation[0]) <= max_dist[box.__getattribute__(class_field)][0] \
                                            and abs(box.ego_translation[1]) <= max_dist[box.__getattribute__(class_field)][1]]
        dist_filter += len(eval_boxes[sample_token])

        
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if box.visibility == 1]
        visibility_filter += len(eval_boxes[sample_token])


    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After Camera visibility based filtering: %d" % visibility_filter)
    
    if bad_conditions:
        # Filter on bad conditions.
        ori_sample_tokens = eval_boxes.sample_tokens
        for sample_token in ori_sample_tokens:
            scene_token = newsc.get('sample', sample_token)['scene_token']
            scene_meta_dict = newsc.get('meta', scene_token)['meta']
            weather = scene_meta_dict['weather']
            lighting = scene_meta_dict['lighting']
            if not (weather == 'rainy' or lighting == 'night'):
                del eval_boxes.boxes[sample_token]

        print("=> After Bad conditions based filtering: ",len(eval_boxes.all))
    
    
    
    #----------------------------------------------------------------
    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
