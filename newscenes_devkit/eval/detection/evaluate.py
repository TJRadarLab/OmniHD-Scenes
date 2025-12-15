# newScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

# ---------------------------------------------
# Modified by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from newscenes_devkit.newscenes import NewScenes
from newscenes_devkit.eval.common.config import config_factory
from newscenes_devkit.eval.common.data_classes import EvalBoxes
from newscenes_devkit.eval.common.loaders import load_prediction, load_gt, filter_eval_boxes
from newscenes_devkit.eval.detection.algo import accumulate, calc_ap, calc_tp
from newscenes_devkit.eval.detection.constants import TP_METRICS
from newscenes_devkit.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from newscenes_devkit.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample


class DetectionEval:
    """
    This is the official newScenes detection evaluation code.
    Results are written to the provided output_dir.

    newScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation errors.
    - newScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    """

    def __init__(self,
                 newsc: NewScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 bad_conditions: bool = False):  #--------add tag---
        """
        Initialize a DetectionEval object.
        :param newsc: A NewScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the newScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.newsc = newsc
        self.result_path = result_path  # json detection result
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config  # Use config_factory in config.py to create DetectionConfig instance

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing newScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.newsc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        #---------------Add filtering conditions for adverse weather----------------
        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(newsc, self.pred_boxes, self.cfg.class_range, verbose=verbose,
                                            bad_conditions=bad_conditions)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(newsc, self.gt_boxes, self.cfg.class_range, verbose=verbose,
                                          bad_conditions=bad_conditions)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        #--------Compute PR curves per class at distance thresholds [1,2,3,4] and accumulate four TP metrics-------------
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:  # [1,2,3,4]
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            #-------------Compute per-class AP at given distance thresholds-------------------
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            #----------------Compute per-class TP errors at the given threshold---------------
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Render PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            # return os.path.join(self.plot_dir, name + '.pdf') #----switch pdf/png if desired
            return os.path.join(self.plot_dir, name + '.png') #----default to png

        #-----------Plot PR and TP curves for every class-------------------------
        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        #-----------Plot PR and TP curves per class-----------------------
        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        #-------PR curves across classes for each distance threshold---------
        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, optionally visualizes samples,
        runs the evaluation, and renders statistic plots.
        :param plot_examples: Number of example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: High-level metrics and meta data.
        """
        # #-------Typically 0; randomly visualize some detections vs GT in BEV-----------
        # #----TODO not finished------
        # if plot_examples > 0:
        #     random.seed(42)
        #     sample_tokens = list(self.sample_tokens)
        #     random.shuffle(sample_tokens)
        #     sample_tokens = sample_tokens[:plot_examples]
        #     example_dir = os.path.join(self.output_dir, 'examples')
        #     if not os.path.isdir(example_dir):
        #         os.mkdir(example_dir)
        #     for sample_token in sample_tokens:
        #         visualize_sample(self.newsc,
        #                          sample_token,
        #                          self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
        #                          self.pred_boxes,
        #                          eval_range=max(self.cfg.class_range.values()),
        #                          savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NOS: %.4f' % (metrics_summary['NOS']))  #---newscenes overall score [NOS]
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                  % (class_name,
                     class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))

        return metrics_summary


class NewScenesEval(DetectionEval):
    """Dummy class for backward-compatibility. Same as DetectionEval."""


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate newScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/newscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/newscenes',
                        help='Default newScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the newScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file. If no path given, the CVPR 2019 configuration is used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--bad_conditions', type=int, default=0,
                        help='Whether to evaluate bad conditions.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    bad_conditions = bool(args.bad_conditions)

    if config_path == '':
        cfg_ = config_factory('detection_newsc_config_final')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    newsc_ = NewScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    newsc_eval = DetectionEval(newsc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_, bad_conditions=bad_conditions)
    newsc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
