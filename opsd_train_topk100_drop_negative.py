"""Train the TopK100 + DropNegativePosition OPSD variant without editing opsd_train.py."""

import runpy

import opsd_trainer
from opsd_trainer_topk_variants import TopKDropNegativePositionOPSDTrainer


opsd_trainer.OPSDTrainer = TopKDropNegativePositionOPSDTrainer
runpy.run_module("opsd_train", run_name="__main__")
