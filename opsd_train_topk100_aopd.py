"""Train the TopK100 + AOPD non-positive handling variant without editing opsd_train.py."""

import runpy

import opsd_trainer
from opsd_trainer_topk_variants import TopKAOPDNonPositiveOPSDTrainer


opsd_trainer.OPSDTrainer = TopKAOPDNonPositiveOPSDTrainer
runpy.run_module("opsd_train", run_name="__main__")
