from data_collator import SelfDistillationDataCollator


class MaskPositionDataCollator(SelfDistillationDataCollator):
    """OPSD collator for mask-based position alignment experiments.

    Rollout inputs stay identical to the standard OPSD collator.  The raw
    problem/solution fields are preserved so the trainer can rebuild loss-time
    teacher/student inputs with privilege tokens masked only for the student and
    shared position ids derived from non-privilege tokens.
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch["problems"] = [feature["problem"] for feature in features]
        batch["solutions"] = [feature["solution"] for feature in features]
        return batch
