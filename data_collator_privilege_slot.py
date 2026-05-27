from data_collator import SelfDistillationDataCollator


class PrivilegeSlotDataCollator(SelfDistillationDataCollator):
    """OPSD collator for position-aligned privilege-slot experiments.

    The standard OPSD collator still builds clean student rollout prompts.  This
    subclass only preserves raw dataset fields so the trainer can rebuild the
    loss-time student/teacher prefixes with equal-length privilege slots.
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch["problems"] = [feature["problem"] for feature in features]
        batch["solutions"] = [feature["solution"] for feature in features]
        return batch
