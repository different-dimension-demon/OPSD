from data_collator import SelfDistillationDataCollator


class StudentAlignTeacherDataCollator(SelfDistillationDataCollator):
    """OPSD collator for student-aligns-to-teacher position experiments.

    Rollout inputs stay identical to the standard OPSD collator.  The raw
    problem/solution fields are preserved so the trainer can rebuild loss-time
    teacher/student inputs with privilege tokens masked only for the student,
    while the privilege slots still advance shared position ids.
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch["problems"] = [feature["problem"] for feature in features]
        batch["solutions"] = [feature["solution"] for feature in features]
        return batch
