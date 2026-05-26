from data_collator import SelfDistillationDataCollator


class StudentGuidedTeacherDataCollator(SelfDistillationDataCollator):
    """Data collator for student-guided teacher experiments.

    It reuses the standard OPSD prompt construction, but keeps raw problem and
    solution text so the trainer can build teacher-guidance prompts after the
    student rollout is generated.
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch["problems"] = [feature["problem"] for feature in features]
        batch["solutions"] = [feature["solution"] for feature in features]
        return batch
