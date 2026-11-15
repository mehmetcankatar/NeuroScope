from typing import Literal
from pydantic import BaseModel, Field, model_validator

Head = Literal["abnormal", "seizure", "sleep"]


class AnalysisRequest(BaseModel):
    heads: list[Head] = Field(default_factory=lambda: ["abnormal", "seizure", "sleep"], min_length=1)
    enter_threshold: float = Field(default=0.7, ge=0.05, le=0.99)
    exit_threshold: float = Field(default=0.45, ge=0.01, le=0.99)
    persistence_s: float = Field(default=3, ge=1, le=60)
    merge_gap_s: float = Field(default=2, ge=0, le=30)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("Exit threshold must not exceed enter threshold")
        self.heads = list(dict.fromkeys(self.heads))
        return self


class AnnotationRequest(BaseModel):
    target_type: Literal["event", "sleep", "manual"]
    target_id: str = Field(min_length=1, max_length=100)
    decision: Literal["accepted", "rejected", "uncertain", "corrected", "unreviewed"]
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    label: str | None = Field(default=None, max_length=80)
    note: str = Field(default="", max_length=4000)
    author: str = Field(default="Local reviewer", min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_interval(self):
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("Provide both start and end")
        if self.start_s is not None and self.end_s <= self.start_s:
            raise ValueError("End must be later than start")
        if self.target_type == "manual" and (self.start_s is None or not self.label):
            raise ValueError("Manual findings need an interval and label")
        return self
