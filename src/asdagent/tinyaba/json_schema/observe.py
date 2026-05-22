
from pydantic import BaseModel, Field
from typing import Literal

class SemanticAssessment(BaseModel):
    sign: bool = Field(..., description="儿童回答是否存在语义关联")
    reason: str = Field(..., description="判断儿童回答是否具有语义的理由")

class EmotionalState(BaseModel):
    stress: Literal["Low", "Medium", "High"]
    engagement: Literal["High", "Medium", "Low"]

class ResponseData(BaseModel):
    response_type: Literal["不相关的回答", "相关的回答", "无响应", "重复"]
    reason: str = Field(..., description="说明你的判断依据，需要引用医生上个回复和儿童的回复详细分析")
    semantic: SemanticAssessment
    quality_assessment: str = Field(..., description="对儿童回复的简短评价")
    emotional_state: EmotionalState
    detailed_observation: str = Field(
        ...,
        description="综合行为与情绪的观察性描述"
    )
