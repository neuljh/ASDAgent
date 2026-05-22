from pydantic import BaseModel, Field
from typing import List, Literal


class ResponseSegment(BaseModel):
    content: str = Field(..., description="该段医生话语的具体内容")
    strategy: Literal["强化", "半辅助", "全辅助", "指令", "其他"] = Field(
        ...,
        description="该段话语采用的ABA策略"
    )


class DoctorFullResponse(BaseModel):
    full_response: str = Field(
        ...,
        description="医生的完整连续回复文本"
    )
    segments: List[ResponseSegment] = Field(
        ...,
        description="将完整回复拆分后的策略标注片段"
    )
