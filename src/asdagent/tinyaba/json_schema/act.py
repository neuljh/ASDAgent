from pydantic import BaseModel, Field
from typing import Literal

class DoctorResponse(BaseModel):
    role: Literal["医生"]
    content: str
    strategy: Literal["指令", "强化", "半辅助", "其他", "全辅助", ""] = Field(..., description="选择的策略,如指令、强化、半辅助、全辅助、其他或空字符串表示无策略")