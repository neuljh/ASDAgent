from pydantic import BaseModel, Field
from typing import Literal, Optional

class RoleContent(BaseModel):
    role: Literal["儿童"]
    content: Optional[str] = Field("", description="口语内容，若无响应则留空")
    type: Literal["不相关的回答", "相关的回答", "无响应", "重复"]
    detail: Optional[str] = Field(None, description="不同儿童回复类型下的具体类型，针对不相关回答的具体类型，如错误回答、代词逆转等")