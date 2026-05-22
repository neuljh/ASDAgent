from pydantic import BaseModel, Field
from typing import Literal

class DoctorOpening(BaseModel):
    role: Literal["医生"]
    content: str
    strategy: Literal["指令"]
    reason: str