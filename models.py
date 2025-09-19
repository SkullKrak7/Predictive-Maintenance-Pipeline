from pydantic import BaseModel, Field


class InputModel(BaseModel):
    rotational_speed: float = Field(..., ge=1100, le=3000, description="rpm")
    torque: float = Field(..., ge=0, le=100, description="Nm")
    tool_wear: float = Field(..., ge=0, le=300, description="minutes")


class OutputModel(BaseModel):
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_label: int = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0.0, le=1.0)
