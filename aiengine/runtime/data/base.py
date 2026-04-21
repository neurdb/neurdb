from typing import Any, Dict, Annotated
from pydantic import BaseModel, ConfigDict, Field


NonEmptyStr = Annotated[str, Field(min_length=1)]

class RuntimeDataModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls.model_validate(data)