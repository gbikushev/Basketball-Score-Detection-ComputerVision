from pydantic import EmailStr
from fastapi_users import schemas

class UserRead(schemas.BaseUser[int]):
    id: int
    email: str
    username: str
    role_id: int
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False

    class Config:
        orm_mode = True


class UserCreate(schemas.CreateUpdateDictModel):
    username: str
    email: EmailStr
    password: str
