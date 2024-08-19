from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, AuthenticationBackend
from fastapi_users.authentication import JWTStrategy

from src.auth.manager import get_user_manager
from src.auth.models import User
from src.config import SECRET_AUTH

# Default cookie transport
cookie_transport = CookieTransport(cookie_name="bonds", cookie_max_age=3600)

# Extended cookie transport for processing
extended_cookie_transport = CookieTransport(cookie_name="bonds", cookie_max_age=7200)


def get_jwt_strategy(processing: bool = False) -> JWTStrategy:
    lifetime = 3600
    if processing:
        lifetime = 7200   
    return JWTStrategy(secret=SECRET_AUTH, lifetime_seconds=lifetime)

# auth_backend = AuthenticationBackend(
#     name="jwt",
#     transport=cookie_transport,
#     get_strategy=get_jwt_strategy,
# )

# Authentication backend
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# Extended authentication backend for processing
extended_auth_backend = AuthenticationBackend(
    name="jwt_extended",
    transport=extended_cookie_transport,
    get_strategy=lambda: get_jwt_strategy(processing=True),
)

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

current_user = fastapi_users.current_user(active=True)