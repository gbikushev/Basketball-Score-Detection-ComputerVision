from fastapi import FastAPI, Depends, Request
from src.process_video.router_videos import router as router_videos
from src.auth.base_config import auth_backend
from src.auth.models import User
from src.auth.schemas import UserRead, UserCreate
from src.auth.base_config import current_user, fastapi_users
from fastapi.responses import HTMLResponse
from starlette.middleware.wsgi import WSGIMiddleware
from src.dashboard.dashboard import get_dash_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse


app = FastAPI(
    title="Basketball App"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the static files from the '/result_videos' directory
app.mount("/result_videos", StaticFiles(directory="result_videos"), name="result_videos")


@app.get("/dashboard-redirect")
async def protected_route(request: Request, user: User = Depends(current_user)):
    cookie_value = request.cookies.get("bonds")
    response = RedirectResponse(url="/dashboard")
    response.set_cookie(key="bonds", value=cookie_value, httponly=True)
    return response

@app.get("/free-throws", response_class=HTMLResponse)
async def get_free_throws_page():
    with open('public_html/free_throws.html', 'r', encoding='utf-8') as f:
        register_page = f.read()
    return register_page

@app.get("/three-point-throws", response_class=HTMLResponse)
async def get_three_point_throws_page():
    with open('public_html/three_point_throws.html', 'r', encoding='utf-8') as f:
        register_page = f.read()
    return register_page

@app.get("/register", response_class=HTMLResponse)
async def get_register_page():
    with open('public_html/register.html', 'r', encoding='utf-8') as f:
        register_page = f.read()
    return register_page

@app.get("/login", response_class=HTMLResponse)
async def get_login_page():
    with open('public_html/login.html', 'r', encoding='utf-8') as f:
        login_page = f.read()
    return login_page

@app.get("/homepage", response_class=HTMLResponse)
async def get_home_page():
    with open('public_html/homepage.html', 'r', encoding='utf-8') as f:
        video_main = f.read()
    return video_main

@app.get("/about", response_class=HTMLResponse)
async def get_home_page():
    with open('public_html/about.html', 'r', encoding='utf-8') as f:
        video_main = f.read()
    return video_main

@app.get("/developers", response_class=HTMLResponse)
async def get_home_page():
    with open('public_html/developers.html', 'r', encoding='utf-8') as f:
        video_main = f.read()
    return video_main

@app.get("/blog", response_class=HTMLResponse)
async def get_home_page():
    with open('public_html/blog.html', 'r', encoding='utf-8') as f:
        video_main = f.read()
    return video_main


app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)


@app.get("/protected-route")
def protected_route(user: User = Depends(current_user)):
    return user.username

app.include_router(router_videos)

# Create the Dash app
dash_app = get_dash_app()

# Mount Dash app to the FastAPI app
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

