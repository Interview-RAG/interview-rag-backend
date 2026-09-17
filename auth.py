import os
import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# This module is imported before `database`, which used to be the first thing
# to call load_dotenv(). Without this call JWT_SECRET below silently fell back
# to its placeholder default while the token *signer* in routes/auth_routes.py
# — imported later, after .env had loaded — used the real value. The verifier
# then rejected every genuine login and accepted anything signed with the
# well-known default string, which is a full authentication bypass.
load_dotenv()

security = HTTPBearer()

PLACEHOLDER_SECRET = "change-me-to-a-strong-random-secret"

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"

if not JWT_SECRET or JWT_SECRET == PLACEHOLDER_SECRET:
    # Fail closed. A guessable signing key lets anyone mint a token for any
    # user id, so refusing to start is safer than serving forgeable auth.
    raise RuntimeError(
        "JWT_SECRET is missing or still set to the placeholder default. "
        "Set a strong random JWT_SECRET in backend/.env (local) and in the "
        "hosting environment (production) before starting the API."
    )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Dependency to verify the custom JWT token and extract the user_id.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user ID found")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
