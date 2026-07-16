import os
import random
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import jwt
import bcrypt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

import database

logger = logging.getLogger(__name__)

# --- Password Hashing ---
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# --- JWT Configuration ---
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-to-a-strong-random-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# --- Email Configuration ---
mail_port = int(os.getenv("MAIL_PORT", 587))
mail_conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", ""),
    MAIL_FROM=os.getenv("MAIL_FROM", os.getenv("MAIL_USERNAME", "noreply@example.com")),
    MAIL_PORT=mail_port,
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS=os.getenv("MAIL_STARTTLS", str(mail_port == 587)).lower() in ("true", "1", "t"),
    MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS", str(mail_port == 465)).lower() in ("true", "1", "t"),
    USE_CREDENTIALS=True,
)
fast_mail = FastMail(mail_conf)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# --- Request Models ---
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# --- Helper to generate OTP ---
def generate_otp() -> str:
    return str(random.randint(100000, 999999))

# --- Endpoints ---

@router.post("/signup")
async def signup(req: SignupRequest):
    """Register a new user: hash password, generate OTP, send verification email."""
    email = req.email.lower().strip()

    # Check if a verified user already exists
    def check_existing():
        return database.supabase.table("users").select("id").eq("email", email).eq("is_verified", True).execute()
    existing = await asyncio.to_thread(check_existing)
    if existing.data:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    hashed_password = hash_password(req.password)

    # Upsert into users table (unverified)
    def upsert_user():
        return database.supabase.table("users").upsert(
            {"email": email, "hashed_password": hashed_password, "is_verified": False},
            on_conflict="email"
        ).execute()
    await asyncio.to_thread(upsert_user)

    # Generate OTP and store it
    otp = generate_otp()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    def upsert_otp():
        return database.supabase.table("otp_codes").upsert(
            {"email": email, "otp": otp, "expires_at": expires_at},
            on_conflict="email"
        ).execute()
    await asyncio.to_thread(upsert_otp)

    # Send OTP email
    try:
        message = MessageSchema(
            subject="InterviewRAG - Verify Your Email",
            recipients=[email],
            body=f"""
            <h2>Welcome to InterviewRAG!</h2>
            <p>Your verification code is:</p>
            <h1 style="letter-spacing: 8px; color: #238636; font-size: 36px;">{otp}</h1>
            <p>This code expires in <b>10 minutes</b>.</p>
            <p>If you did not request this, you can safely ignore this email.</p>
            """,
            subtype=MessageType.html,
        )
        await fast_mail.send_message(message)
        logger.info(f"OTP email sent to {email}")
    except Exception as e:
        logger.error(f"Failed to send OTP email: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to send verification email. Please check SMTP configuration.")

    return {"message": "Verification code sent to your email."}


@router.post("/verify-otp")
async def verify_otp(req: VerifyOtpRequest):
    """Verify the OTP, mark user as verified, and return a JWT token."""
    email = req.email.lower().strip()

    # Fetch OTP record
    def get_otp():
        return database.supabase.table("otp_codes").select("*").eq("email", email).execute()
    otp_resp = await asyncio.to_thread(get_otp)

    if not otp_resp.data:
        raise HTTPException(status_code=400, detail="No OTP found for this email. Please sign up again.")

    otp_record = otp_resp.data[0]

    # Check expiry
    expires_at = datetime.fromisoformat(otp_record["expires_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail="OTP has expired. Please sign up again.")

    # Check OTP match
    if otp_record["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP.")

    # Mark user as verified
    def verify_user():
        return database.supabase.table("users").update({"is_verified": True}).eq("email", email).execute()
    user_resp = await asyncio.to_thread(verify_user)

    # Fetch the user to get the ID
    def get_user():
        return database.supabase.table("users").select("id").eq("email", email).execute()
    user_data = await asyncio.to_thread(get_user)

    if not user_data.data:
        raise HTTPException(status_code=500, detail="User record not found after verification.")

    user_id = str(user_data.data[0]["id"])

    # Clean up OTP
    def delete_otp():
        return database.supabase.table("otp_codes").delete().eq("email", email).execute()
    await asyncio.to_thread(delete_otp)

    token = create_access_token(user_id, email)
    logger.info(f"User {email} verified and logged in.")
    return {"token": token, "user": {"id": user_id, "email": email}}


@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate a user with email/password and return a JWT token."""
    email = req.email.lower().strip()

    def get_user():
        return database.supabase.table("users").select("*").eq("email", email).eq("is_verified", True).execute()
    user_resp = await asyncio.to_thread(get_user)

    if not user_resp.data:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = user_resp.data[0]

    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(str(user["id"]), email)
    logger.info(f"User {email} logged in.")
    return {"token": token, "user": {"id": str(user["id"]), "email": email}}


@router.post("/resend-otp")
async def resend_otp(req: SignupRequest):
    """Resend OTP for an unverified user."""
    email = req.email.lower().strip()

    # Check user exists and is unverified
    def check_user():
        return database.supabase.table("users").select("*").eq("email", email).eq("is_verified", False).execute()
    user_resp = await asyncio.to_thread(check_user)

    if not user_resp.data:
        raise HTTPException(status_code=400, detail="No pending signup found for this email.")

    # Verify the password matches the stored hash
    user = user_resp.data[0]
    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    otp = generate_otp()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    def upsert_otp():
        return database.supabase.table("otp_codes").upsert(
            {"email": email, "otp": otp, "expires_at": expires_at},
            on_conflict="email"
        ).execute()
    await asyncio.to_thread(upsert_otp)

    try:
        message = MessageSchema(
            subject="InterviewRAG - Your New Verification Code",
            recipients=[email],
            body=f"""
            <h2>InterviewRAG Verification</h2>
            <p>Your new verification code is:</p>
            <h1 style="letter-spacing: 8px; color: #238636; font-size: 36px;">{otp}</h1>
            <p>This code expires in <b>10 minutes</b>.</p>
            """,
            subtype=MessageType.html,
        )
        await fast_mail.send_message(message)
        logger.info(f"Resent OTP email to {email}")
    except Exception as e:
        logger.error(f"Failed to resend OTP email: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to send verification email.")

    return {"message": "A new verification code has been sent to your email."}

@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """Generate OTP for password reset and send email."""
    email = req.email.lower().strip()

    def check_existing():
        return database.supabase.table("users").select("id").eq("email", email).eq("is_verified", True).execute()
    existing = await asyncio.to_thread(check_existing)
    if not existing.data:
        raise HTTPException(status_code=404, detail="No verified account found with this email.")

    otp = generate_otp()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    def upsert_otp():
        return database.supabase.table("otp_codes").upsert(
            {"email": email, "otp": otp, "expires_at": expires_at},
            on_conflict="email"
        ).execute()
    await asyncio.to_thread(upsert_otp)

    try:
        message = MessageSchema(
            subject="InterviewRAG - Password Reset",
            recipients=[email],
            body=f"""
            <h2>InterviewRAG Password Reset</h2>
            <p>Your password reset code is:</p>
            <h1 style="letter-spacing: 8px; color: #238636; font-size: 36px;">{otp}</h1>
            <p>This code expires in <b>10 minutes</b>.</p>
            """,
            subtype=MessageType.html,
        )
        await fast_mail.send_message(message)
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to send reset email.")

    return {"message": "Password reset code has been sent to your email."}

@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Verify OTP and update user's password."""
    email = req.email.lower().strip()

    def fetch_otp():
        return database.supabase.table("otp_codes").select("*").eq("email", email).execute()
    otp_data = await asyncio.to_thread(fetch_otp)

    if not otp_data.data:
        raise HTTPException(status_code=400, detail="No active reset request found for this email.")

    record = otp_data.data[0]
    if record["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid reset code.")

    expires_at = datetime.fromisoformat(record["expires_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail="Reset code has expired. Please request a new one.")

    hashed_password = hash_password(req.new_password)

    def update_password():
        return database.supabase.table("users").update(
            {"hashed_password": hashed_password}
        ).eq("email", email).execute()
    await asyncio.to_thread(update_password)

    def delete_otp():
        return database.supabase.table("otp_codes").delete().eq("email", email).execute()
    await asyncio.to_thread(delete_otp)

    return {"message": "Password successfully reset."}
