import logging
import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import PasswordResetToken, User
from app.schema import (
    UserCreate,
    UserLogin,
    TokenResponse,
    PasswordResetConfirm,
    PasswordResetRequest,
)
from app.services.auth_service import hash_password, verify_password, create_access_token
from app.config import settings
from app.rate_limit import limiter, RATE_LIMIT_AUTH

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

RESET_RESPONSE = {"message": "If that email is registered, recovery instructions have been sent."}


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _send_reset_email(email: str, token: str) -> None:
    if not settings.SMTP_HOST or not settings.SMTP_FROM_EMAIL:
        if settings.is_production:
            raise RuntimeError("Password recovery email is not configured")
        logger.info("Development password reset token for %s: %s", email, token)
        return

    message = EmailMessage()
    message["Subject"] = "Password recovery"
    message["From"] = settings.SMTP_FROM_EMAIL
    message["To"] = email
    message.set_content(
        "Use this password reset token in the recovery form. "
        f"It expires in {settings.PASSWORD_RESET_EXPIRE_MINUTES} minutes:\n\n{token}"
    )
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
        server.starttls()
        if settings.SMTP_USERNAME:
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD.get_secret_value())
        server.send_message(message)


@router.post("/register", response_model=TokenResponse)
@limiter.limit(RATE_LIMIT_AUTH)
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user.
    
    - **email**: User email address
    - **password**: User password (min 8 characters recommended)
    """
    try:
        # Check whether email already exists
        result = await db.execute(
            select(User).where(func.lower(User.email) == user_data.email)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.warning(f"Registration attempt with existing email: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Hash password
        hashed_password = hash_password(user_data.password)

        # Create user
        user = User(
            email=user_data.email,
            password_hash=hashed_password,
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        # Create JWT
        access_token = create_access_token(user_id=str(user.id), role=user.role)
        
        logger.info(f"User registered successfully: {user.email}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
        }
    
    except HTTPException:
        raise
    except IntegrityError as exc:
        await db.rollback()
        logger.info("Concurrent registration rejected by email uniqueness constraint")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc
    except SQLAlchemyError as exc:
        await db.rollback()
        logger.exception("Database unavailable during registration")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is temporarily unavailable",
        ) from exc
    except Exception as exc:
        await db.rollback()
        logger.exception("Unexpected registration failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        ) from exc


@router.post("/login", response_model=TokenResponse)
@limiter.limit(RATE_LIMIT_AUTH)
async def login(
    request: Request,
    user_data: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    Login user and get access token.
    
    - **email**: User email address
    - **password**: User password
    """
    try:
        result = await db.execute(
            select(User).where(func.lower(User.email) == user_data.email)
        )
        user = result.scalar_one_or_none()

        if not user:
            logger.warning(f"Login attempt with non-existent email: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Verify password
        stored_hash = getattr(user, "password_hash", None) or getattr(user, "hashed_password", None)
        if not stored_hash:
            logger.warning(f"User record missing password hash for: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if not verify_password(user_data.password, stored_hash):
            logger.warning(f"Failed login attempt for user: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Create JWT
        access_token = create_access_token(
            user_id=str(user.id),
            role=user.role
        )
        
        logger.info(f"User logged in successfully: {user.email}")

        return {
            "access_token": access_token,
            "token_type": "bearer",
        }
    
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.exception("Database unavailable during login")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is temporarily unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected login failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        ) from exc


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(RATE_LIMIT_AUTH)
async def forgot_password(
    request: Request,
    payload: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
):
    """Issue a short-lived one-time reset token without revealing account existence."""
    user = await db.scalar(select(User).where(func.lower(User.email) == payload.email))
    if user:
        token = secrets.token_urlsafe(48)
        reset_record = PasswordResetToken(
            user_id=user.id,
            token_hash=_hash_reset_token(token),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES),
        )
        db.add(reset_record)
        await db.commit()
        try:
            _send_reset_email(user.email, token)
        except Exception:
            await db.delete(reset_record)
            await db.commit()
            logger.exception("Password recovery delivery failed")
    return RESET_RESPONSE


@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(RATE_LIMIT_AUTH)
async def reset_password(
    request: Request,
    payload: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
):
    """Consume a valid reset token and replace the user's password."""
    token_hash = _hash_reset_token(payload.token)
    reset_record = await db.scalar(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    )
    now = datetime.now(timezone.utc)
    if not reset_record or reset_record.used_at or reset_record.expires_at <= now:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

    user = await db.get(User, reset_record.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")
    user.password_hash = hash_password(payload.password)
    reset_record.used_at = now
    await db.commit()