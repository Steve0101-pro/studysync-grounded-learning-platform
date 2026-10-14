"""Shared Pydantic contracts for application APIs."""

import uuid
from datetime import date, datetime
from typing import Any, Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator
from decimal import Decimal


class BoundingBox(BaseModel):
    """Bounding box coordinates for text elements."""
    left: int = Field(ge=0, description="Left coordinate")
    top: int = Field(ge=0, description="Top coordinate")
    width: int = Field(gt=0, description="Width in pixels")
    height: int = Field(gt=0, description="Height in pixels")


class LayoutElement(BaseModel):
    """Individual extracted layout element."""
    text: str = Field(min_length=1, max_length=5000, description="Extracted text")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score normalized to 0-1")
    page_number: int = Field(ge=1, description="Page number")
    block_number: int = Field(ge=0, description="Block number")
    paragraph_number: int = Field(ge=0, description="Paragraph number")
    line_number: int = Field(ge=0, description="Line number")
    word_number: int = Field(ge=0, description="Word number")
    bounding_box: BoundingBox

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: float) -> float:
        """Accept OCR confidence values in either 0-1 or 0-100 scale."""
        if value is None:
            return value

        numeric_value = float(value)
        if numeric_value > 1.0:
            numeric_value = numeric_value / 100.0
        if numeric_value < 0.0 or numeric_value > 1.0:
            raise ValueError("Confidence must be between 0 and 1 or 0 and 100")
        return numeric_value


class InvoiceData(BaseModel):
    """Structured invoice data extraction."""
    invoice_number: Optional[str] = Field(None, max_length=100, description="Invoice number")
    customer_name: Optional[str] = Field(None, max_length=255, description="Customer name")
    invoice_date: Optional[date] = Field(None, description="Invoice date")
    total_amount: Optional[Decimal] = Field(None, ge=0, description="Total amount")

    @field_validator("invoice_number", "customer_name", "invoice_date", "total_amount", mode="before")
    @classmethod
    def normalize_missing_values(cls, value: Any) -> Any:
        """Convert common model placeholders into actual null values."""
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}):
            return None
        return value


class CombinedLayoutData(BaseModel):
    """Combined layout and invoice data."""
    raw_layout: List[LayoutElement] = Field(default_factory=list, description="Raw layout elements")
    invoice_data: Optional[InvoiceData] = Field(None, description="Structured invoice data")


class DocumentResponse(BaseModel):
    """Document response model."""
    id: uuid.UUID
    filename: str = Field(max_length=255, description="Document filename")
    content_type: str = Field(description="MIME type")
    storage_path: str = Field(description="Storage path in bucket")
    extracted_text: Optional[str] = Field(None, description="Extracted text content")
    layout_data: Optional[CombinedLayoutData] = Field(None, description="Layout and structured data")
    created_at: datetime

    model_config = {"from_attributes": True}


class StudySourceResponse(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str
    page_count: int
    unit_count: int
    created_at: datetime
    source_type: str = "document"
    vector_indexed: bool | None = None
    graph_indexed: bool | None = None
    chunking_strategy: str | None = None
    figure_descriptions: int | None = None


class StudySearchResult(BaseModel):
    source_id: uuid.UUID
    filename: str
    location_type: str
    location_number: int
    chunk_index: int
    score: float
    content: str


class StudySearchResponse(BaseModel):
    query: str
    retrieval: str
    results: list[StudySearchResult]


class StudyTutorCitation(BaseModel):
    source_id: uuid.UUID
    filename: str
    location_type: str
    location_number: int
    content: str


class StudyTutorResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    citations: list[StudyTutorCitation]
    retrieval_backend: str = "lexical"
    generation_provider: str = "source_excerpt"


class StudyQuizCreate(BaseModel):
    topic: str = Field(min_length=2, max_length=255)


class StudyQuizOption(BaseModel):
    content: str
    filename: str
    source_id: uuid.UUID
    location_type: str
    location_number: int


class StudyQuizResponse(BaseModel):
    id: uuid.UUID
    topic: str
    prompt: str
    options: list[StudyQuizOption]
    answered_index: int | None = None
    correct: bool | None = None
    explanation: StudyQuizOption | None = None


class StudyQuizAnswer(BaseModel):
    selected_index: int = Field(ge=0, le=3)


class StudyProgressTopic(BaseModel):
    topic: str
    answered: int
    correct: int
    accuracy: float
    mastery_estimate: float
    recommended_review_in_days: int | None


class StudyProgressResponse(BaseModel):
    total_sources: int
    total_questions: int
    answered_questions: int
    correct_answers: int
    practice_accuracy: float
    topics: list[StudyProgressTopic]


# ============================================================
# AUTHENTICATION SCHEMAS
# ============================================================

class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr = Field(description="User email address")
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Password (min 8 characters, should include uppercase, lowercase, numbers)"
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets complexity requirements."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr = Field(description="User email address")
    password: str = Field(min_length=1, max_length=128, description="User password")

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().casefold() if isinstance(value, str) else value


class PasswordResetRequest(BaseModel):
    email: EmailStr

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().casefold() if isinstance(value, str) else value


class PasswordResetConfirm(BaseModel):
    token: str = Field(min_length=32, max_length=256)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        if not any(char.isupper() for char in value):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in value):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(char.isdigit() for char in value):
            raise ValueError("Password must contain at least one digit")
        return value


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")


class UserResponse(BaseModel):
    """User public response model."""
    id: uuid.UUID
    email: str = Field(description="User email")
    role: str = Field(description="User role")

    model_config = {"from_attributes": True}


# ============================================================
# MULTIMODAL SCHEMAS
# ============================================================

class MultimodalResponse(BaseModel):
    """Multimodal processing response."""
    conversation_id: uuid.UUID = Field(description="Conversation session ID")
    transcript: str = Field(min_length=1, max_length=50000, description="Speech-to-text transcript")
    response_text: str = Field(min_length=1, max_length=50000, description="AI response text")
    audio_storage_path: str = Field(description="Generated audio storage path")
    audio_content_type: str = Field(default="audio/wav", description="Audio MIME type")
    audio_base64: str = Field(description="Base64-encoded playable audio")
    experiment_variant: str = Field(default="control", max_length=80)


class FeedbackCreate(BaseModel):
    conversation_id: uuid.UUID
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=5000)
    model_version: str = Field(default="unknown", max_length=150)
    prompt_version: str = Field(default="unknown", max_length=80)


class FeedbackResponse(FeedbackCreate):
    id: uuid.UUID
    review_status: str
    reviewer_notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackReviewUpdate(BaseModel):
    review_status: str = Field(pattern="^(pending|approved|rejected|escalated)$")
    reviewer_notes: Optional[str] = Field(default=None, max_length=5000)


class ConversationResponse(BaseModel):
    """Conversation session response."""
    id: uuid.UUID
    user_id: uuid.UUID
    title: str = Field(max_length=255, description="Conversation title")
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    """Message response model."""
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(min_length=1, max_length=50000, description="Message content")
    created_at: datetime

    model_config = {"from_attributes": True}


# ============================================================
# ERROR RESPONSE SCHEMAS
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for debugging")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    detail: List[dict] = Field(description="Validation error details")
    error_code: str = Field(default="VALIDATION_ERROR", description="Error code")