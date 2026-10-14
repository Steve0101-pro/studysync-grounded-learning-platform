"""SQLAlchemy models used by the application."""

import uuid

from sqlalchemy import Column, Index
from sqlalchemy import DateTime
from sqlalchemy import String
from sqlalchemy import Text, ForeignKey, Integer, UniqueConstraint
from sqlalchemy import func, text
from sqlalchemy.orm import MappedColumn,mapped_column,relationship
from sqlalchemy.dialects.postgresql import UUID,JSONB
from datetime import datetime
from app.database import Base
from typing import Any, Optional



class User(Base):

    __tablename__ = "users"
    __table_args__ = (Index("uq_users_email_lower", func.lower(text("email")), unique=True),)

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4,)

    email: MappedColumn[str] = mapped_column(String(255),unique=True,index=True,nullable=False,)

    password_hash: MappedColumn[str] = mapped_column(String(255),nullable=False,)

    role: MappedColumn[str] = mapped_column(String(50),default="user",nullable=False,)

    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True),server_default=func.now(),nullable=False,)

    documents = relationship("Document",back_populates="owner",)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    token_hash: MappedColumn[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    expires_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: MappedColumn[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)







class Document(Base):

    __tablename__ = "documents"

    id : MappedColumn[uuid.UUID] = Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4,index=True)

    user_id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)  # ADD THIS
    
    filename:MappedColumn[str] = mapped_column(String(255),nullable=False,)

    content_type: MappedColumn[str] = mapped_column(String(100),nullable=False,)

    layout_data: MappedColumn[Optional[dict[str,Any]]] = mapped_column(JSONB,nullable=True,)
     
    storage_path: MappedColumn[str] = mapped_column(String(500),nullable=False,)

    extracted_text: MappedColumn[str] = mapped_column(Text,nullable=True,)

    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True),server_default=func.now(),nullable=False,)

    owner = relationship("User",back_populates="documents",)


class Conversation(Base):

    __tablename__ = "conversations"

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4,)

    user_id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("users.id"),nullable=False,)

    title: MappedColumn[str | None] = mapped_column(String(255),nullable=True,)

    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),server_default=func.now(),nullable=False,)


class Message(Base):

    __tablename__ = "messages"

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4,)

    conversation_id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("conversations.id"),nullable=False,index=True,)

    role: MappedColumn[str] = mapped_column(String(50),nullable=False,)

    content: MappedColumn[str] = mapped_column(Text,nullable=False,)

    metadata_json: MappedColumn[dict | None] = mapped_column(JSONB,nullable=True,)

    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True),server_default=func.now(),nullable=False,)


class ResponseFeedback(Base):
    __tablename__ = "response_feedback"

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    rating: MappedColumn[int | None] = mapped_column(nullable=True)
    comment: MappedColumn[str | None] = mapped_column(Text, nullable=True)
    review_status: MappedColumn[str] = mapped_column(String(30), nullable=False, server_default="pending")
    reviewer_notes: MappedColumn[str | None] = mapped_column(Text, nullable=True)
    model_version: MappedColumn[str] = mapped_column(String(150), nullable=False)
    prompt_version: MappedColumn[str] = mapped_column(String(80), nullable=False)
    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id: MappedColumn[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    action: MappedColumn[str] = mapped_column(String(80), nullable=False)
    resource_type: MappedColumn[str] = mapped_column(String(80), nullable=False)
    resource_id: MappedColumn[str | None] = mapped_column(String(150), nullable=True)
    metadata_json: MappedColumn[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


class TTSRecord(Base):
    __tablename__ = "tts_records"
    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)    
    text: MappedColumn[str] = mapped_column(Text, nullable=False)    
    storage_path: MappedColumn[str] = mapped_column(String(500), nullable=False)
    reference_filename: MappedColumn[str] = mapped_column(String(255), nullable=False)
    created_at: MappedColumn[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class StudySource(Base):
    __tablename__ = "study_sources"
    __table_args__ = (
        UniqueConstraint("user_id", "content_sha256", name="uq_study_sources_user_checksum"),
    )

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: MappedColumn[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    filename: MappedColumn[str] = mapped_column(String(255), nullable=False)
    content_type: MappedColumn[str] = mapped_column(String(100), nullable=False)
    storage_path: MappedColumn[str] = mapped_column(String(500), nullable=False)
    content_sha256: MappedColumn[str] = mapped_column(String(64), nullable=False)
    page_count: MappedColumn[int] = mapped_column(Integer, nullable=False)
    unit_count: MappedColumn[int] = mapped_column(Integer, nullable=False)
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    content_units = relationship(
        "StudyContentUnit",
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class StudyContentUnit(Base):
    __tablename__ = "study_content_units"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "location_type",
            "location_number",
            "chunk_level",
            "chunk_index",
            name="uq_study_content_location_chunk",
        ),
    )

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: MappedColumn[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("study_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    location_type: MappedColumn[str] = mapped_column(String(16), nullable=False)
    location_number: MappedColumn[int] = mapped_column(Integer, nullable=False)
    chunk_level: MappedColumn[str] = mapped_column(String(16), nullable=False, default="child")
    chunk_index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    parent_unit_id: MappedColumn[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("study_content_units.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    char_start: MappedColumn[int | None] = mapped_column(Integer, nullable=True)
    char_end: MappedColumn[int | None] = mapped_column(Integer, nullable=True)
    content: MappedColumn[str] = mapped_column(Text, nullable=False)

    source = relationship("StudySource", back_populates="content_units")


class StudyQuizQuestion(Base):
    __tablename__ = "study_quiz_questions"

    id: MappedColumn[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: MappedColumn[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic: MappedColumn[str] = mapped_column(String(255), nullable=False, index=True)
    prompt: MappedColumn[str] = mapped_column(Text, nullable=False)
    options: MappedColumn[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    answer_index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    source_id: MappedColumn[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("study_sources.id", ondelete="CASCADE"), nullable=False
    )
    location_type: MappedColumn[str] = mapped_column(String(16), nullable=False)
    location_number: MappedColumn[int] = mapped_column(Integer, nullable=False)
    deduplication_key: MappedColumn[str | None] = mapped_column(String(64), nullable=True)
    answered_index: MappedColumn[int | None] = mapped_column(Integer, nullable=True)
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    answered_at: MappedColumn[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
