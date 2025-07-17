"""
Command-specific data models for different action types
"""
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime

class MeetingCommand(BaseModel):
    """Meeting scheduling command"""
    title: str
    attendees: List[str] = []
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[int] = None  # minutes
    location: Optional[str] = None
    description: Optional[str] = None
    calendar_id: Optional[str] = "primary"
    meeting_type: str = "meeting"  # meeting, appointment, call, etc.

class EmailCommand(BaseModel):
    """Email sending command"""
    to: List[str]
    cc: List[str] = []
    bcc: List[str] = []
    subject: str
    body: str
    attachments: List[str] = []  # File paths or IDs
    priority: str = "normal"  # low, normal, high
    reply_to: Optional[str] = None
    send_at: Optional[datetime] = None  # Schedule email

class ReminderCommand(BaseModel):
    """Reminder creation command"""
    title: str
    description: Optional[str] = None
    reminder_time: datetime
    repeat: Optional[str] = None  # daily, weekly, monthly
    notification_methods: List[str] = ["email"]
    snooze_options: List[int] = [5, 10, 15]  # minutes
    priority: int = 1  # 1-5

class SearchCommand(BaseModel):
    """Search command for calendar or email"""
    query: str
    search_type: str  # calendar, email, contacts
    date_range: Optional[dict] = None  # {start: date, end: date}
    filters: dict = {}
    max_results: int = 10

class CancelCommand(BaseModel):
    """Cancellation command"""
    item_type: str  # meeting, reminder, email
    item_id: Optional[str] = None
    item_title: Optional[str] = None
    cancel_reason: Optional[str] = None
    notify_attendees: bool = True

class TaskCommand(BaseModel):
    """Task creation/management command"""
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: int = 1  # 1-5
    category: Optional[str] = None
    assignee: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed

class ContactCommand(BaseModel):
    """Contact management command"""
    action: str  # add, update, search, delete
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None


