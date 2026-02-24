"""
Database models representing courses, students, sessions, and attendance.
"""
from sqlalchemy import (Column, Integer, String, Date, Time,
                        DateTime, Float, LargeBinary,
                        Text, ForeignKey, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Course(Base):
    """A course containing multiple class sessions."""
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)

    sessions = relationship("Session", back_populates="course", 
                           cascade="all, delete-orphan")


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    embedding = Column(LargeBinary, nullable=False)   # 512-float32 bytes
    photo_path = Column(Text, nullable=True)          # Path to student folder or photo


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    description = Column(Text)
    frames_deleted = Column(Integer, default=0)       # 0 = not yet, 1 = deleted

    course = relationship("Course", back_populates="sessions")
    attendances = relationship("Attendance", back_populates="session",
                               cascade="all, delete-orphan")
    logs = relationship("AttendanceLog", back_populates="session",
                        cascade="all, delete-orphan")


class Attendance(Base):
    __tablename__ = "attendance"
    session_id = Column(Integer, ForeignKey("sessions.id"), primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), primary_key=True)
    first_seen = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)

    session = relationship("Session", back_populates="attendances")
    student = relationship("Student")


class AttendanceLog(Base):
    __tablename__ = "attendance_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    frame_path = Column(Text, nullable=True)   # nullable for when frames are deleted
    frame_ts = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox = Column(Text, nullable=True)

    session = relationship("Session", back_populates="logs")
    student = relationship("Student")

