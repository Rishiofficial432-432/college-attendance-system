# models.py ---------------------------------------------------------
from sqlalchemy import (Column, Integer, String, Date, Time,
                        DateTime, Float, LargeBinary,
                        Text, ForeignKey, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    embedding = Column(LargeBinary, nullable=False)   # 512-float32 bytes


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    description = Column(Text)

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
    frame_path = Column(Text, nullable=False)   # relative path under audit folder
    frame_ts = Column(Float, nullable=False)    # seconds from video start
    confidence = Column(Float, nullable=False)
    bbox = Column(Text, nullable=True)          # JSON string "[x,y,w,h]"

    session = relationship("Session", back_populates="logs")
    student = relationship("Student")
