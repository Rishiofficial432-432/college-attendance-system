import pandas as pd
import os
from datetime import datetime
import time

class AttendanceLogger:
    def __init__(self, file_path='attendance.csv', cooldown=60):
        """
        Initialize Attendance Logger.
        
        Args:
            file_path (str): Path to CSV file.
            cooldown (int): Seconds to wait before logging the same person again.
        """
        self.file_path = file_path
        self.cooldown = cooldown
        self.last_logged = {} # {name: timestamp}
        
        # Create file if not exists
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
            df.to_csv(self.file_path, index=False)

    def mark(self, name):
        """
        Mark attendance for a person.
        
        Args:
            name (str): Name of the person.
            
        Returns:
            bool: True if marked, False if cooldown active.
        """
        now = time.time()
        
        if name in self.last_logged:
            elapsed = now - self.last_logged[name]
            if elapsed < self.cooldown:
                return False
        
        self.last_logged[name] = now
        
        # Log to CSV
        dt = datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")
        
        new_record = pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})
        new_record.to_csv(self.file_path, mode='a', header=False, index=False)
        
        print(f"[ATTENDANCE] Marked: {name} at {time_str}")
        return True
