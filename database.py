import sqlite3
import os
from datetime import datetime

class FaceDatabase:
    def __init__(self):
        self.db_name = "face_recognition.db"
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create detected_faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                confidence TEXT NOT NULL,
                detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                detector_type TEXT,
                image_path TEXT
            )
        ''')
        
        # Create known_people table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
                face_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")
    
    def save_detection(self, name, confidence, detector_type="YOLOv8", image_path=None):
        """Save a face detection to database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO detected_faces (name, confidence, detector_type, image_path)
                VALUES (?, ?, ?, ?)
            ''', (name, confidence, detector_type, image_path))
            
            conn.commit()
            print(f"✅ Saved detection: {name} with {confidence} confidence")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
        finally:
            conn.close()
    
    def get_detection_history(self, limit=100):
        """Get detection history"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, confidence, detection_time, detector_type 
            FROM detected_faces 
            ORDER BY detection_time DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries - FIXED FIELD NAMES
        history = []
        for row in results:
            # Format the detection_time for better display
            detection_time = row[2]
            if detection_time:
                try:
                    # Convert to readable format
                    dt = datetime.strptime(detection_time, '%Y-%m-%d %H:%M:%S')
                    formatted_time = dt.strftime('%I:%M %p | %b %d, %Y')  # Example: "08:35 PM | Oct 24, 2024"
                except:
                    formatted_time = detection_time  # Fallback to original if parsing fails
            else:
                formatted_time = "Unknown time"
                
            history.append({
                "name": row[0],
                "confidence": row[1],
                "detection_time": formatted_time,  # CHANGED FROM "time" TO "detection_time"
                "detector_type": row[3]  # CHANGED FROM "detector" TO "detector_type"
            })
        
        return history
    
    def get_statistics(self):
        """Get detection statistics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute('SELECT COUNT(*) FROM detected_faces')
        total_detections = cursor.fetchone()[0]
        
        # Known vs Unknown
        cursor.execute('SELECT COUNT(*) FROM detected_faces WHERE name != "Unknown"')
        known_detections = cursor.fetchone()[0]
        
        # Recent detections (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM detected_faces 
            WHERE detection_time > datetime('now', '-1 day')
        ''')
        recent_detections = cursor.fetchone()[0]
        
        # Top detected person
        cursor.execute('''
            SELECT name, COUNT(*) as count 
            FROM detected_faces 
            WHERE name != "Unknown" 
            GROUP BY name 
            ORDER BY count DESC 
            LIMIT 1
        ''')
        top_person_result = cursor.fetchone()
        top_person = top_person_result[0] if top_person_result else "None"
        top_count = top_person_result[1] if top_person_result else 0
        
        conn.close()
        
        return {
            "total_detections": total_detections,
            "known_detections": known_detections,
            "unknown_detections": total_detections - known_detections,
            "recent_detections": recent_detections,
            "top_person": top_person,
            "top_count": top_count
        }
    
    def clear_history(self):
        """Clear all detection history"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM detected_faces')
        conn.commit()
        conn.close()
        return True