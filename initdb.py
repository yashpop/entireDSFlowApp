from app import app, db 
from app.models import Experiment

db.create_all()
print("Database initialized")