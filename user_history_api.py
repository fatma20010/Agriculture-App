from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import json
import logging

# Import your User and AnalysisRecord models and get_db, get_current_user dependencies from your main app
from models_and_deps import AnalysisRecord, get_db, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get('/user/history')
async def get_user_history(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all analysis records for the current user."""
    records = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == current_user.id).order_by(AnalysisRecord.created_at.desc()).all()
    results = []
    for record in records:
        try:
            data = json.loads(str(record.result_data))
            summary = {
                "species": data.get("species", "Unknown"),
                "confidence": data.get("confidence", 0),
                "yellow_percentage": data.get("yellow_percentage", 0),
                "spot_count": data.get("spot_count", 0),
                "estimated_compost_grams": data.get("estimated_compost_grams", 0)
            }
            results.append({
                "id": record.id,
                "analysis_type": record.analysis_type,
                "result_summary": summary,
                "created_at": record.created_at
            })
        except Exception as e:
            logger.error(f"Error processing record {record.id}: {str(e)}")
    return results
