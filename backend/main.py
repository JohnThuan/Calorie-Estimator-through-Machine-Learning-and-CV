"""
NutriLens FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import json
import os

from services.image_processor import ImageProcessor
from services.food_recognizer_efficientnet_b4 import FoodRecognizerEfficientNetB4
from services.calorie_calculator import CalorieCalculator
from services.baseline_manager import BaselineManager
from models.response_models import AnalysisResponse, BaselineResponse, HistoryResponse

app = FastAPI(title="NutriLens API", version="1.0.0")

# CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
image_processor = ImageProcessor()
food_recognizer = FoodRecognizerEfficientNetB4()
calorie_calculator = CalorieCalculator()
baseline_manager = BaselineManager()

# Ensure data directories exist
os.makedirs("data/history", exist_ok=True)
os.makedirs("data/baselines", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NutriLens API",
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    user_id: str = "default"
):
    """
    Analyze uploaded meal image and return calorie estimation
    
    - Processes image
    - Recognizes food items
    - Estimates portions
    - Calculates calorie ranges
    """
    try:
        print(f"[DEBUG] Received file: {file.filename}, content_type: {file.content_type}")
        
        # Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await file.read()
        print(f"[DEBUG] Image bytes length: {len(image_bytes)}")
        
        processed_image = image_processor.process_image(image_bytes)
        print(f"[DEBUG] Image processed successfully")
        
        # Recognize food items (top-K predictions)
        food_predictions = food_recognizer.recognize(processed_image)
        
        # Estimate portion sizes
        portions = image_processor.estimate_portions(processed_image, food_predictions)
        
        # Calculate calorie ranges
        calorie_data = calorie_calculator.calculate_ranges(food_predictions, portions)
        
        # Save to history
        analysis_id = f"{user_id}_{datetime.now().timestamp()}"
        history_entry = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "foods": food_predictions,
            "portions": portions,
            "calories": calorie_data,
            "image_path": f"data/uploads/{analysis_id}.jpg"
        }
        
        # Save image
        with open(history_entry["image_path"], "wb") as f:
            f.write(image_bytes)
        
        # Save history entry
        history_file = f"data/history/{user_id}.json"
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        history.append(history_entry)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=history_entry["timestamp"],
            foods=food_predictions,
            calories_min=calorie_data["min"],
            calories_max=calorie_data["max"],
            calories_estimate=calorie_data["estimate"],
            confidence=calorie_data["confidence"],
            portions=portions,
            nutrients=calorie_data.get("nutrients", {})
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] Analysis failed: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


class BaselineSetRequest(BaseModel):
    user_id: str
    analysis_id: str
    actual_calories: Optional[int] = None
    notes: Optional[str] = None


@app.post("/baseline/set", response_model=BaselineResponse)
async def set_baseline(request: BaselineSetRequest):
    """
    Save a meal analysis as baseline calibration data
    """
    try:
        baseline = baseline_manager.set_baseline(
            request.user_id,
            request.analysis_id,
            request.actual_calories,
            request.notes
        )
        return BaselineResponse(**baseline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set baseline: {str(e)}")


class BaselineCompareRequest(BaseModel):
    user_id: str
    analysis_id: str


@app.post("/baseline/compare")
async def compare_to_baseline(request: BaselineCompareRequest):
    """
    Compare a meal analysis to saved baseline
    """
    try:
        comparison = baseline_manager.compare_to_baseline(
            request.user_id,
            request.analysis_id
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/history", response_model=HistoryResponse)
async def get_history(user_id: str = "default", limit: int = 50):
    """
    Retrieve analysis history for a user
    """
    try:
        history_file = f"data/history/{user_id}.json"
        if not os.path.exists(history_file):
            return HistoryResponse(user_id=user_id, entries=[])
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Return most recent entries
        history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        return HistoryResponse(
            user_id=user_id,
            entries=history,
            total_count=len(history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/history/export")
async def export_history(user_id: str = "default", format: str = "json"):
    """
    Export history in JSON or CSV format
    """
    try:
        history_file = f"data/history/{user_id}.json"
        if not os.path.exists(history_file):
            raise HTTPException(status_code=404, detail="No history found")
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        if format == "csv":
            # Convert to CSV
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Timestamp", "Foods", "Calories (Est)", "Confidence"])
            for entry in history:
                writer.writerow([
                    entry["timestamp"],
                    ", ".join([f["name"] for f in entry["foods"]]),
                    entry["calories"]["estimate"],
                    entry["calories"]["confidence"]
                ])
            return {"content": output.getvalue(), "format": "csv"}
        
        return {"content": history, "format": "json"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


