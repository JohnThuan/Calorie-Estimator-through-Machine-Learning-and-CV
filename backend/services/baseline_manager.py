"""
Baseline management service
Handles baseline calibration and comparison
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


class BaselineManager:
    def __init__(self):
        self.baseline_dir = "data/baselines"
        os.makedirs(self.baseline_dir, exist_ok=True)
    
    def set_baseline(
        self,
        user_id: str,
        analysis_id: str,
        actual_calories: Optional[int],
        notes: Optional[str]
    ) -> Dict:
        """
        Save a meal analysis as baseline calibration reference
        """
        # Load the analysis from history
        history_file = f"data/history/{user_id}.json"
        if not os.path.exists(history_file):
            raise ValueError("Analysis not found in history")
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Find the specific analysis
        analysis = None
        for entry in history:
            if entry["id"] == analysis_id:
                analysis = entry
                break
        
        if not analysis:
            raise ValueError("Analysis ID not found")
        
        # Create baseline entry
        baseline_id = f"baseline_{user_id}_{datetime.now().timestamp()}"
        baseline = {
            "baseline_id": baseline_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "reference_analysis_id": analysis_id,
            "estimated_calories": analysis["calories"]["estimate"],
            "actual_calories": actual_calories,
            "foods": analysis["foods"],
            "portions": analysis["portions"],
            "notes": notes
        }
        
        # Save baseline
        baseline_file = f"{self.baseline_dir}/{user_id}.json"
        baselines = []
        if os.path.exists(baseline_file):
            with open(baseline_file, "r") as f:
                baselines = json.load(f)
        
        baselines.append(baseline)
        
        with open(baseline_file, "w") as f:
            json.dump(baselines, f, indent=2)
        
        return baseline
    
    def compare_to_baseline(self, user_id: str, analysis_id: str) -> Dict:
        """
        Compare a meal analysis to user's baseline data
        """
        # Load baselines
        baseline_file = f"{self.baseline_dir}/{user_id}.json"
        if not os.path.exists(baseline_file):
            return {
                "has_baseline": False,
                "message": "No baseline data available. Set a baseline first!"
            }
        
        with open(baseline_file, "r") as f:
            baselines = json.load(f)
        
        # Load current analysis
        history_file = f"data/history/{user_id}.json"
        with open(history_file, "r") as f:
            history = json.load(f)
        
        current_analysis = None
        for entry in history:
            if entry["id"] == analysis_id:
                current_analysis = entry
                break
        
        if not current_analysis:
            raise ValueError("Analysis not found")
        
        # Find similar baselines (same food types)
        similar_baselines = self._find_similar_baselines(
            current_analysis,
            baselines
        )
        
        if not similar_baselines:
            return {
                "has_baseline": True,
                "similar_meals": 0,
                "message": "No similar baseline meals found"
            }
        
        # Calculate comparison metrics
        current_estimate = current_analysis["calories"]["estimate"]
        baseline_estimates = [b["estimated_calories"] for b in similar_baselines]
        baseline_actuals = [b["actual_calories"] for b in similar_baselines if b["actual_calories"]]
        
        avg_baseline = sum(baseline_estimates) / len(baseline_estimates)
        
        comparison = {
            "has_baseline": True,
            "similar_meals": len(similar_baselines),
            "current_estimate": current_estimate,
            "baseline_avg_estimate": int(avg_baseline),
            "difference": int(current_estimate - avg_baseline),
            "difference_percent": round((current_estimate - avg_baseline) / avg_baseline * 100, 1),
        }
        
        if baseline_actuals:
            avg_actual = sum(baseline_actuals) / len(baseline_actuals)
            comparison["baseline_avg_actual"] = int(avg_actual)
            comparison["typical_error"] = int(avg_baseline - avg_actual)
        
        return comparison
    
    def _find_similar_baselines(self, analysis: Dict, baselines: list) -> list:
        """
        Find baselines with similar food items
        """
        similar = []
        current_foods = set(f["name"] for f in analysis["foods"])
        
        for baseline in baselines:
            baseline_foods = set(f["name"] for f in baseline["foods"])
            
            # Calculate food overlap
            overlap = len(current_foods & baseline_foods)
            if overlap > 0:
                similar.append(baseline)
        
        return similar


