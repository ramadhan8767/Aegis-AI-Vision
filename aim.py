"""
Advanced AI-Powered Image Analysis System
Educational Purpose Only - Not for Cheating in Games
Developed for learning Computer Vision and AI Ethics
"""

import json
import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict, deque
import pickle
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA STRUCTURES ====================

@dataclass
class DetectionPattern:
    """Data structure for detection patterns"""
    pattern_id: str
    color_range: List[List[int]]
    shape_type: str
    size_range: Tuple[int, int]
    success_rate: float
    detection_count: int
    first_detected: str
    last_updated: str
    features: Dict[str, Any]
    confidence_score: float

@dataclass
class EnemyProfile:
    """Data structure for enemy profiles"""
    enemy_id: str
    name: str
    detection_patterns: List[str]
    movement_patterns: Dict[str, List[Tuple[int, int]]]
    encounter_timestamps: List[str]
    kill_count: int
    death_count: int
    avg_engagement_distance: float
    behavior_score: float
    first_seen: str
    last_seen: str

@dataclass
class GameSession:
    """Data structure for game sessions"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    maps_played: List[str]
    performance_metrics: Dict[str, float]
    ethical_reminders_shown: int
    analysis_count: int

# ==================== ADVANCED AI SYSTEM ====================

class AdvancedAIAimAnalyzer:
    """
    Advanced AI-Powered Image Analysis System
    For Educational and Research Purposes Only
    """
    
    def __init__(self, config_file: str = "ai_aim_analyzer_config.json"):
        self.config_file = config_file
        self.patterns_file = "ai_learning_patterns.json"
        self.profiles_file = "enemy_behavior_profiles.json"
        self.sessions_file = "game_sessions_history.json"
        self.model_file = "ai_aim_model.pkl"
        self.history_file = "analysis_history.json"
        
        # Initialize data stores
        self.detection_patterns = {}
        self.enemy_profiles = {}
        self.game_sessions = {}
        self.current_session = None
        self.analysis_history = deque(maxlen=1000)
        
        # AI Models
        self.kmeans_model = None
        self.classifier_model = None
        self.feature_scaler = None
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_detections = 0
        self.learning_rate = 0.15
        self.confidence_threshold = 0.75
        
        # Initialize the system
        self.initialize_ai_system()
        
    # ==================== INITIALIZATION ====================
    
    def initialize_ai_system(self):
        """Initialize the AI system with all components"""
        print("🤖 Initializing Advanced AI Analysis System...")
        print("=" * 70)
        
        self.load_configuration()
        self.load_ai_patterns()
        self.load_behavior_profiles()
        self.load_game_sessions()
        self.load_ai_models()
        
        self.display_system_banner()
        self.start_new_session()
        
    def load_configuration(self):
        """Load system configuration from JSON"""
        default_config = {
            "system_info": {
                "name": "Advanced AI Aim Analyzer - Educational Version",
                "version": "3.0.0",
                "developer": "AI Research Team",
                "purpose": "Educational & Research Only",
                "license": "Non-Commercial, Educational Use"
            },
            
            "ai_settings": {
                "learning_mode": "adaptive",
                "max_patterns": 200,
                "pattern_decay_rate": 0.95,
                "min_confidence_threshold": 0.65,
                "adaptive_learning_rate": True,
                "enable_neural_patterns": True,
                "real_time_processing": False,
                "cache_size": 1000
            },
            
            "detection_settings": {
                "color_profiles": {
                    "player_red": {"hsv_range": [[0, 100, 100], [10, 255, 255]], "priority": 1},
                    "player_blue": {"hsv_range": [[100, 100, 100], [130, 255, 255]], "priority": 2},
                    "critical_area": {"hsv_range": [[0, 150, 150], [20, 255, 255]], "priority": 3},
                    "environment": {"hsv_range": [[40, 40, 40], [80, 255, 255]], "priority": 0}
                },
                "min_contour_area": 75,
                "max_contour_area": 1000,
                "contour_approximation": 0.03,
                "morphology_kernel": 5,
                "edge_detection": True
            },
            
            "performance_settings": {
                "analysis_interval": 0.1,
                "max_history_frames": 300,
                "auto_save_interval": 30,
                "visualization_enabled": True,
                "debug_mode": False
            },
            
            "ethical_guidelines": {
                "primary_rule": "This system is for educational purposes only",
                "rules": [
                    "Never use AI to cheat in competitive games",
                    "Respect game developers' terms of service",
                    "Maintain fair play for all participants",
                    "Use knowledge for positive applications",
                    "Report vulnerabilities, don't exploit them"
                ],
                "consequences": [
                    "Game account banning",
                    "Legal repercussions",
                    "Loss of credibility",
                    "Damage to gaming community"
                ]
            },
            
            "feature_flags": {
                "enable_pattern_learning": True,
                "enable_profile_tracking": True,
                "enable_performance_analytics": True,
                "enable_ethical_monitoring": True,
                "enable_ai_insights": True
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print("✅ Configuration loaded successfully")
            else:
                self.config = default_config
                self.save_configuration()
                print("🆕 New configuration created")
                
        except Exception as e:
            print(f"❌ Configuration loading error: {e}")
            self.config = default_config
            
    def save_configuration(self):
        """Save system configuration to JSON"""
        try:
            self.config['system_info']['last_modified'] = datetime.now().isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print("💾 Configuration saved")
        except Exception as e:
            print(f"❌ Configuration save error: {e}")
    
    # ==================== DATA MANAGEMENT ====================
    
    def load_ai_patterns(self):
        """Load AI learning patterns from JSON"""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                    self.detection_patterns = {
                        k: DetectionPattern(**v) for k, v in patterns_data.items()
                    }
                print(f"📊 Loaded {len(self.detection_patterns)} learning patterns")
            else:
                print("📝 No existing patterns found")
        except Exception as e:
            print(f"❌ Pattern loading error: {e}")
            self.detection_patterns = {}
    
    def save_ai_patterns(self):
        """Save AI learning patterns to JSON"""
        try:
            patterns_dict = {k: asdict(v) for k, v in self.detection_patterns.items()}
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_dict, f, indent=4, ensure_ascii=False)
            print(f"💾 Saved {len(self.detection_patterns)} patterns")
        except Exception as e:
            print(f"❌ Pattern save error: {e}")
    
    def load_behavior_profiles(self):
        """Load enemy behavior profiles from JSON"""
        try:
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    self.enemy_profiles = {
                        k: EnemyProfile(**v) for k, v in profiles_data.items()
                    }
                print(f"🎯 Loaded {len(self.enemy_profiles)} behavior profiles")
            else:
                print("👤 No existing profiles found")
        except Exception as e:
            print(f"❌ Profile loading error: {e}")
            self.enemy_profiles = {}
    
    def save_behavior_profiles(self):
        """Save enemy behavior profiles to JSON"""
        try:
            profiles_dict = {k: asdict(v) for k, v in self.enemy_profiles.items()}
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_dict, f, indent=4, ensure_ascii=False)
            print(f"💾 Saved {len(self.enemy_profiles)} profiles")
        except Exception as e:
            print(f"❌ Profile save error: {e}")
    
    def load_game_sessions(self):
        """Load game sessions history from JSON"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    self.game_sessions = json.load(f)
                print(f"📅 Loaded {len(self.game_sessions)} game sessions")
            else:
                print("🕒 No session history found")
        except Exception as e:
            print(f"❌ Session loading error: {e}")
            self.game_sessions = {}
    
    def save_game_sessions(self):
        """Save game sessions to JSON"""
        try:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.game_sessions, f, indent=4, ensure_ascii=False)
            print(f"💾 Saved {len(self.game_sessions)} sessions")
        except Exception as e:
            print(f"❌ Session save error: {e}")
    
    def load_ai_models(self):
        """Load trained AI models from disk"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    models = pickle.load(f)
                    self.kmeans_model = models.get('kmeans')
                    self.classifier_model = models.get('classifier')
                    self.feature_scaler = models.get('scaler')
                print("🧠 AI models loaded successfully")
            else:
                print("🤖 No trained models found")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
    
    def save_ai_models(self):
        """Save AI models to disk"""
        try:
            models = {
                'kmeans': self.kmeans_model,
                'classifier': self.classifier_model,
                'scaler': self.feature_scaler
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(models, f)
            print("💾 AI models saved")
        except Exception as e:
            print(f"❌ Model save error: {e}")
    
    # ==================== SESSION MANAGEMENT ====================
    
    def start_new_session(self, map_name: str = "training_ground"):
        """Start a new analysis session"""
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]
        self.current_session = GameSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            maps_played=[map_name],
            performance_metrics={
                "total_analyses": 0,
                "detection_accuracy": 0.0,
                "average_confidence": 0.0,
                "patterns_learned": 0,
                "ethical_checks_passed": 0
            },
            ethical_reminders_shown=0,
            analysis_count=0
        )
        print(f"🚀 New session started: {session_id}")
        print(f"📍 Map: {map_name}")
        
        self.display_ethical_warning()
    
    def end_current_session(self):
        """End the current analysis session"""
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()
            self.current_session.performance_metrics["patterns_learned"] = len(self.detection_patterns)
            
            # Calculate session performance
            if self.current_session.analysis_count > 0:
                accuracy = self.successful_detections / self.total_analyses
                self.current_session.performance_metrics["detection_accuracy"] = accuracy
            
            # Save session
            self.game_sessions[self.current_session.session_id] = asdict(self.current_session)
            self.save_game_sessions()
            
            print(f"🛑 Session {self.current_session.session_id} ended")
            print(f"📊 Performance: {self.current_session.performance_metrics}")
            
            self.current_session = None
    
    # ==================== CORE AI ANALYSIS ====================
    
    def analyze_image(self, image_path: str, real_time: bool = False) -> Dict[str, Any]:
        """
        Advanced AI-powered image analysis
        Returns comprehensive analysis results
        """
        self.total_analyses += 1
        
        if self.current_session:
            self.current_session.analysis_count += 1
        
        print(f"\n{'='*60}")
        print(f"🔍 AI Analysis #{self.total_analyses}")
        print(f"{'='*60}")
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return {"error": "Image loading failed"}
        
        # Start analysis timer
        start_time = time.time()
        
        # Multi-stage analysis pipeline
        analysis_results = {
            "metadata": {
                "image_path": image_path,
                "image_size": image.shape,
                "analysis_timestamp": datetime.now().isoformat(),
                "session_id": self.current_session.session_id if self.current_session else None,
                "real_time_mode": real_time
            },
            "stage_results": {},
            "detections": [],
            "ai_insights": [],
            "performance_metrics": {},
            "ethical_check": {}
        }
        
        # Stage 1: Multi-method detection
        stage1_start = time.time()
        detections = self.multi_method_detection(image)
        analysis_results["stage_results"]["multi_method_detection"] = {
            "time_ms": (time.time() - stage1_start) * 1000,
            "detections_found": len(detections)
        }
        analysis_results["detections"] = detections
        
        # Stage 2: Pattern matching and learning
        if detections and self.config["feature_flags"]["enable_pattern_learning"]:
            stage2_start = time.time()
            pattern_matches = self.pattern_matching_analysis(detections, image)
            analysis_results["stage_results"]["pattern_matching"] = {
                "time_ms": (time.time() - stage2_start) * 1000,
                "patterns_matched": len(pattern_matches)
            }
            analysis_results["pattern_matches"] = pattern_matches
            
            # Learn from successful detections
            self.adaptive_learning(detections, image)
        
        # Stage 3: Behavioral analysis
        if self.config["feature_flags"]["enable_profile_tracking"]:
            stage3_start = time.time()
            behavioral_insights = self.behavioral_analysis(detections)
            analysis_results["stage_results"]["behavioral_analysis"] = {
                "time_ms": (time.time() - stage3_start) * 1000,
                "insights_generated": len(behavioral_insights)
            }
            analysis_results["behavioral_insights"] = behavioral_insights
        
        # Stage 4: AI Insights generation
        if self.config["feature_flags"]["enable_ai_insights"]:
            ai_insights = self.generate_ai_insights(analysis_results)
            analysis_results["ai_insights"] = ai_insights
        
        # Stage 5: Performance metrics
        analysis_results["performance_metrics"] = self.calculate_performance_metrics(
            detections, 
            time.time() - start_time
        )
        
        # Stage 6: Ethical compliance check
        analysis_results["ethical_check"] = self.perform_ethical_check(analysis_results)
        
        # Save results to history
        self.save_analysis_results(analysis_results)
        
        # Display visualization if enabled
        if self.config["performance_settings"]["visualization_enabled"]:
            self.display_advanced_visualization(image, analysis_results)
        
        print(f"✅ Analysis completed in {(time.time() - start_time)*1000:.1f}ms")
        print(f"🎯 Detections: {len(detections)} | Confidence: {analysis_results['performance_metrics'].get('avg_confidence', 0):.1%}")
        
        return analysis_results
    
    def multi_method_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Combine multiple detection methods for robust results"""
        detections = []
        
        # Method 1: Color-based detection (HSV)
        color_detections = self.color_based_detection(image)
        detections.extend(color_detections)
        
        # Method 2: Edge-based detection
        edge_detections = self.edge_based_detection(image)
        detections.extend(edge_detections)
        
        # Method 3: Template matching (if patterns exist)
        if self.detection_patterns:
            template_detections = self.template_matching_detection(image)
            detections.extend(template_detections)
        
        # Remove duplicates and low-confidence detections
        detections = self.filter_and_merge_detections(detections)
        
        return detections
    
    def color_based_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Advanced color-based object detection"""
        detections = []
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_profiles = self.config["detection_settings"]["color_profiles"]
        
        for profile_name, profile_config in color_profiles.items():
            if profile_config.get("priority", 0) > 0:  # Only detect priority objects
                hsv_range = profile_config["hsv_range"]
                min_hsv = np.array(hsv_range[0])
                max_hsv = np.array(hsv_range[1])
                
                mask = cv2.inRange(hsv_image, min_hsv, max_hsv)
                
                # Apply morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    min_area = self.config["detection_settings"]["min_contour_area"]
                    max_area = self.config["detection_settings"]["max_contour_area"]
                    
                    if min_area < area < max_area:
                        # Get bounding box and center
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Calculate shape features
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        
                        detection = {
                            "method": "color_based",
                            "profile": profile_name,
                            "position": {"x": center_x, "y": center_y},
                            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                            "area": area,
                            "circularity": circularity,
                            "aspect_ratio": w / h if h > 0 else 0,
                            "confidence": 0.7,  # Base confidence
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Adjust confidence based on features
                        if 0.7 < circularity < 1.3:  # Near-circular objects
                            detection["confidence"] *= 1.2
                        
                        detections.append(detection)
        
        return detections
    
    def edge_based_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Edge-based detection for shape recognition"""
        if not self.config["detection_settings"]["edge_detection"]:
            return []
        
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = self.config["detection_settings"]["min_contour_area"]
            
            if area > min_area:
                # Approximate the contour
                epsilon = self.config["detection_settings"]["contour_approximation"] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Determine shape type
                shape_type = self.determine_shape_type(len(approx), area, w/h if h > 0 else 0)
                
                detection = {
                    "method": "edge_based",
                    "shape_type": shape_type,
                    "position": {"x": center_x, "y": center_y},
                    "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                    "area": area,
                    "vertices": len(approx),
                    "confidence": 0.6,
                    "timestamp": datetime.now().isoformat()
                }
                
                detections.append(detection)
        
        return detections
    
    def template_matching_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Template matching using learned patterns"""
        detections = []
        
        # Convert to HSV for better matching
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for pattern_id, pattern in self.detection_patterns.items():
            if pattern.confidence_score > self.confidence_threshold:
                # Create mask for pattern's color range
                min_hsv = np.array(pattern.color_range[0])
                max_hsv = np.array(pattern.color_range[1])
                mask = cv2.inRange(hsv_image, min_hsv, max_hsv)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if pattern.size_range[0] < area < pattern.size_range[1]:
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Calculate match score
                        match_score = self.calculate_pattern_match_score(contour, pattern)
                        
                        if match_score > 0.6:
                            detection = {
                                "method": "template_matching",
                                "pattern_id": pattern_id,
                                "shape_type": pattern.shape_type,
                                "position": {"x": center_x, "y": center_y},
                                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                                "area": area,
                                "match_score": match_score,
                                "confidence": pattern.confidence_score * match_score,
                                "timestamp": datetime.now().isoformat()
                            }
                            detections.append(detection)
        
        return detections
    
    # ==================== PATTERN LEARNING ====================
    
    def pattern_matching_analysis(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """Match detections against learned patterns"""
        matches = []
        
        for detection in detections:
            best_match = None
            best_score = 0
            
            for pattern_id, pattern in self.detection_patterns.items():
                match_score = self.calculate_detection_similarity(detection, pattern)
                
                if match_score > best_score and match_score > 0.7:
                    best_score = match_score
                    best_match = {
                        "pattern_id": pattern_id,
                        "match_score": match_score,
                        "pattern_data": asdict(pattern)
                    }
            
            if best_match:
                matches.append({
                    "detection_id": detection.get("method", "unknown") + "_" + str(len(matches)),
                    "matched_pattern": best_match,
                    "similarity_breakdown": self.get_similarity_breakdown(detection, pattern)
                })
        
        return matches
    
    def adaptive_learning(self, detections: List[Dict], image: np.ndarray):
        """Adaptive learning from successful detections"""
        if not self.config["feature_flags"]["enable_pattern_learning"]:
            return
        
        for detection in detections:
            if detection.get("confidence", 0) > self.confidence_threshold:
                # Extract features for learning
                features = self.extract_learning_features(detection, image)
                
                # Generate pattern ID
                pattern_id = self.generate_pattern_id(features)
                
                if pattern_id in self.detection_patterns:
                    # Update existing pattern
                    pattern = self.detection_patterns[pattern_id]
                    pattern.detection_count += 1
                    pattern.success_rate = min(0.99, pattern.success_rate + 0.01)
                    pattern.confidence_score = pattern.success_rate * 0.8 + detection["confidence"] * 0.2
                    pattern.last_updated = datetime.now().isoformat()
                    
                    # Adaptive size range adjustment
                    current_area = detection["area"]
                    pattern.size_range = (
                        min(pattern.size_range[0], current_area * 0.9),
                        max(pattern.size_range[1], current_area * 1.1)
                    )
                else:
                    # Create new pattern
                    new_pattern = DetectionPattern(
                        pattern_id=pattern_id,
                        color_range=detection.get("color_profile", [[0, 100, 100], [10, 255, 255]]),
                        shape_type=detection.get("shape_type", "unknown"),
                        size_range=(detection["area"] * 0.8, detection["area"] * 1.2),
                        success_rate=0.7,
                        detection_count=1,
                        first_detected=datetime.now().isoformat(),
                        last_updated=datetime.now().isoformat(),
                        features=features,
                        confidence_score=detection["confidence"]
                    )
                    self.detection_patterns[pattern_id] = new_pattern
                
                self.successful_detections += 1
        
        # Save patterns periodically
        if len(self.detection_patterns) % 10 == 0:
            self.save_ai_patterns()
    
    # ==================== BEHAVIORAL ANALYSIS ====================
    
    def behavioral_analysis(self, detections: List[Dict]) -> List[str]:
        """Analyze enemy behavior patterns"""
        insights = []
        
        if not detections:
            return ["No behavior to analyze"]
        
        # Group detections by type/profile
        detection_groups = defaultdict(list)
        for detection in detections:
            profile = detection.get("profile", detection.get("shape_type", "unknown"))
            detection_groups[profile].append(detection)
        
        # Generate insights for each group
        for profile, group_detections in detection_groups.items():
            if len(group_detections) > 1:
                # Calculate movement patterns
                positions = [d["position"] for d in group_detections]
                avg_x = sum(p["x"] for p in positions) / len(positions)
                avg_y = sum(p["y"] for p in positions) / len(positions)
                
                # Calculate spread
                spread_x = max(p["x"] for p in positions) - min(p["x"] for p in positions)
                spread_y = max(p["y"] for p in positions) - min(p["y"] for p in positions)
                
                insight = f"{profile}: Cluster at ({avg_x:.0f}, {avg_y:.0f}), Spread: {spread_x}x{spread_y}"
                insights.append(insight)
        
        # Predict next likely positions
        if len(detections) >= 3:
            movement_trend = self.analyze_movement_trend(detections)
            if movement_trend:
                insights.append(f"Movement trend: {movement_trend}")
        
        return insights
    
    def analyze_movement_trend(self, detections: List[Dict]) -> Optional[str]:
        """Analyze movement trends from detection history"""
        if len(detections) < 3:
            return None
        
        # Sort by timestamp if available
        positions = [d["position"] for d in detections[-3:]]
        
        # Calculate direction
        dx = positions[-1]["x"] - positions[0]["x"]
        dy = positions[-1]["y"] - positions[0]["y"]
        
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        
        speed = np.sqrt(dx**2 + dy**2) / 3  # Approximate speed
        
        return f"Moving {direction} at {speed:.1f} pixels/frame"
    
    # ==================== AI INSIGHTS GENERATION ====================
    
    def generate_ai_insights(self, analysis_results: Dict) -> List[str]:
        """Generate AI-powered insights from analysis"""
        insights = []
        
        detections = analysis_results.get("detections", [])
        
        if not detections:
            insights.append("No targets detected in current analysis")
            insights.append("Consider adjusting detection parameters")
            return insights
        
        # Detection quality insights
        confidences = [d.get("confidence", 0) for d in detections]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        insights.append(f"Detection confidence: {avg_confidence:.1%}")
        
        if avg_confidence > 0.8:
            insights.append("High confidence detections - system is performing well")
        elif avg_confidence < 0.5:
            insights.append("Low confidence - consider recalibration")
        
        # Spatial distribution insights
        positions = [d["position"] for d in detections]
        if len(positions) > 1:
            center_x = sum(p["x"] for p in positions) / len(positions)
            center_y = sum(p["y"] for p in positions) / len(positions)
            insights.append(f"Targets clustered around ({center_x:.0f}, {center_y:.0f})")
        
        # Pattern learning insights
        pattern_count = len(self.detection_patterns)
        if pattern_count > 0:
            top_pattern = max(self.detection_patterns.values(), key=lambda p: p.success_rate)
            insights.append(f"Best pattern: {top_pattern.pattern_id[:8]} ({top_pattern.success_rate:.1%} success)")
        
        # Performance insights
        if self.total_analyses > 10:
            success_rate = self.successful_detections / self.total_analyses
            insights.append(f"Overall success rate: {success_rate:.1%}")
            
            if success_rate < 0.3:
                insights.append("⚠️ Performance below expected threshold")
            elif success_rate > 0.7:
                insights.append("✅ System performing above expectations")
        
        # Learning progress insights
        if len(self.detection_patterns) > 20:
            insights.append(f"System has learned {len(self.detection_patterns)} patterns")
            insights.append("Consider exporting knowledge for future use")
        
        return insights
    
    # ==================== HELPER METHODS ====================
    
    def determine_shape_type(self, vertices: int, area: float, aspect_ratio: float) -> str:
        """Determine shape type based on geometric properties"""
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            if 0.9 < aspect_ratio < 1.1:
                return "square"
            else:
                return "rectangle"
        elif vertices > 8:
            return "circle"
        else:
            return "polygon"
    
    def calculate_pattern_match_score(self, contour, pattern: DetectionPattern) -> float:
        """Calculate match score between contour and pattern"""
        # This is a simplified version - can be enhanced with ML
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Compare with pattern features
        pattern_circularity = pattern.features.get("circularity", 1.0)
        pattern_area = pattern.features.get("area", area)
        
        # Calculate similarity scores
        area_similarity = 1 - min(1, abs(area - pattern_area) / max(area, pattern_area))
        shape_similarity = 1 - min(1, abs(circularity - pattern_circularity))
        
        return (area_similarity + shape_similarity) / 2
    
    def calculate_detection_similarity(self, detection: Dict, pattern: DetectionPattern) -> float:
        """Calculate similarity between detection and pattern"""
        similarity = 0.0
        
        # Area similarity
        detection_area = detection.get("area", 0)
        pattern_min, pattern_max = pattern.size_range
        if pattern_min <= detection_area <= pattern_max:
            area_ratio = detection_area / ((pattern_min + pattern_max) / 2)
            similarity += min(1.0, 1 - abs(1 - area_ratio)) * 0.3
        
        # Shape similarity
        if detection.get("shape_type") == pattern.shape_type:
            similarity += 0.3
        
        # Confidence boost
        similarity += pattern.success_rate * 0.4
        
        return min(similarity, 1.0)
    
    def filter_and_merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter and merge duplicate detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        filtered = []
        used_positions = []
        
        for detection in detections:
            pos = detection["position"]
            too_close = False
            
            for used_pos in used_positions:
                distance = np.sqrt((pos["x"] - used_pos["x"])**2 + (pos["y"] - used_pos["y"])**2)
                if distance < 20:  # Minimum distance threshold
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(detection)
                used_positions.append(pos)
        
        return filtered
    
    def extract_learning_features(self, detection: Dict, image: np.ndarray) -> Dict[str, Any]:
        """Extract features for machine learning"""
        x, y = detection["position"]["x"], detection["position"]["y"]
        w, h = detection["bounding_box"]["width"], detection["bounding_box"]["height"]
        
        features = {
            "position": (x, y),
            "size": (w, h),
            "area": detection["area"],
            "aspect_ratio": w / h if h > 0 else 0,
            "confidence": detection.get("confidence", 0.5)
        }
        
        # Extract color features from ROI
        roi = image[max(0, y-h//2):min(image.shape[0], y+h//2),
                   max(0, x-w//2):min(image.shape[1], x+w//2)]
        
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            features["color_mean"] = np.mean(hsv_roi, axis=(0, 1)).tolist()
            features["color_std"] = np.std(hsv_roi, axis=(0, 1)).tolist()
        
        return features
    
    def generate_pattern_id(self, features: Dict) -> str:
        """Generate unique pattern ID from features"""
        id_string = json.dumps(features, sort_keys=True)
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def calculate_performance_metrics(self, detections: List[Dict], processing_time: float) -> Dict[str, float]:
        """Calculate performance metrics for the analysis"""
        metrics = {
            "processing_time_ms": processing_time * 1000,
            "detection_count": len(detections),
            "avg_confidence": 0.0,
            "detection_density": 0.0
        }
        
        if detections:
            confidences = [d.get("confidence", 0) for d in detections]
            metrics["avg_confidence"] = sum(confidences) / len(confidences)
            metrics["max_confidence"] = max(confidences)
            metrics["min_confidence"] = min(confidences)
        
        # Update session metrics
        if self.current_session:
            self.current_session.performance_metrics["total_analyses"] += 1
            if metrics["avg_confidence"] > 0:
                self.current_session.performance_metrics["average_confidence"] = (
                    self.current_session.performance_metrics["average_confidence"] * 0.8 + 
                    metrics["avg_confidence"] * 0.2
                )
        
        return metrics
    
    def get_similarity_breakdown(self, detection: Dict, pattern: DetectionPattern) -> Dict[str, float]:
        """Get detailed similarity breakdown"""
        return {
            "area_similarity": 0.8,  # Simplified
            "shape_similarity": 0.7,
            "color_similarity": 0.9,
            "position_consistency": 0.6,
            "overall": self.calculate_detection_similarity(detection, pattern)
        }
    
    # ==================== ETHICAL COMPLIANCE ====================
    
    def display_ethical_warning(self):
        """Display ethical warning message"""
        print("\n" + "="*70)
        print("⚠️  ETHICAL USE WARNING - EDUCATIONAL SYSTEM ONLY ⚠️")
        print("="*70)
        
        guidelines = self.config["ethical_guidelines"]
        print(f"\nPrimary Rule: {guidelines['primary_rule']}")
        print("\nRules of Ethical Use:")
        for i, rule in enumerate(guidelines["rules"], 1):
            print(f"  {i}. {rule}")
        
        print("\nPotential Consequences of Misuse:")
        for i, consequence in enumerate(guidelines["consequences"], 1):
            print(f"  {i}. {consequence}")
        
        print("\n" + "="*70)
        print("By using this system, you agree to use it for educational purposes only")
        print("="*70 + "\n")
        
        if self.current_session:
            self.current_session.ethical_reminders_shown += 1
    
    def perform_ethical_check(self, analysis_results: Dict) -> Dict[str, Any]:
        """Perform ethical compliance check"""
        check = {
            "passed": True,
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Check 1: Too many high-confidence detections (potential cheating)
        detections = analysis_results.get("detections", [])
        high_conf_detections = [d for d in detections if d.get("confidence", 0) > 0.9]
        
        if len(high_conf_detections) > 10:
            check["warnings"].append("High number of confident detections detected")
            check["passed"] = False
        
        # Check 2: Unusual pattern learning rate
        if len(self.detection_patterns) > 100:
            check["warnings"].append("Extensive pattern learning detected")
        
        # Check 3: Rapid analysis (potential automation)
        if analysis_results["performance_metrics"]["processing_time_ms"] < 10:
            check["warnings"].append("Extremely fast analysis detected")
        
        # Update session ethical checks
        if self.current_session:
            if check["passed"]:
                self.current_session.performance_metrics["ethical_checks_passed"] += 1
        
        return check
    
    # ==================== VISUALIZATION ====================
    
    def display_advanced_visualization(self, image: np.ndarray, analysis_results: Dict):
        """Display advanced visualization of analysis results"""
        plt.figure(figsize=(20, 12))
        
        # 1. Original image with detections
        plt.subplot(2, 3, 1)
        display_img = image.copy()
        
        # Draw detections
        for detection in analysis_results["detections"]:
            bbox = detection["bounding_box"]
            pos = detection["position"]
            conf = detection.get("confidence", 0)
            
            # Color based on confidence
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(display_img, 
                         (bbox["x"], bbox["y"]), 
                         (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]), 
                         color, 2)
            
            # Draw center point
            cv2.circle(display_img, (pos["x"], pos["y"]), 5, (255, 0, 0), -1)
            
            # Add confidence label
            label = f"{conf:.1%}"
            cv2.putText(display_img, label, 
                       (bbox["x"], bbox["y"] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detections ({len(analysis_results["detections"])} objects)')
        plt.axis('off')
        
        # 2. Confidence distribution
        plt.subplot(2, 3, 2)
        if analysis_results["detections"]:
            confidences = [d.get("confidence", 0) for d in analysis_results["detections"]]
            plt.hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=self.confidence_threshold, color='red', linestyle='--', label=f'Threshold ({self.confidence_threshold})')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Confidence Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No detections', ha='center', va='center')
            plt.axis('off')
        
        # 3. System performance
        plt.subplot(2, 3, 3)
        perf_metrics = analysis_results["performance_metrics"]
        metrics_to_show = {
            'Processing Time (ms)': perf_metrics.get('processing_time_ms', 0),
            'Detection Count': perf_metrics.get('detection_count', 0),
            'Avg Confidence': perf_metrics.get('avg_confidence', 0) * 100,
            'Max Confidence': perf_metrics.get('max_confidence', 0) * 100
        }
        
        bars = plt.bar(range(len(metrics_to_show)), list(metrics_to_show.values()), 
                      color=['blue', 'green', 'orange', 'red'])
        plt.xticks(range(len(metrics_to_show)), list(metrics_to_show.keys()), rotation=45)
        plt.ylabel('Value')
        plt.title('Performance Metrics')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 4. AI Insights
        plt.subplot(2, 3, 4)
        insights = analysis_results.get("ai_insights", [])
        if insights:
            for i, insight in enumerate(insights[:6]):  # Show first 6 insights
                plt.text(0.1, 0.9 - i*0.15, f"• {insight}", fontsize=9, 
                        verticalalignment='top', wrap=True)
        else:
            plt.text(0.5, 0.5, 'No AI insights generated', ha='center', va='center')
        plt.axis('off')
        plt.title('AI Insights & Recommendations')
        
        # 5. Pattern Learning Status
        plt.subplot(2, 3, 5)
        if self.detection_patterns:
            # Get top patterns by success rate
            top_patterns = sorted(self.detection_patterns.values(), 
                                 key=lambda p: p.success_rate, 
                                 reverse=True)[:5]
            
            pattern_ids = [p.pattern_id[:8] for p in top_patterns]
            success_rates = [p.success_rate * 100 for p in top_patterns]
            
            bars = plt.barh(pattern_ids, success_rates, color='green', alpha=0.7)
            plt.xlabel('Success Rate (%)')
            plt.title('Top Learned Patterns')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2., 
                        f'{width:.1f}%', va='center')
        else:
            plt.text(0.5, 0.5, 'No patterns learned yet', ha='center', va='center')
            plt.axis('off')
        
        # 6. Ethical Compliance Status
        plt.subplot(2, 3, 6)
        ethical_check = analysis_results.get("ethical_check", {})
        
        if ethical_check.get("passed", True):
            status_text = "✅ ETHICAL COMPLIANCE PASSED"
            color = 'green'
        else:
            status_text = "⚠️ ETHICAL WARNINGS DETECTED"
            color = 'red'
        
        plt.text(0.5, 0.7, status_text, ha='center', va='center', 
                fontsize=14, fontweight='bold', color=color)
        
        warnings = ethical_check.get("warnings", [])
        if warnings:
            plt.text(0.5, 0.5, "Warnings:", ha='center', va='center', fontweight='bold')
            for i, warning in enumerate(warnings[:3]):  # Show first 3 warnings
                plt.text(0.5, 0.4 - i*0.1, f"• {warning}", ha='center', va='center', fontsize=9)
        else:
            plt.text(0.5, 0.5, "No warnings detected", ha='center', va='center')
        
        plt.axis('off')
        plt.title('Ethical Compliance Check')
        
        plt.tight_layout()
        plt.show()
    
    # ==================== DATA SAVING ====================
    
    def save_analysis_results(self, analysis_results: Dict):
        """Save analysis results to history"""
        self.analysis_history.append(analysis_results)
        
        # Save to file periodically
        if len(self.analysis_history) % 50 == 0:
            try:
                history_list = list(self.analysis_history)
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(history_list, f, indent=2, ensure_ascii=False)
                print(f"💾 Analysis history saved ({len(history_list)} records)")
            except Exception as e:
                print(f"❌ History save error: {e}")
    
    def save_all_data(self):
        """Save all system data"""
        print("\n💾 Saving all system data...")
        self.save_configuration()
        self.save_ai_patterns()
        self.save_behavior_profiles()
        self.save_game_sessions()
        self.save_ai_models()
        
        # Save current analysis history
        if self.analysis_history:
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(list(self.analysis_history), f, indent=2, ensure_ascii=False)
                print(f"💾 Analysis history saved ({len(self.analysis_history)} records)")
            except Exception as e:
                print(f"❌ History save error: {e}")
        
        print("✅ All data saved successfully")
    
    # ==================== SYSTEM UTILITIES ====================
    
    def display_system_banner(self):
        """Display system information banner"""
        print("\n" + "="*70)
        print("🤖 ADVANCED AI AIM ANALYZER - EDUCATIONAL VERSION 3.0")
        print("="*70)
        
        info = self.config["system_info"]
        print(f"System: {info['name']}")
        print(f"Version: {info['version']}")
        print(f"Purpose: {info['purpose']}")
        print(f"License: {info['license']}")
        
        print(f"\n⚙️  System Status:")
        print(f"  • Patterns Learned: {len(self.detection_patterns)}")
        print(f"  • Profiles Tracked: {len(self.enemy_profiles)}")
        print(f"  • Sessions Recorded: {len(self.game_sessions)}")
        print(f"  • Total Analyses: {self.total_analyses}")
        
        print("\n" + "="*70)
        print("⚠️  IMPORTANT: This system is for EDUCATIONAL PURPOSES ONLY")
        print("="*70 + "\n")
    
    def create_sample_training_image(self) -> str:
        """Create a sample image for training and demonstration"""
        print("\n🎨 Creating sample training image...")
        
        # Create a complex image with various shapes and colors
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        image[:] = (50, 50, 50)  # Gray background
        
        # Add various shapes (simulating game elements)
        shapes = [
            # (shape, center, size, color, label)
            ("circle", (150, 150), 40, (0, 0, 255), "Enemy A"),  # Red
            ("rectangle", (300, 200), (60, 40), (255, 0, 0), "Enemy B"),  # Blue
            ("circle", (450, 120), 30, (0, 255, 0), "Critical"),  # Green
            ("triangle", (200, 350), 50, (0, 255, 255), "Target"),  # Yellow
            ("circle", (500, 300), 35, (255, 0, 255), "Enemy C"),  # Magenta
            ("rectangle", (350, 450), (70, 50), (255, 255, 0), "Objective"),  # Cyan
            ("circle", (600, 200), 45, (0, 165, 255), "Boss"),  # Orange
            ("triangle", (700, 400), 60, (255, 255, 255), "Special")  # White
        ]
        
        for shape_type, center, size, color, label in shapes:
            if shape_type == "circle":
                cv2.circle(image, center, size, color, -1)
            elif shape_type == "rectangle":
                x, y = center
                w, h = size
                cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            elif shape_type == "triangle":
                pts = np.array([
                    [center[0], center[1]-size],
                    [center[0]-size, center[1]+size],
                    [center[0]+size, center[1]+size]
                ], np.int32)
                cv2.fillPoly(image, [pts], color)
            
            # Add label
            cv2.putText(image, label, (center[0]-40, center[1]+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add some noise/distractions
        for _ in range(50):
            x, y = np.random.randint(0, 800), np.random.randint(0, 600)
            cv2.circle(image, (x, y), 3, (100, 100, 100), -1)
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_training_image_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        
        print(f"✅ Sample image created: {filename}")
        return filename
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system report"""
        print("\n📊 Generating comprehensive system report...")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_info": self.config["system_info"],
            "performance_summary": {
                "total_analyses": self.total_analyses,
                "successful_detections": self.successful_detections,
                "success_rate": self.successful_detections / self.total_analyses if self.total_analyses > 0 else 0,
                "patterns_learned": len(self.detection_patterns),
                "profiles_tracked": len(self.enemy_profiles),
                "sessions_recorded": len(self.game_sessions)
            },
            "ai_capabilities": {
                "learning_enabled": self.config["feature_flags"]["enable_pattern_learning"],
                "adaptive_learning": self.config["ai_settings"]["adaptive_learning_rate"],
                "current_confidence_threshold": self.confidence_threshold,
                "model_status": "Loaded" if self.kmeans_model else "Not loaded"
            },
            "top_performing_patterns": [],
            "ethical_compliance": {
                "warnings_issued": sum(1 for h in self.analysis_history 
                                     if not h.get("ethical_check", {}).get("passed", True)),
                "reminders_shown": sum(s.get("ethical_reminders_shown", 0) 
                                     for s in self.game_sessions.values())
            },
            "recommendations": []
        }
        
        # Add top patterns
        if self.detection_patterns:
            top_patterns = sorted(self.detection_patterns.values(), 
                                 key=lambda p: p.success_rate, 
                                 reverse=True)[:5]
            report["top_performing_patterns"] = [
                {
                    "pattern_id": p.pattern_id[:8],
                    "success_rate": p.success_rate,
                    "detection_count": p.detection_count,
                    "shape_type": p.shape_type
                } for p in top_patterns
            ]
        
        # Add recommendations
        if report["performance_summary"]["success_rate"] < 0.5:
            report["recommendations"].append("Consider recalibrating detection parameters")
        if len(self.detection_patterns) < 10:
            report["recommendations"].append("Train system with more diverse images")
        if report["ethical_compliance"]["warnings_issued"] > 5:
            report["recommendations"].append("Review ethical compliance guidelines")
        
        # Save report to file
        report_file = f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"✅ System report saved to: {report_file}")
        return report

# ==================== MAIN INTERFACE ====================

def main():
    """Main interface for the Advanced AI Aim Analyzer"""
    print("🚀 Starting Advanced AI Aim Analyzer...")
    
    # Initialize the AI system
    ai_system = AdvancedAIAimAnalyzer()
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("🤖 ADVANCED AI AIM ANALYZER - MAIN MENU")
        print("="*60)
        print("1. 🔍 Analyze an image")
        print("2. 🎨 Create and analyze sample image")
        print("3. 📊 View system statistics")
        print("4. 🧠 Train AI models")
        print("5. 📋 Generate system report")
        print("6. ⚙️  System configuration")
        print("7. 💾 Save all data")
        print("8. ⚠️  Show ethical guidelines")
        print("9. 🚪 Exit system")
        print("="*60)
        
        try:
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                # Analyze user-provided image
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    real_time = input("Real-time mode? (y/n): ").strip().lower() == 'y'
                    results = ai_system.analyze_image(image_path, real_time)
                    print(f"\nAnalysis completed. Found {len(results.get('detections', []))} objects.")
                else:
                    print(f"❌ Image not found: {image_path}")
            
            elif choice == '2':
                # Create and analyze sample image
                sample_path = ai_system.create_sample_training_image()
                results = ai_system.analyze_image(sample_path)
                print(f"\nSample analysis completed. Found {len(results.get('detections', []))} objects.")
            
            elif choice == '3':
                # Display system statistics
                print(f"\n📈 SYSTEM STATISTICS:")
                print(f"   • Total analyses performed: {ai_system.total_analyses}")
                print(f"   • Successful detections: {ai_system.successful_detections}")
                print(f"   • Patterns learned: {len(ai_system.detection_patterns)}")
                print(f"   • Enemy profiles: {len(ai_system.enemy_profiles)}")
                print(f"   • Game sessions: {len(ai_system.game_sessions)}")
                
                if ai_system.detection_patterns:
                    success_rates = [p.success_rate for p in ai_system.detection_patterns.values()]
                    print(f"   • Average pattern success rate: {sum(success_rates)/len(success_rates):.1%}")
            
            elif choice == '4':
                # Train AI models
                print("\n🧠 AI Model Training")
                print("This feature is under development.")
                print("Currently, models are trained automatically during analysis.")
            
            elif choice == '5':
                # Generate system report
                report = ai_system.generate_system_report()
                print("\n✅ System report generated successfully")
                print(f"   • Success rate: {report['performance_summary']['success_rate']:.1%}")
                print(f"   • Patterns learned: {report['performance_summary']['patterns_learned']}")
                print(f"   • Ethical warnings: {report['ethical_compliance']['warnings_issued']}")
            
            elif choice == '6':
                # System configuration
                print("\n⚙️  SYSTEM CONFIGURATION")
                print("Current settings:")
                print(f"   • Confidence threshold: {ai_system.confidence_threshold}")
                print(f"   • Learning rate: {ai_system.learning_rate}")
                print(f"   • Max patterns: {ai_system.config['ai_settings']['max_patterns']}")
                
                change = input("\nChange confidence threshold? (y/n): ").strip().lower()
                if change == 'y':
                    try:
                        new_threshold = float(input("New threshold (0.1-0.9): "))
                        if 0.1 <= new_threshold <= 0.9:
                            ai_system.confidence_threshold = new_threshold
                            print(f"✅ Threshold updated to {new_threshold}")
                        else:
                            print("❌ Threshold must be between 0.1 and 0.9")
                    except ValueError:
                        print("❌ Invalid input")
            
            elif choice == '7':
                # Save all data
                ai_system.save_all_data()
            
            elif choice == '8':
                # Show ethical guidelines
                ai_system.display_ethical_warning()
            
            elif choice == '9':
                # Exit system
                print("\n" + "="*60)
                print("👋 Thank you for using the Advanced AI Aim Analyzer!")
                print("⚠️  Remember: This system is for EDUCATIONAL PURPOSES ONLY")
                print("="*60)
                
                # End current session
                if ai_system.current_session:
                    ai_system.end_current_session()
                
                # Save all data before exiting
                save_data = input("\nSave all data before exiting? (y/n): ").strip().lower()
                if save_data == 'y':
                    ai_system.save_all_data()
                
                break
            
            else:
                print("❌ Invalid option. Please choose 1-9.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  System interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

# ==================== QUICK START FUNCTION ====================

def quick_start_demo():
    """Quick start demonstration of the AI system"""
    print("🚀 QUICK START DEMO - AI AIM ANALYZER")
    print("="*60)
    
    # Initialize system
    ai_system = AdvancedAIAimAnalyzer()
    
    # Create and analyze sample image
    print("\n1. Creating sample training image...")
    sample_path = ai_system.create_sample_training_image()
    
    print("\n2. Analyzing sample image with AI...")
    results = ai_system.analyze_image(sample_path)
    
    print("\n3. Generating system report...")
    report = ai_system.generate_system_report()
    
    print("\n4. Displaying key findings...")
    print(f"   • Objects detected: {len(results.get('detections', []))}")
    print(f"   • Analysis time: {results['performance_metrics']['processing_time_ms']:.1f}ms")
    print(f"   • Average confidence: {results['performance_metrics'].get('avg_confidence', 0):.1%}")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("="*60)
    
    # Save data
    ai_system.save_all_data()
    
    return ai_system, results

if __name__ == "__main__":
    # Check if user wants quick demo
    print("Welcome to Advanced AI Aim Analyzer!")
    print("Choose an option:")
    print("1. Quick Start Demo")
    print("2. Full Interactive System")
    
    choice = input("Select (1-2): ").strip()
    
    if choice == '1':
        quick_start_demo()
    else:
        main()