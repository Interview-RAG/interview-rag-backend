import re

def calculate_general_score(parsed_data: dict) -> dict:
    """
    Calculates a deterministic ATS General Health Score out of 100.
    Returns the score and a list of feedback items (deductions and praises).
    """
    score = 100
    feedback = []
    
    # 1. Check Completeness
    contact = parsed_data.get("contact_info") or {}
    if not contact.get("email"):
        score -= 5
        feedback.append({"type": "negative", "message": "Missing email address in contact info."})
    if not contact.get("phone"):
        score -= 5
        feedback.append({"type": "negative", "message": "Missing phone number in contact info."})
        
    if not parsed_data.get("summary"):
        score -= 10
        feedback.append({"type": "negative", "message": "Missing professional summary."})
    else:
        feedback.append({"type": "positive", "message": "Professional summary included."})
        
    if not parsed_data.get("skills") or len(parsed_data.get("skills")) == 0:
        score -= 10
        feedback.append({"type": "negative", "message": "No skills section found."})
        
    if not parsed_data.get("experience") or len(parsed_data.get("experience")) == 0:
        score -= 15
        feedback.append({"type": "negative", "message": "No work experience found."})
        
    if not parsed_data.get("education") or len(parsed_data.get("education")) == 0:
        score -= 10
        feedback.append({"type": "negative", "message": "No education history found."})
        
    # 2. Check Quantifiable Metrics & Action Verbs in Experience
    experience = parsed_data.get("experience") or []
    metrics_deduction = 0
    bullets_checked = 0
    
    # Simple regex to find numbers (digits) or percentages
    metric_pattern = re.compile(r'\d+|%|percent|million|billion|thousand', re.IGNORECASE)
    
    for exp in experience:
        desc = exp.get("description", "")
        
        # If it's a string, split by newlines. If it's an array, use it directly.
        bullets = desc if isinstance(desc, list) else desc.split('\n')
        
        for bullet in bullets:
            if not bullet.strip():
                continue
            bullets_checked += 1
            if not metric_pattern.search(bullet):
                metrics_deduction += 2
                
    if bullets_checked > 0:
        # Cap deduction at 20 points
        if metrics_deduction > 20:
            metrics_deduction = 20
        
        if metrics_deduction > 0:
            score -= metrics_deduction
            feedback.append({"type": "negative", "message": f"Missing quantifiable metrics (numbers/percentages) in {metrics_deduction // 2} experience bullet points."})
        else:
            feedback.append({"type": "positive", "message": "Excellent use of quantifiable metrics in experience."})

    # Floor the score at 0
    score = max(0, score)
    
    return {
        "score": score,
        "feedback": feedback
    }

def calculate_targeted_score(parsed_data: dict, jd_keywords: list) -> dict:
    """
    Calculates a targeted ATS match score based on extracted job description keywords.
    """
    score = 100
    feedback = []
    
    jd_words = set(kw.lower() for kw in jd_keywords)
    
    # Extract words from resume
    resume_text = ""
    skills = parsed_data.get("skills", [])
    resume_text += " ".join(skills).lower() + " "
    
    for exp in parsed_data.get("experience", []):
        desc = exp.get("description", "")
        if isinstance(desc, list):
            resume_text += " ".join(desc).lower() + " "
        else:
            resume_text += desc.lower() + " "
            
    for proj in parsed_data.get("projects", []):
        techs = proj.get("technologies", [])
        resume_text += " ".join(techs).lower() + " "
        
    # Calculate overlap
    if not jd_words:
        return {"score": 0, "feedback": [{"type": "negative", "message": "Could not extract valid skills from Job Description."}]}
        
    overlap = set()
    missing = set()
    
    for kw in jd_words:
        # Check if the keyword exists as a substring in the resume text
        if kw in resume_text:
            overlap.add(kw)
        else:
            missing.add(kw)
            
    match_percentage = len(overlap) / len(jd_words)
    
    # Scoring math: 
    raw_score = int(match_percentage * 100)
    score = min(100, max(10, raw_score))
    
    if score >= 80:
        feedback.append({"type": "positive", "message": f"Strong keyword alignment ({len(overlap)}/{len(jd_words)} skills matched)."})
    elif score >= 50:
        feedback.append({"type": "neutral", "message": f"Moderate keyword alignment ({len(overlap)}/{len(jd_words)} skills matched). Consider adding more JD skills."})
    else:
        feedback.append({"type": "negative", "message": f"Low keyword alignment ({len(overlap)}/{len(jd_words)} skills matched). You may be missing core requirements."})
        
    if missing:
        sample_missing = list(missing)[:7]
        feedback.append({"type": "negative", "message": f"Missing core skills: {', '.join(sample_missing)}"})

    return {
        "score": score,
        "feedback": feedback
    }
