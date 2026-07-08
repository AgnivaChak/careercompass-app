import json
import re


def load_project_ideas(json_path="assets/project_ideas.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _keyword_matches(keywords, tokens):
    """Return True if any project keyword appears in the token list."""
    token_set = set(tokens)
    token_text = " ".join(tokens)
    for kw in keywords:
        if kw in token_set:
            return True
        if re.search(r"\b" + re.escape(kw) + r"\b", token_text):
            return True
    return False


def recommend_projects(jd_keywords, resume_keywords, ideas):
    aligned_projects = []
    suggested_projects = []

    for idea in ideas:
        project_title = idea.get("project_title", idea.get("title", "Untitled Project")).strip()
        description = idea.get("description", "").strip()
        keywords = [kw.lower() for kw in idea.get("keywords", [])]

        # Flags
        matches_resume = _keyword_matches(keywords, resume_keywords)
        matches_jd = _keyword_matches(keywords, jd_keywords)

        project_summary = {
            "project_title": project_title,  # Ensure key is consistent
            "description": description,
            "domain": idea.get("domain", ""),
            "tools": idea.get("tools", []),
            "keywords": keywords,
            "level": idea.get("level", "")
        }

        if matches_resume:
            aligned_projects.append(project_summary)
        elif matches_jd:
            suggested_projects.append(project_summary)

    return {
        "aligned": aligned_projects,
        "suggested": suggested_projects
    }
