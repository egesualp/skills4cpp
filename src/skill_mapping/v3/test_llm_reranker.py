"""
test_llm_reranker.py - Quick test script for LLM re-ranking

Tests the re-ranking logic on a small sample without making API calls.
"""

from dataclasses import dataclass
from typing import Dict, List

from skill_mapping.v3.llm_reranker import LLMReranker, SkillCandidate


def test_rescore_skills():
    """Test the re-scoring logic without API calls."""
    
    # Create mock candidates
    candidates = [
        SkillCandidate(
            skill_uri="http://example.com/skill1",
            skill_name="Python Programming",
            skill_description="Write code in Python",
            original_score=0.95,
            original_rank=1,
        ),
        SkillCandidate(
            skill_uri="http://example.com/skill2",
            skill_name="Data Analysis",
            skill_description="Analyze data",
            original_score=0.90,
            original_rank=2,
        ),
        SkillCandidate(
            skill_uri="http://example.com/skill3",
            skill_name="Project Management",
            skill_description="Manage projects",
            original_score=0.85,
            original_rank=3,
        ),
        SkillCandidate(
            skill_uri="http://example.com/skill4",
            skill_name="Cooking",
            skill_description="Prepare food",
            original_score=0.80,
            original_rank=4,
        ),
    ]
    
    # Mock classifications: skill 4 is irrelevant, skill 3 is optional, 1-2 are essential
    classifications = {
        1: "Essential",
        2: "Essential",
        3: "Optional",
        4: "Irrelevant",
    }
    
    # Create reranker (no API key needed for this test)
    reranker = LLMReranker(api_key="dummy")
    
    # Re-score
    ranked_skills = reranker._rescore_skills(candidates, classifications)
    
    # Print results
    print("Re-scoring Results:")
    print("=" * 80)
    for skill in ranked_skills:
        print(f"Rank {skill.final_rank}: {skill.skill_name}")
        print(f"  Tier: {skill.tier}")
        print(f"  Original Score: {skill.original_score:.4f} (rank {skill.original_rank})")
        print(f"  Final Score: {skill.final_score:.4f}")
        print()
    
    # Verify tier separation
    essential_skills = [s for s in ranked_skills if s.tier == "Essential"]
    optional_skills = [s for s in ranked_skills if s.tier == "Optional"]
    irrelevant_skills = [s for s in ranked_skills if s.tier == "Irrelevant"]
    
    print("Verification:")
    print("=" * 80)
    print(f"Essential skills: {len(essential_skills)} (ranks 1-{len(essential_skills)})")
    print(f"Optional skills: {len(optional_skills)} (ranks {len(essential_skills)+1}-{len(essential_skills)+len(optional_skills)})")
    print(f"Irrelevant skills: {len(irrelevant_skills)} (ranks {len(essential_skills)+len(optional_skills)+1}+)")
    
    # Check that tier separation works
    if essential_skills:
        max_essential_score = max(s.final_score for s in essential_skills)
    else:
        max_essential_score = 0
    
    if optional_skills:
        min_optional_score = min(s.final_score for s in optional_skills)
        max_optional_score = max(s.final_score for s in optional_skills)
    else:
        min_optional_score = max_optional_score = 0
    
    if irrelevant_skills:
        min_irrelevant_score = min(s.final_score for s in irrelevant_skills)
    else:
        min_irrelevant_score = 0
    
    if optional_skills and essential_skills:
        assert max_essential_score > max_optional_score, "Essential should score higher than Optional"
        print("✓ Essential skills rank above Optional skills")
    
    if irrelevant_skills and optional_skills:
        assert max_optional_score > min_irrelevant_score, "Optional should score higher than Irrelevant"
        print("✓ Optional skills rank above Irrelevant skills")
    
    # Check that within-tier ordering is preserved
    for i in range(len(essential_skills) - 1):
        assert essential_skills[i].original_score >= essential_skills[i+1].original_score, \
            "Within Essential tier, original order should be preserved"
    print("✓ Within-tier ordering preserved")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_rescore_skills()







