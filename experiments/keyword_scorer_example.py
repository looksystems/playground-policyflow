"""
Example usage of KeywordScorerNode.

This demonstrates how to use the KeywordScorerNode to score input text
based on weighted keyword presence. It's a deterministic node (no LLM)
that can be used for fast, cheap routing and filtering in workflows.
"""

# Example 1: Email Priority Scoring
# This node can determine if an email needs urgent attention based on keywords

example_keywords = {
    # Positive indicators (urgent/important)
    "urgent": 0.5,
    "critical": 0.8,
    "asap": 0.6,
    "emergency": 1.0,
    "deadline": 0.4,

    # Negative indicators (spam/low priority)
    "spam": -1.0,
    "unsubscribe": -0.8,
    "marketing": -0.5,
}

example_thresholds = {
    "high": 0.7,      # Score >= 0.7 → high priority
    "medium": 0.3,    # Score >= 0.3 → medium priority
    # Score < 0.3 → low priority
}

# Usage in a workflow:
# from policyflow.nodes import KeywordScorerNode
#
# scorer = KeywordScorerNode(
#     keywords=example_keywords,
#     thresholds=example_thresholds
# )
#
# # In your flow:
# shared = {"input_text": "URGENT: Critical deadline approaching ASAP!"}
# prep_res = scorer.prep(shared)
# exec_res = scorer.exec(prep_res)
# action = scorer.post(shared, prep_res, exec_res)
#
# print(f"Action: {action}")  # Returns: "high"
# print(f"Score: {shared['keyword_score']['score']}")  # 2.4
# print(f"Matched: {shared['keyword_score']['matched_keywords']}")
# # {
# #   'urgent': {'count': 1, 'weight': 0.5, 'contribution': 0.5},
# #   'critical': {'count': 1, 'weight': 0.8, 'contribution': 0.8},
# #   'asap': {'count': 1, 'weight': 0.6, 'contribution': 0.6},
# #   'deadline': {'count': 1, 'weight': 0.4, 'contribution': 0.4}
# # }


# Example 2: Content Moderation
# Filter harmful content before sending to expensive LLM

moderation_keywords = {
    # Prohibited content
    "hate": -2.0,
    "violence": -2.0,
    "abuse": -1.5,

    # Borderline content (needs review)
    "controversial": -0.5,
    "sensitive": -0.4,
}

moderation_thresholds = {
    "high": 0.0,      # No negative keywords → proceed
    "medium": -1.0,   # Minor concerns → review
    # Score < -1.0 → reject
}

# This allows you to:
# - Route "low" score messages directly to rejection (no LLM needed)
# - Route "medium" to human review
# - Route "high" to normal LLM processing


# Example 3: Multi-language Support Detection
# Detect language based on common words (simple heuristic)

language_keywords = {
    # Spanish indicators
    "el": 0.2,
    "la": 0.2,
    "de": 0.2,
    "que": 0.2,

    # French indicators
    "le": 0.2,
    "de": 0.2,  # Note: overlaps with Spanish
    "et": 0.2,

    # English is default, no keywords needed
}

language_thresholds = {
    "high": 0.8,   # Strong non-English signal
    "medium": 0.3, # Possible non-English
}

# Use this to route non-English text to translation services before evaluation


# Example 4: Financial Advisory Compliance
# Score messages for compliance risk based on prohibited terms

compliance_keywords = {
    # High risk terms
    "guaranteed": -2.0,
    "risk-free": -3.0,
    "cannot lose": -3.0,

    # Medium risk terms
    "definitely": -1.0,
    "always": -0.8,
    "never": -0.8,

    # Safe terms (positive indicators)
    "may": 0.5,
    "potential": 0.5,
    "risk": 0.8,
    "disclosure": 1.0,
}

compliance_thresholds = {
    "high": 0.5,    # Well-disclaimed, compliant
    "medium": -0.5, # Needs review
    # Score < -0.5 → likely non-compliant
}


print(__doc__)
print("\nKeywordScorerNode can be used for:")
print("- Email priority routing")
print("- Content moderation/filtering")
print("- Language detection")
print("- Compliance screening")
print("- Spam detection")
print("- Topic classification (simple cases)")
print("\nBenefits:")
print("- Zero cost (no LLM calls)")
print("- Instant response time")
print("- Fully deterministic")
print("- Easy to audit and explain")
print("- Can reduce LLM calls by 50-80% in many workflows")
