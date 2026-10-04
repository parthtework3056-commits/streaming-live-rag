import uuid

# Define a robust set of 25 scenarios for G1-G6 testing
# Mix of eligible, ineligible, compound, delta, and formatting queries

SCENARIOS = [
    # 1. Workshop (Compound, Eligible)
    {
        "id": "q_01",
        "type": "compound",
        "chunks": [
            {"text": "I need to plan a customer workshop in...", "offset": 0.0, "is_last": False},
            {"text": "...Pune for 30 people, and I need...", "offset": 0.8, "is_last": False},
            {"text": "...cancellation policy and catering options.", "offset": 1.6, "is_last": False},
            {"text": "[Utterance End]", "offset": 2.1, "is_last": True}
        ],
        "delta": [
            {"text": "Actually, the workshop is for an international client.", "offset": 3.0, "is_last": True}
        ]
    },
    # 2. Short general query (Ineligible / Wait)
    {
        "id": "q_02",
        "type": "incomplete",
        "chunks": [
            {"text": "I was wondering...", "offset": 0.0, "is_last": False},
            {"text": "[Utterance End]", "offset": 0.5, "is_last": True}
        ],
        "delta": []
    },
    # 3. Restructure (Ineligible / Suppress)
    {
        "id": "q_03",
        "type": "formatting",
        "chunks": [
            {"text": "Can you turn the last answer into 3 bullets?", "offset": 0.0, "is_last": True}
        ],
        "delta": []
    },
    # 4. Travel reimbursement (Eligible)
    {
        "id": "q_04",
        "type": "simple",
        "chunks": [
            {"text": "I need to travel to Mumbai next week...", "offset": 0.0, "is_last": False},
            {"text": "...what are the meal limits?", "offset": 1.0, "is_last": False},
            {"text": "[Utterance End]", "offset": 1.5, "is_last": True}
        ],
        "delta": [
            {"text": "What about international travel?", "offset": 3.0, "is_last": True}
        ]
    },
    # 5. Flight policy (Eligible)
    {
        "id": "q_05",
        "type": "simple",
        "chunks": [
            {"text": "What is the policy for...", "offset": 0.0, "is_last": False},
            {"text": "...booking a flight to London?", "offset": 1.2, "is_last": False},
            {"text": "[Utterance End]", "offset": 1.8, "is_last": True}
        ],
        "delta": [
            {"text": "I already booked it post-travel.", "offset": 3.0, "is_last": True}
        ]
    },
    # 6. Compound Event Planning
    {
        "id": "q_06",
        "type": "compound",
        "chunks": [
            {"text": "We are hosting a seminar in Delhi...", "offset": 0.0, "is_last": False},
            {"text": "...for 50 attendees. Can you check...", "offset": 1.0, "is_last": False},
            {"text": "...the catering budget and the venue capacity?", "offset": 2.0, "is_last": False},
            {"text": "[Utterance End]", "offset": 2.5, "is_last": True}
        ],
        "delta": []
    },
    # 7. Single fast query
    {
        "id": "q_07",
        "type": "simple",
        "chunks": [
            {"text": "What is the corporate travel allowance?", "offset": 0.0, "is_last": True}
        ],
        "delta": []
    },
    # 8. Another incomplete
    {
        "id": "q_08",
        "type": "incomplete",
        "chunks": [
            {"text": "Can you tell me...", "offset": 0.0, "is_last": False},
            {"text": "[Utterance End]", "offset": 0.8, "is_last": True}
        ],
        "delta": []
    },
    # 9. Complex Compound
    {
        "id": "q_09",
        "type": "compound",
        "chunks": [
            {"text": "I need to know the cancellation penalty...", "offset": 0.0, "is_last": False},
            {"text": "...and the maximum room capacity...", "offset": 1.0, "is_last": False},
            {"text": "...for a meeting in Bangalore.", "offset": 2.0, "is_last": True}
        ],
        "delta": []
    },
    # 10. Executive Override Delta
    {
        "id": "q_10",
        "type": "simple",
        "chunks": [
            {"text": "What is the daily meal limit?", "offset": 0.0, "is_last": True}
        ],
        "delta": [
            {"text": "Wait, I am an executive director.", "offset": 2.0, "is_last": True}
        ]
    },
]

# Generate remaining 15 programmatic variants to hit 25 cases
for i in range(11, 26):
    SCENARIOS.append({
        "id": f"q_{i:02d}",
        "type": "simple" if i % 2 == 0 else "compound",
        "chunks": [
            {"text": f"Tell me about corporate policies part {i}...", "offset": 0.0, "is_last": False},
            {"text": "...specifically regarding travel and expenses.", "offset": 1.0, "is_last": False} if i % 2 != 0 else {"text": "...and nothing else.", "offset": 1.0, "is_last": False},
            {"text": "[Utterance End]", "offset": 1.5, "is_last": True}
        ],
        "delta": []
    })
