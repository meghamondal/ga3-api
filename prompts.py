"""
All prompts used by the GA3 endpoints.

Keeping them here makes main.py much cleaner.
"""


# =====================================================
# Q2
# =====================================================

IMAGE_QA_PROMPT = """
Answer the question about this image.

Rules:

- Return ONLY the requested answer.
- Do NOT explain.
- Do NOT include units.
- Do NOT include currency symbols.
- Numbers must be plain numbers.
- Return valid JSON only.

Example:

{
    "answer":"4089.35"
}
"""


# =====================================================
# Q3
# =====================================================

INVOICE_EXTRACTION_PROMPT = """
Extract these fields from the invoice.

Return EXACTLY these keys:

invoice_no
date
vendor
amount
tax
currency

Rules:

- date → YYYY-MM-DD
- amount = subtotal BEFORE tax
- tax = tax amount only
- currency = ISO code (INR, USD, EUR...)
- Use null if missing.

Return JSON only.
"""


# =====================================================
# Q4
# =====================================================

DYNAMIC_EXTRACTION_PROMPT = """
Extract information from the text.

Return EXACTLY the keys provided in the schema.

Rules:

- No extra keys.
- Missing values -> null.
- Dates -> YYYY-MM-DD.
- Integers -> JSON integers.
- Floats -> JSON numbers.
- Boolean -> true/false.
- Arrays -> JSON arrays.

Return JSON only.
"""


# =====================================================
# Q7
# =====================================================

STRUCTURED_INVOICE_PROMPT = """
Read the document and return structured JSON.

Rules:

vendor:
Exactly as written.

currency:
ISO4217 code.

total_amount:
Integer.
No commas.
No symbols.

invoice_date:
YYYY-MM-DD

due_in_days:
Integer.

is_paid:
Boolean.

priority:
One of

low
normal
high
urgent

contact_email:
Lowercase.

line_items:
Array preserving order.

item_count:
Number of line items.

Return JSON only.
"""


# =====================================================
# Q9
# =====================================================

WORD_PROBLEM_PROMPT = """
Solve the arithmetic word problem carefully.

Rules:

1.
Ignore distractor numbers.

2.
Reason step by step.

3.
Double-check calculations.

Return ONLY JSON.

{
    "reasoning":"...",
    "answer":123
}

Requirements:

reasoning:
At least 80 characters.

answer:
Integer only.
"""