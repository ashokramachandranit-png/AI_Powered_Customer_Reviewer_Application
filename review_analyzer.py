
from config import get_llm
from schemas import ReviewAnalysis
from prompts.review_analysis_prompt import review_analysis_prompt


# Create the OpenAI model
llm = get_llm()

# Enable structured output using the Pydantic schema
structured_llm = llm.with_structured_output(ReviewAnalysis)

# Build the LangChain processing chain
review_analysis_chain = review_analysis_prompt | structured_llm


def analyze_review(review: dict) -> dict:
    """
    Analyse one customer review and return structured results.
    """

    result = review_analysis_chain.invoke(
        {
            "review_id": review["review_id"],
            "date": review["date"],
            "platform": review["platform"],
            "rating": review["rating"],
            "review_text": review["review_text"],
        }
    )

    return result.model_dump()
