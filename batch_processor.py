
from review_analyzer import review_analysis_chain


def process_reviews(
    reviews: list[dict],
    max_concurrency: int = 5
) -> list[dict]:
    """
    Analyse multiple customer reviews using LangChain batch processing.
    """

    if not reviews:
        return []

    # Prepare inputs for the LangChain chain
    inputs = [
        {
            "review_id": review["review_id"],
            "date": review["date"],
            "platform": review["platform"],
            "rating": review["rating"],
            "review_text": review["review_text"],
        }
        for review in reviews
    ]

    # Process reviews in batches
    results = review_analysis_chain.batch(
        inputs,
        config={"max_concurrency": max_concurrency}
    )

    # Convert Pydantic objects into dictionaries
    return [
        result.model_dump()
        for result in results
    ]
