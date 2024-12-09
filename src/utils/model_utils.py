def verify_model_compatibility(model_name: str, expected_dimension: int) -> bool:
    """Verify that the selected model matches the expected embedding dimension."""
    from sentence_transformers import SentenceTransformer
    
    try:
        model = SentenceTransformer(model_name)
        actual_dimension = model.get_sentence_embedding_dimension()
        return actual_dimension == expected_dimension
    except Exception as e:
        logger.error(f"Error verifying model compatibility: {str(e)}")
        return False 