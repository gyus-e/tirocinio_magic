def build_cag_context(system_prompt: str, document_texts: list[str]) -> str:
    # TODO: Explicitly tell to end the answer with a <|endoftext|> token
    return f"""
    <|system|>
    {system_prompt}
    <|user|>
    Contesto:
    {document_texts}
    """
