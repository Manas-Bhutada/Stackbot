from app.core.embed import EmbeddingManager

manager = EmbeddingManager()
manager.load_data("data/qna.json")
manager.build_index()
