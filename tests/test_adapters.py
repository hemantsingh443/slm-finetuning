import unittest
from src.data.adapters import (
    _clean_text,
    TinyStoriesAdapter,
    WikiTextAdapter,
    OpenWebTextAdapter,
    AgNewsAdapter,
    DailyDialogAdapter,
    SquadAdapter,
    Eli5Adapter,
    YelpReviewAdapter,
)

class TestAdapters(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(_clean_text("Hello   World"), "Hello World")
        self.assertEqual(_clean_text("  line1\nline2  "), "line1 line2")
        self.assertEqual(_clean_text(None), "")
        self.assertEqual(_clean_text(123), "123")

    def test_tinystories_adapter(self):
        adapter = TinyStoriesAdapter()
        self.assertEqual(adapter.adapt({"text": " Once upon a time   "}), "Once upon a time")
        self.assertEqual(adapter.adapt({}), "")

    def test_wikitext_adapter(self):
        adapter = WikiTextAdapter()
        self.assertEqual(adapter.adapt({"text": "Some text  "}), "Some text")

    def test_openwebtext_adapter(self):
        adapter = OpenWebTextAdapter()
        self.assertEqual(adapter.adapt({"text": "Web content"}), "Web content")

    def test_agnews_adapter(self):
        adapter = AgNewsAdapter()
        # Direct text mapping
        self.assertEqual(adapter.adapt({"text": "News article"}), "News article")
        # Fallback split mapping
        self.assertEqual(adapter.adapt({"title": "Headline", "description": "Story body"}), "Headline Story body")

    def test_dailydialog_adapter(self):
        adapter = DailyDialogAdapter()
        self.assertEqual(adapter.adapt({"dialog": ["Hi", "Hello", "Bye"]}), "Hi Hello Bye")
        self.assertEqual(adapter.adapt({"dialog": "Single turn"}), "Single turn")

    def test_squad_adapter(self):
        adapter = SquadAdapter()
        example = {"context": "This is context.", "question": "What is it?"}
        self.assertEqual(adapter.adapt(example), "Context: This is context. Question: What is it?")

    def test_eli5_adapter(self):
        adapter = Eli5Adapter()
        # Test dict of lists structure (e.g. dany0407/eli5_category)
        example_dict = {"answers": {"text": ["Answer A", "Answer B"]}}
        self.assertEqual(adapter.adapt(example_dict), "Answer A Answer B")
        
        # Test list of dicts structure
        example_list = {"answers": [{"text": "Ans 1"}, {"text": "Ans 2"}]}
        self.assertEqual(adapter.adapt(example_list), "Ans 1 Ans 2")

    def test_yelpreview_adapter(self):
        adapter = YelpReviewAdapter()
        self.assertEqual(adapter.adapt({"text": "Great service!"}), "Great service!")


if __name__ == "__main__":
    unittest.main()
