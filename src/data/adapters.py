from abc import ABC, abstractmethod

class BaseDatasetAdapter(ABC):
    @abstractmethod
    def adapt(self, example: dict) -> str:
        pass


def _clean_text(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Normalize whitespace and strip
    text = " ".join(text.split())
    return text.strip()


class TinyStoriesAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        return _clean_text(example.get("text"))


class WikiTextAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        return _clean_text(example.get("text"))


class OpenWebTextAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        return _clean_text(example.get("text"))


class AgNewsAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        text = example.get("text")
        if text is None:
            # Fallback for other namespace versions containing separated title and description
            title = example.get("title", "")
            desc = example.get("description", "")
            text = f"{title}\n{desc}".strip()
        return _clean_text(text)


class DailyDialogAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        dialog = example.get("dialog", [])
        if isinstance(dialog, list):
            return _clean_text("\n".join(str(x) for x in dialog))
        return _clean_text(dialog)


class SquadAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        context = example.get("context", "")
        question = example.get("question", "")
        text = f"Context: {context}\nQuestion: {question}"
        return _clean_text(text)


class Eli5Adapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        answers = example.get("answers", [])
        texts = []

        if isinstance(answers, dict):
            value = answers.get("text", [])
            if isinstance(value, list):
                texts.extend(str(x) for x in value)
            elif value is not None:
                texts.append(str(value))

        elif isinstance(answers, list):
            for ans in answers:
                if isinstance(ans, dict):
                    val = ans.get("text")
                    if isinstance(val, list):
                        texts.extend(str(x) for x in val)
                    elif val is not None:
                        texts.append(str(val))
                elif ans is not None:
                    texts.append(str(ans))

        return _clean_text("\n".join(texts))


class YelpReviewAdapter(BaseDatasetAdapter):
    def adapt(self, example: dict) -> str:
        return _clean_text(example.get("text"))


DATASET_REGISTRY = {
    "roneneldan/TinyStories": TinyStoriesAdapter,
    "Salesforce/wikitext": WikiTextAdapter,
    "Skylion007/openwebtext": OpenWebTextAdapter,
    "fancyzhx/ag_news": AgNewsAdapter,
    "DeepPavlov/daily_dialog": DailyDialogAdapter,
    "rajpurkar/squad": SquadAdapter,
    "dany0407/eli5_category": Eli5Adapter,
    "Yelp/yelp_review_full": YelpReviewAdapter,
}


def get_adapter(dataset_name: str) -> BaseDatasetAdapter:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' is not registered. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]()
