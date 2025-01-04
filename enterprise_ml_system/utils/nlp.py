from typing import Dict, List, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging

logger = logging.getLogger(__name__)

class AdvancedNLPPipeline:
    """Handles advanced NLP tasks using transformer models."""

    def __init__(self, model_config: Dict[str, Any]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['base_model'])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_config['base_model'])

        # Set up specialized pipelines
        self.summarizer = pipeline("summarization", model=model_config['summarization_model'])
        self.qa_pipeline = pipeline("question-answering", model=model_config['qa_model'])
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_config['sentiment_model'])

    async def process_text(self, text: str, tasks: List[str]) -> Dict[str, Any]:
        """Process text through multiple NLP pipelines."""
        results = {}
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Process different tasks
            for task in tasks:
                if task == 'classification':
                    outputs = self.model(**inputs)
                    results['classification'] = outputs.logits.softmax(1).tolist()

                elif task == 'summarization':
                    summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
                    results['summary'] = summary[0]['summary_text']

                elif task == 'qa':
                    # Assuming question is provided in text format: "Q: ...? context"
                    parts = text.split('?')
                    if len(parts) == 2:
                        question = parts[0].replace('Q:', '').strip()
                        context = parts[1].strip()
                        answer = self.qa_pipeline(question=question, context=context)
                        results['qa_answer'] = answer

                elif task == 'sentiment':
                    sentiment = self.sentiment_pipeline(text)
                    results['sentiment'] = sentiment

            return results

        except Exception as e:
            logger.error(f"NLP processing failed: {e}")
            raise