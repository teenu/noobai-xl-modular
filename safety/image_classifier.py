"""Post-generation image safety classification.

Provides integration points for NSFW and CSAM detection on generated images.
This module defines the interface — concrete implementations plug in via
the classifier registry.

Classifier priority (defense in depth):
1. Local classifier (NudeNet, WD-Tagger) — fast, no network call
2. Commercial API (Hive Moderation, Azure Content Safety) — higher accuracy
3. Fallback: block if any classifier is unavailable and strict mode is on
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol
from PIL import Image

logger = logging.getLogger(__name__)


class Rating(Enum):
    """Content rating levels."""
    SAFE = "safe"
    QUESTIONABLE = "questionable"
    EXPLICIT = "explicit"
    BLOCKED = "blocked"  # CSAM or prohibited content


@dataclass
class ClassificationResult:
    """Result from an image safety classifier."""
    rating: Rating
    confidence: float = 0.0
    reason: Optional[str] = None
    classifier_name: str = ""
    details: dict = field(default_factory=dict)


class ImageClassifier(Protocol):
    """Protocol for image safety classifiers."""

    @property
    def name(self) -> str: ...

    def classify(self, image: Image.Image) -> ClassificationResult: ...

    def is_available(self) -> bool: ...


class ClassifierPipeline:
    """Runs an image through a chain of safety classifiers.

    Returns the most restrictive rating across all classifiers.
    If any classifier returns BLOCKED, the image is blocked.
    """

    def __init__(self, strict: bool = True):
        self._classifiers: list[ImageClassifier] = []
        self._strict = strict

    def register(self, classifier: ImageClassifier) -> None:
        """Add a classifier to the pipeline."""
        if classifier.is_available():
            self._classifiers.append(classifier)
            logger.info(f"Registered safety classifier: {classifier.name}")
        else:
            logger.warning(f"Classifier unavailable, skipped: {classifier.name}")

    @property
    def has_classifiers(self) -> bool:
        return len(self._classifiers) > 0

    def classify(self, image: Image.Image) -> ClassificationResult:
        """Run all classifiers and return the most restrictive result."""
        if not self._classifiers:
            if self._strict:
                return ClassificationResult(
                    rating=Rating.BLOCKED,
                    reason="No safety classifiers available (strict mode)",
                    classifier_name="pipeline",
                )
            return ClassificationResult(
                rating=Rating.SAFE,
                classifier_name="pipeline",
                reason="No classifiers registered (permissive mode)",
            )

        worst = ClassificationResult(rating=Rating.SAFE, classifier_name="pipeline")
        _rating_severity = {Rating.SAFE: 0, Rating.QUESTIONABLE: 1, Rating.EXPLICIT: 2, Rating.BLOCKED: 3}

        for classifier in self._classifiers:
            try:
                result = classifier.classify(image)
                if _rating_severity[result.rating] > _rating_severity[worst.rating]:
                    worst = result
                if result.rating == Rating.BLOCKED:
                    logger.warning(f"Image blocked by {classifier.name}: {result.reason}")
                    return result
            except Exception as e:
                logger.error(f"Classifier {classifier.name} failed: {e}")
                if self._strict:
                    return ClassificationResult(
                        rating=Rating.BLOCKED,
                        reason=f"Classifier error in strict mode: {classifier.name}",
                        classifier_name=classifier.name,
                    )

        return worst
