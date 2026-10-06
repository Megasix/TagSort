"""TagSort: read specimen ID tags in photos.

Only names exported from this module are part of the public API.
"""

from tagsort.errors import PatternError, ProfileError, TagSortError
from tagsort.profile import Profile, TagSpec
from tagsort.types import Candidate, ImageInfo, Point, ReadResult, Tag, TagSource, TagStatus

__all__ = [
    "Candidate",
    "ImageInfo",
    "PatternError",
    "Point",
    "Profile",
    "ProfileError",
    "ReadResult",
    "Tag",
    "TagSortError",
    "TagSource",
    "TagSpec",
    "TagStatus",
    "__version__",
]

__version__ = "0.0.1"
