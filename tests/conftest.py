import pytest

from dermavit.images import MemoryImageSource
from dermavit.metadata import validate_metadata
from dermavit.synthetic import make_dataset


@pytest.fixture(scope="session")
def synth():
    meta, images, test_meta, test_images = make_dataset(n_lesions=260, size=32, seed=3, n_test=60)
    return validate_metadata(meta), images, test_meta, test_images


@pytest.fixture(scope="session")
def meta(synth):
    return synth[0]


@pytest.fixture(scope="session")
def source(synth):
    return MemoryImageSource({**synth[1], **synth[3]})
