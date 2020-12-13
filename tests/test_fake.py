from src.models.fakemodel import Fakemodel


def test_fakemodel():
    assert Fakemodel().say() == 'Hello'
