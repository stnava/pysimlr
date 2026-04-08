import pysimlr

def test_import():
    try:
        import pysimlr
        assert True
    except ImportError:
        assert False
