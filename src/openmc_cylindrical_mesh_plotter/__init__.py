from .core import *

try:
    import streamlit
    from .app import *
except:
    pass
    # streamlit is not in installed so GUI will not work
