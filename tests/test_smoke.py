import importlib.util
from pathlib import Path

def test_project_modules_importable():
    """
    Minimal smoke test to ensure backend modules import without executing UI,
    network, vectorstore, or model initialization.
    """
    import src.graph
    import src.tools
    import src.rag
    import src.llm
    import src.database
    import src.config

    assert True


def test_app_module_exists_and_resolves():
    """
    Ensures the Streamlit UI module exists and can be resolved by Python
    without actually importing it (avoids execution side effects).
    """
    app_path = Path("src/app.py")
    assert app_path.exists()

    spec = importlib.util.find_spec("src.app")
    assert spec is not None
