import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streamlit.testing.v1 import AppTest

def test_all_pages_and_interactions():
    print("Testing Streamlit pages via AppTest...")
    
    # 1. Test Home page
    at = AppTest.from_file(str(BASE_DIR / "app.py"))
    at.run()
    assert not at.exception, f"Home page exception: {at.exception}"
    print("[PASS] Page 1: Dashboard / Home loaded cleanly.")

    # 2. Test Risk Prediction page
    at.sidebar.radio[0].set_value("⚡ Risk Prediction").run()
    assert not at.exception, f"Risk Prediction page exception: {at.exception}"
    print("[PASS] Page 2: Risk Prediction loaded cleanly.")

    # Test clicking persona preset button
    if len(at.button) > 0:
        at.button[0].click().run()
        assert not at.exception, f"Persona preset click exception: {at.exception}"
        print("[PASS] Persona preset click executed successfully.")

    # 3. Test Model Performance page
    at.sidebar.radio[0].set_value("📊 Model Performance").run()
    assert not at.exception, f"Model Performance page exception: {at.exception}"
    print("[PASS] Page 3: Model Performance loaded cleanly.")

    # 4. Test Feature Insights page
    at.sidebar.radio[0].set_value("🔍 Feature Insights").run()
    assert not at.exception, f"Feature Insights page exception: {at.exception}"
    print("[PASS] Page 4: Feature Insights loaded cleanly.")

    # 5. Test Responsible AI page
    at.sidebar.radio[0].set_value("⚖️ Responsible AI & Limitations").run()
    assert not at.exception, f"Responsible AI page exception: {at.exception}"
    print("[PASS] Page 5: Responsible AI & Limitations loaded cleanly.")

    print("\nALL 5 STREAMLIT APP PAGES AND INTERACTIONS PASSED WITH ZERO EXCEPTIONS!")

if __name__ == "__main__":
    test_all_pages_and_interactions()
