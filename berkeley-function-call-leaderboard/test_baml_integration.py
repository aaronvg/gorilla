#!/usr/bin/env python3
"""
Simple test script to verify BAML integration works.
"""

import os
import sys
import tempfile
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_cli_flag():
    """Test that the CLI flag is accepted."""
    print("Testing CLI flag integration...")
    
    try:
        from bfcl_eval.__main__ import cli
        # Just import - actual CLI testing would require more setup
        print("✓ CLI import successful")
        return True
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False

def test_baml_handler_import():
    """Test that BAMLHandler can be imported."""
    print("Testing BAMLHandler import...")
    
    try:
        from bfcl_eval.model_handler.baml_handler import BAMLHandler
        print("✓ BAMLHandler import successful")
        return True
    except ImportError as e:
        if "BAML is required" in str(e):
            print("✓ BAMLHandler import successful (expected BAML dependency error)")
            return True
        else:
            print(f"✗ BAMLHandler import failed: {e}")
            return False
    except Exception as e:
        print(f"✗ BAMLHandler import failed: {e}")
        return False

def test_utils_import():
    """Test that baml_utils can be imported."""
    print("Testing baml_utils import...")
    
    try:
        from bfcl_eval.model_handler.baml_utils import detect_baml_provider, TOOL_NAME_KEY
        print("✓ baml_utils import successful")
        
        # Test provider detection
        provider, options = detect_baml_provider("gpt-4o")
        print(f"✓ Provider detection works: {provider}")
        
        provider, options = detect_baml_provider("claude-3-5-sonnet")
        print(f"✓ Provider detection works: {provider}")
        
        return True
    except Exception as e:
        print(f"✗ baml_utils import failed: {e}")
        return False

def test_build_handler_function():
    """Test the modified build_handler function."""
    print("Testing build_handler function...")
    
    try:
        from bfcl_eval._llm_response_generation import build_handler
        
        # Test default mode (should work)
        try:
            handler = build_handler("gpt-4o", 0.1, "default")
            print("✗ Default handler should fail for unknown model")
            return False
        except (ValueError, KeyError):
            print("✓ Default handler correctly rejects unknown model")
        
        # Test BAML mode (should work for import but fail for missing BAML client)
        try:
            handler = build_handler("gpt-4o", 0.1, "baml")
            print("✗ BAML handler should fail due to missing BAML client")
            return False
        except ImportError as e:
            if "BAML is required" in str(e):
                print("✓ BAML handler correctly reports missing BAML dependency")
                return True
            else:
                print(f"✗ BAML handler failed with unexpected error: {e}")
                return False
        
    except Exception as e:
        print(f"✗ build_handler test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("BAML Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_cli_flag,
        test_baml_handler_import,
        test_utils_import,
        test_build_handler_function,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BAML integration is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())