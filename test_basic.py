#!/usr/bin/env python3
"""
Basic test script for HR Chatbot core modules.

This script tests the basic functionality of the core modules
without requiring external dependencies like langchain.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_config_module():
    """Test the config module."""
    print("Testing config module...")
    
    # Test with API key
    test_env = {'OPENAI_API_KEY': 'test-key-123'}
    old_env = os.environ.copy()
    
    try:
        # Set test environment
        os.environ.update(test_env)
        
        # Import and test
        from config import load_config, validate_config
        
        config = load_config()
        assert config.openai_api_key == 'test-key-123'
        assert config.chunk_size == 500
        print("✓ Config module basic functionality works")
        
        # Test validation
        validate_config(config)
        print("✓ Config validation works")
        
    except Exception as e:
        print(f"✗ Config module failed: {e}")
        return False
    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(old_env)
    
    return True


def test_validation_module():
    """Test the validation module."""
    print("Testing validation module...")
    
    try:
        from validation import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Test valid query
        result = validator.validate_query("What is the vacation policy?")
        assert result == "What is the vacation policy?"
        print("✓ Query validation works")
        
        # Test empty query
        try:
            validator.validate_query("")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✓ Empty query validation works")
        
        # Test dangerous content
        try:
            validator.validate_query("<script>alert('xss')</script>")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✓ Dangerous content detection works")
        
        # Test string sanitization
        sanitized = validator.sanitize_string("<script>test</script>")
        assert "<script>" not in sanitized
        print("✓ String sanitization works")
        
    except Exception as e:
        print(f"✗ Validation module failed: {e}")
        return False
    
    return True


def test_cache_module():
    """Test the cache module."""
    print("Testing cache module...")
    
    try:
        from cache import QueryCache, EmbeddingCache
        
        # Test query cache
        cache = QueryCache(max_size=5, ttl=3600)
        
        # Test cache operations
        test_result = ("test answer", [])
        cache.set("test query", "gpt-4o", 4, test_result)
        
        cached = cache.get("test query", "gpt-4o", 4)
        assert cached == test_result
        print("✓ Query cache works")
        
        # Test cache miss
        miss = cache.get("nonexistent", "gpt-4o", 4)
        assert miss is None
        print("✓ Cache miss handling works")
        
        # Test embedding cache with temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            emb_cache = EmbeddingCache(temp_dir)
            
            test_embedding = [0.1, 0.2, 0.3]
            emb_cache.set_embedding("test content", "test-model", test_embedding)
            
            cached_emb = emb_cache.get_embedding("test content", "test-model")
            assert cached_emb == test_embedding
            print("✓ Embedding cache works")
        
    except Exception as e:
        print(f"✗ Cache module failed: {e}")
        return False
    
    return True


def test_file_operations():
    """Test basic file operations."""
    print("Testing file operations...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_pdf = temp_path / "test.pdf"
            test_pdf.write_text("dummy pdf content")
            
            test_txt = temp_path / "test.txt"
            test_txt.write_text("dummy text content")
            
            from validation import get_validator
            validator = get_validator()
            
            # Test folder validation
            validated_folder = validator.validate_folder_path(str(temp_path))
            assert validated_folder.exists()
            print("✓ Folder validation works")
            
            # Test PDF file validation
            validated_pdf = validator.validate_pdf_file(str(test_pdf))
            assert validated_pdf.exists()
            print("✓ PDF file validation works")
            
    except Exception as e:
        print(f"✗ File operations failed: {e}")
        return False
    
    return True


def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing basic imports...")
    
    modules_to_test = [
        'config',
        'validation', 
        'cache',
    ]
    
    # Test modules that don't require external dependencies
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} module imports successfully")
        except Exception as e:
            print(f"✗ {module_name} module import failed: {e}")
            return False
    
    # Test modules that require external dependencies
    external_modules = [
        'chunking',
        'hr_chatbot', 
        'services',
        'async_hr_chatbot'
    ]
    
    for module_name in external_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name} module imports successfully")
        except ImportError as e:
            if "langchain" in str(e) or "openai" in str(e):
                print(f"⚠ {module_name} requires external dependencies (expected): {e}")
            else:
                print(f"✗ {module_name} has unexpected import error: {e}")
                return False
        except Exception as e:
            print(f"✗ {module_name} has syntax/other error: {e}")
            return False
    
    return True


def test_environment_setup():
    """Test environment setup."""
    print("Testing environment setup...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    
    # Check for required files
    required_files = [
        'config.py',
        'validation.py',
        'cache.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting HR Chatbot Basic Tests\n")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Basic Imports", test_basic_imports),
        ("Config Module", test_config_module),
        ("Validation Module", test_validation_module),
        ("Cache Module", test_cache_module),
        ("File Operations", test_file_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())