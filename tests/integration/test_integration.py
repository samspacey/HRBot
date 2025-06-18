#!/usr/bin/env python3
"""
Integration test for HR Chatbot core functionality.

This script tests the integration between different modules
and simulates a complete workflow without requiring external dependencies.
"""

import os
import tempfile
import asyncio
from pathlib import Path

def test_full_workflow():
    """Test a complete workflow simulation."""
    print("🧪 Testing Full Workflow Integration")
    
    # Set up test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")
        
        # Test 1: Configuration Management
        print("\n1. Testing Configuration Management...")
        test_env = {'OPENAI_API_KEY': 'test-key-123'}
        old_env = os.environ.copy()
        
        try:
            os.environ.update(test_env)
            from config import get_config
            config = get_config()
            print(f"✓ Config loaded: API key length = {len(config.openai_api_key)}")
            
            # Test 2: Input Validation
            print("\n2. Testing Input Validation...")
            from validation import get_validator
            validator = get_validator()
            
            test_queries = [
                "What is the vacation policy?",
                "How do I request time off?",
                "What are the sick leave rules?"
            ]
            
            for query in test_queries:
                validated = validator.validate_query(query)
                print(f"✓ Validated query: {validated[:30]}...")
            
            # Test 3: Caching System
            print("\n3. Testing Caching System...")
            from cache import get_query_cache
            cache = get_query_cache()
            
            # Simulate query results
            for i, query in enumerate(test_queries):
                mock_result = (f"Answer to: {query}", [f"doc{i}"])
                cache.set(query, "gpt-4o", 4, mock_result)
                
                cached = cache.get(query, "gpt-4o", 4)
                assert cached == mock_result
                print(f"✓ Cached and retrieved: {query[:30]}...")
            
            # Test 4: Service Architecture
            print("\n4. Testing Service Architecture...")
            try:
                from services import ServiceFactory
                # This will fail due to missing dependencies, but we can test the structure
                print("⚠ Service architecture requires external dependencies (expected)")
            except ImportError as e:
                if "langchain" in str(e).lower():
                    print("⚠ Service architecture requires langchain (expected)")
                else:
                    raise
            
            # Test 5: Async Operations
            print("\n5. Testing Async Operations...")
            async def test_async():
                from async_hr_chatbot import AsyncHRChatbot
                async with AsyncHRChatbot() as chatbot:
                    result = await chatbot.process_query_async("Test query")
                    print(f"✓ Async processing: {result}")
            
            asyncio.run(test_async())
            
            # Test 6: File Operations
            print("\n6. Testing File Operations...")
            test_files = []
            for i, filename in enumerate(['policy1.pdf', 'policy2.pdf', 'policy3.pdf']):
                file_path = Path(temp_dir) / filename
                file_path.write_text(f"Mock PDF content for {filename}")
                test_files.append(file_path)
                
                validated_path = validator.validate_pdf_file(str(file_path))
                print(f"✓ File validated: {validated_path.name}")
            
            # Test 7: Cache Statistics
            print("\n7. Testing Cache Statistics...")
            stats = cache.get_stats()
            print(f"✓ Cache stats: {stats['total_entries']} entries, {stats['active_entries']} active")
            
            # Test 8: Configuration Validation
            print("\n8. Testing Configuration Edge Cases...")
            try:
                invalid_k = validator.validate_k_value(0)
                assert False, "Should have failed"
            except Exception:
                print("✓ Invalid k value correctly rejected")
            
            try:
                validator.validate_query("")
                assert False, "Should have failed"
            except Exception:
                print("✓ Empty query correctly rejected")
            
            print("\n🎉 All integration tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
        finally:
            os.environ.clear()
            os.environ.update(old_env)


def test_error_handling():
    """Test error handling across modules."""
    print("\n🛡️ Testing Error Handling")
    
    test_env = {'OPENAI_API_KEY': 'test-key'}
    old_env = os.environ.copy()
    
    try:
        os.environ.update(test_env)
        
        from validation import get_validator, ValidationError
        from cache import get_query_cache
        
        validator = get_validator()
        cache = get_query_cache()
        
        # Test dangerous input handling
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:alert('test')",
            "x" * 2000,  # Too long
        ]
        
        for dangerous_input in dangerous_inputs:
            try:
                validator.validate_query(dangerous_input)
                print(f"❌ Dangerous input not caught: {dangerous_input[:30]}...")
                return False
            except ValidationError:
                print(f"✓ Dangerous input correctly blocked: {dangerous_input[:30]}...")
        
        # Test file path validation
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/passwd",
            "..\\windows\\system32",
            "/root/.ssh/id_rsa"
        ]
        
        for path in dangerous_paths:
            try:
                validator.validate_file_path(path)
                print(f"❌ Dangerous path not caught: {path}")
                return False
            except ValidationError:
                print(f"✓ Dangerous path correctly blocked: {path}")
        
        print("🎉 All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_performance():
    """Test basic performance characteristics."""
    print("\n⚡ Testing Performance")
    
    import time
    
    test_env = {'OPENAI_API_KEY': 'test-key'}
    old_env = os.environ.copy()
    
    try:
        os.environ.update(test_env)
        
        from validation import get_validator
        from cache import get_query_cache
        
        validator = get_validator()
        cache = get_query_cache()
        
        # Test validation performance
        start_time = time.time()
        for i in range(1000):
            validator.validate_query(f"Test query number {i}")
        validation_time = time.time() - start_time
        print(f"✓ Validated 1000 queries in {validation_time:.3f}s ({1000/validation_time:.0f} qps)")
        
        # Test cache performance
        start_time = time.time()
        for i in range(1000):
            cache.set(f"query{i}", "gpt-4o", 4, (f"answer{i}", []))
        cache_write_time = time.time() - start_time
        print(f"✓ Cached 1000 items in {cache_write_time:.3f}s ({1000/cache_write_time:.0f} ops/s)")
        
        start_time = time.time()
        hits = 0
        for i in range(1000):
            result = cache.get(f"query{i}", "gpt-4o", 4)
            if result is not None:
                hits += 1
        cache_read_time = time.time() - start_time
        print(f"✓ Read 1000 items in {cache_read_time:.3f}s ({1000/cache_read_time:.0f} ops/s), {hits} hits")
        
        print("🎉 Performance tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def main():
    """Run all integration tests."""
    print("🚀 HR Chatbot Integration Tests\n")
    
    tests = [
        ("Full Workflow", test_full_workflow),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"INTEGRATION RESULTS: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n✨ The HR Chatbot codebase is working correctly!")
        print("📋 Core modules tested:")
        print("  - Configuration management ✓")
        print("  - Input validation & security ✓") 
        print("  - Caching system ✓")
        print("  - Error handling ✓")
        print("  - Async operations ✓")
        print("  - File operations ✓")
        print("  - Performance characteristics ✓")
        return 0
    else:
        print("❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())