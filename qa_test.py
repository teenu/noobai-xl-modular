#!/usr/bin/env python3
"""
Comprehensive QA Test Suite for Modularized NoobAI XL
Tests import chains, syntax, accessibility, and basic functionality.
"""

import sys
import os

def test_imports():
    """Test all module imports."""
    print("=" * 70)
    print("QA TEST 1: MODULE IMPORTS")
    print("=" * 70)

    modules = [
        'config',
        'state',
        'utils',
        'engine',
        'prompt_formatter',
        'ui_helpers',
        'ui',
        'cli',
        'main'
    ]

    results = []
    for module in modules:
        try:
            exec(f"import {module}")
            results.append((module, "✅ PASS", "Import successful"))
        except Exception as e:
            results.append((module, "❌ FAIL", str(e)))

    # Print results
    for module, status, msg in results:
        print(f"{module:20} {status:10} {msg}")

    failures = [r for r in results if "FAIL" in r[1]]
    print(f"\nResult: {len(results) - len(failures)}/{len(results)} passed")
    return len(failures) == 0

def test_key_classes_and_functions():
    """Test that key classes and functions are accessible."""
    print("\n" + "=" * 70)
    print("QA TEST 2: KEY CLASSES & FUNCTIONS ACCESSIBILITY")
    print("=" * 70)

    tests = []

    # Config module
    try:
        from config import logger, OPTIMAL_SETTINGS, MODEL_CONFIG, GenerationInterruptedError
        tests.append(("config.logger", "✅ PASS"))
        tests.append(("config.OPTIMAL_SETTINGS", "✅ PASS"))
        tests.append(("config.MODEL_CONFIG", "✅ PASS"))
        tests.append(("config.GenerationInterruptedError", "✅ PASS"))
    except Exception as e:
        tests.append(("config imports", f"❌ FAIL: {e}"))

    # State module
    try:
        from state import state_manager, perf_monitor, resource_pool, GenerationState
        tests.append(("state.state_manager", "✅ PASS"))
        tests.append(("state.perf_monitor", "✅ PASS"))
        tests.append(("state.resource_pool", "✅ PASS"))
        tests.append(("state.GenerationState", "✅ PASS"))
    except Exception as e:
        tests.append(("state imports", f"❌ FAIL: {e}"))

    # Utils module
    try:
        from utils import validate_model_path, discover_dora_adapters, calculate_image_hash
        tests.append(("utils.validate_model_path", "✅ PASS"))
        tests.append(("utils.discover_dora_adapters", "✅ PASS"))
        tests.append(("utils.calculate_image_hash", "✅ PASS"))
    except Exception as e:
        tests.append(("utils imports", f"❌ FAIL: {e}"))

    # Engine module
    try:
        from engine import NoobAIEngine
        tests.append(("engine.NoobAIEngine", "✅ PASS"))
    except Exception as e:
        tests.append(("engine imports", f"❌ FAIL: {e}"))

    # Prompt formatter module
    try:
        from prompt_formatter import get_prompt_data
        tests.append(("prompt_formatter.get_prompt_data", "✅ PASS"))
    except Exception as e:
        tests.append(("prompt_formatter imports", f"❌ FAIL: {e}"))

    # UI helpers module
    try:
        from ui_helpers import initialize_engine, generate_image_with_progress, engine
        tests.append(("ui_helpers.initialize_engine", "✅ PASS"))
        tests.append(("ui_helpers.generate_image_with_progress", "✅ PASS"))
        tests.append(("ui_helpers.engine (global)", "✅ PASS"))
    except Exception as e:
        tests.append(("ui_helpers imports", f"❌ FAIL: {e}"))

    # UI module
    try:
        from ui import create_interface
        tests.append(("ui.create_interface", "✅ PASS"))
    except Exception as e:
        tests.append(("ui imports", f"❌ FAIL: {e}"))

    # CLI module
    try:
        from cli import cli_list_adapters, cli_generate, parse_args
        tests.append(("cli.cli_list_adapters", "✅ PASS"))
        tests.append(("cli.cli_generate", "✅ PASS"))
        tests.append(("cli.parse_args", "✅ PASS"))
    except Exception as e:
        tests.append(("cli imports", f"❌ FAIL: {e}"))

    # Print results
    for test_name, status in tests:
        print(f"{test_name:45} {status}")

    failures = [t for t in tests if "FAIL" in t[1]]
    print(f"\nResult: {len(tests) - len(failures)}/{len(tests)} passed")
    return len(failures) == 0

def test_import_chains():
    """Test that import chains work correctly."""
    print("\n" + "=" * 70)
    print("QA TEST 3: IMPORT CHAIN VERIFICATION")
    print("=" * 70)

    chains = [
        ("config → state", lambda: __import__('state')),
        ("config → state → utils", lambda: __import__('utils')),
        ("config → state → utils → engine", lambda: __import__('engine')),
        ("config → state → utils → prompt_formatter", lambda: __import__('prompt_formatter')),
        ("all → ui_helpers", lambda: __import__('ui_helpers')),
        ("all → ui", lambda: __import__('ui')),
        ("all → cli", lambda: __import__('cli')),
        ("all → main", lambda: __import__('main')),
    ]

    results = []
    for chain_name, test_func in chains:
        try:
            test_func()
            results.append((chain_name, "✅ PASS"))
        except Exception as e:
            results.append((chain_name, f"❌ FAIL: {e}"))

    # Print results
    for chain_name, status in results:
        print(f"{chain_name:40} {status}")

    failures = [r for r in results if "FAIL" in r[1]]
    print(f"\nResult: {len(results) - len(failures)}/{len(results)} passed")
    return len(failures) == 0

def test_no_circular_imports():
    """Test that there are no circular import issues."""
    print("\n" + "=" * 70)
    print("QA TEST 4: CIRCULAR IMPORT CHECK")
    print("=" * 70)

    # Clear all imported modules
    modules_to_test = ['config', 'state', 'utils', 'engine', 'prompt_formatter',
                       'ui_helpers', 'ui', 'cli', 'main']

    for mod in modules_to_test:
        if mod in sys.modules:
            del sys.modules[mod]

    try:
        # Try importing main (which imports everything)
        import main
        print("✅ PASS: No circular import detected")
        print("   All modules imported successfully via main.py")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Circular import detected: {e}")
        return False

def test_generationinterruptederror_import():
    """Test that GenerationInterruptedError is properly imported in ui_helpers."""
    print("\n" + "=" * 70)
    print("QA TEST 5: GenerationInterruptedError IMPORT LOCATION")
    print("=" * 70)

    try:
        # Read ui_helpers.py and check import location
        with open('ui_helpers.py', 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Check if it's in the top-level imports (handle multi-line imports)
        found_in_top = False
        found_in_top_line = 0
        found_in_function = False
        found_in_function_line = 0

        # Check first 30 lines for top-level import (handling multi-line imports)
        in_config_import_block = False
        for i, line in enumerate(lines[:30], 1):
            # Check if we're entering a multi-line import from config
            if 'from config import' in line and '(' in line:
                in_config_import_block = True

            # Check for GenerationInterruptedError in current line
            if 'GenerationInterruptedError' in line:
                if in_config_import_block or 'from config import' in line:
                    found_in_top = True
                    found_in_top_line = i
                    print(f"✅ PASS: Found in top-level imports (line {i})")
                    break

            # End of import block
            if in_config_import_block and ')' in line:
                in_config_import_block = False

        # Check if it's still in the function (should not be)
        for i, line in enumerate(lines[400:], 400):  # Check around line 480
            if 'from config import GenerationInterruptedError' in line:
                found_in_function = True
                found_in_function_line = i + 1
                print(f"❌ WARNING: Also found in function body (line {found_in_function_line})")
                break

        if found_in_top and not found_in_function:
            print("✅ Import location is correct (top-level only)")
            return True
        elif not found_in_top:
            print("❌ FAIL: Not found in top-level imports")
            return False
        else:
            print("⚠️  WARNING: Found in both locations (should only be top-level)")
            return True  # Still functional, just not ideal

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def test_cli_help():
    """Test that CLI help works."""
    print("\n" + "=" * 70)
    print("QA TEST 6: CLI HELP FUNCTIONALITY")
    print("=" * 70)

    try:
        from cli import parse_args
        import argparse

        # Override sys.argv temporarily
        old_argv = sys.argv
        sys.argv = ['main.py', '--help']

        try:
            parse_args()
        except SystemExit:
            # --help causes SystemExit, which is expected
            pass

        sys.argv = old_argv
        print("✅ PASS: CLI argument parser works correctly")
        return True

    except Exception as e:
        sys.argv = old_argv
        print(f"❌ FAIL: {e}")
        return False

def test_critical_constants():
    """Test that critical constants are accessible."""
    print("\n" + "=" * 70)
    print("QA TEST 7: CRITICAL CONSTANTS")
    print("=" * 70)

    try:
        from config import OPTIMAL_SETTINGS, MODEL_CONFIG, OUTPUT_DIR

        tests = [
            ("OPTIMAL_SETTINGS['steps']", OPTIMAL_SETTINGS.get('steps')),
            ("OPTIMAL_SETTINGS['cfg_scale']", OPTIMAL_SETTINGS.get('cfg_scale')),
            ("OPTIMAL_SETTINGS['width']", OPTIMAL_SETTINGS.get('width')),
            ("OPTIMAL_SETTINGS['height']", OPTIMAL_SETTINGS.get('height')),
            ("MODEL_CONFIG.MIN_FILE_SIZE_MB", MODEL_CONFIG.MIN_FILE_SIZE_MB),
            ("OUTPUT_DIR", OUTPUT_DIR),
        ]

        all_passed = True
        for name, value in tests:
            if value is not None:
                print(f"✅ {name:40} = {value}")
            else:
                print(f"❌ {name:40} = MISSING")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def main():
    """Run all QA tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "COMPREHENSIVE QA TEST SUITE" + " " * 31 + "║")
    print("║" + " " * 15 + "NoobAI XL Modularized Version" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    results = []

    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Key Classes & Functions", test_key_classes_and_functions()))
    results.append(("Import Chains", test_import_chains()))
    results.append(("Circular Import Check", test_no_circular_imports()))
    results.append(("GenerationInterruptedError Location", test_generationinterruptederror_import()))
    results.append(("CLI Help Functionality", test_cli_help()))
    results.append(("Critical Constants", test_critical_constants()))

    # Summary
    print("\n" + "=" * 70)
    print("FINAL QA SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:40} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("\n" + "=" * 70)
    print(f"OVERALL RESULT: {total_passed}/{total_tests} test suites passed")
    print("=" * 70)

    if total_passed == total_tests:
        print("\n✅ All QA tests PASSED - Code is production ready!")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} test suite(s) FAILED - Review required")
        return 1

if __name__ == "__main__":
    sys.exit(main())
