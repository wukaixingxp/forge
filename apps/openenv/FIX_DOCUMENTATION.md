# Fix for ModuleNotFoundError: No module named 'julia_utils'

## Problem

The application was crashing with the following error when using Monarch actors:

```
ModuleNotFoundError: No module named 'julia_utils'
```

This error occurred when remote Monarch actors tried to unpickle function references that were loaded from the `julia_utils` module.

## Root Cause

The issue happened because:

1. The main process loads functions from `julia_utils` using `load_function_from_string()`
2. These functions are passed as parameters to actor classes (`GenericDatasetActor`, `GenericRewardActor`)
3. When actors are spawned as remote actors, the function objects are pickled and sent to remote processes
4. During unpickling, Python needs to import the `julia_utils` module
5. **The openenv directory wasn't in `sys.path` yet** because:
   - The unpickling happens during actor initialization (when deserializing constructor parameters)
   - The `setup()` endpoint runs AFTER actor initialization
   - Therefore, `sys.path` wasn't modified before unpickling occurred

## Solution

Added module-level code to `/home/kaiwu/work/kaiwu/forge/apps/openenv/main.py` that adds the openenv directory to `sys.path` BEFORE any actor definitions:

```python
# CRITICAL: Add openenv directory to sys.path at module level
# This ensures that when remote actors unpickle function references (e.g., julia_utils functions),
# the module can be imported successfully. This must happen BEFORE any actor definitions.
_openenv_dir = Path(__file__).parent
if str(_openenv_dir) not in sys.path:
    sys.path.insert(0, str(_openenv_dir))
```

This code runs when the module is first imported, ensuring that:
- Remote actors that import `main.py` will have the openenv directory in their `sys.path`
- Functions from `julia_utils` can be successfully unpickled in remote processes
- The fix happens early enough to prevent the ModuleNotFoundError

## Testing

Created comprehensive tests to verify the fix:

1. **test_module_import.py** - Tests basic import and pickling functionality
2. **test_monarch_actor_simulation.py** - Simulates the exact Monarch actor scenario where a remote process receives pickled functions

Both test suites pass successfully, confirming that:
- `julia_utils` can be imported after importing `main.py`
- Functions from `julia_utils` can be pickled and unpickled across process boundaries
- Remote actors can successfully deserialize function references

## Files Modified

- `/home/kaiwu/work/kaiwu/forge/apps/openenv/main.py` - Added module-level sys.path setup

## Files Added

- `/home/kaiwu/work/kaiwu/forge/apps/openenv/test_module_import.py` - Basic import/pickle tests
- `/home/kaiwu/work/kaiwu/forge/apps/openenv/test_monarch_actor_simulation.py` - Comprehensive simulation tests

## Verification

Run tests to verify the fix:
```bash
cd /home/kaiwu/work/kaiwu/forge/apps/openenv
python test_module_import.py
python test_monarch_actor_simulation.py
```

Both should show "✓ All tests passed!"
