# Files Involved ONLY When Prefix Caching is ENABLED

## Core Prefix Cache Files (v1/core/)

### 1. `kv_cache_manager.py`
**Functions involved**:
- `get_computed_blocks()` - Look up cached blocks for a request
- `cache_blocks()` - Store computed blocks in cache

**Only called when**: `enable_prefix_caching=True`

### 2. `kv_cache_coordinator.py`
**Functions involved**:
- `find_longest_cache_hit()` - Coordinate cache hit finding across cache groups

**Only called when**: `enable_prefix_caching=True` AND not skipping cache read

### 3. `single_type_kv_cache_manager.py`
**Functions involved**:
- `FullAttentionManager.find_longest_cache_hit()` - Find longest prefix match
- `cache_full_blocks()` - Cache computed blocks with hash

**Only called when**: `enable_prefix_caching=True`

### 4. `block_pool.py`
**Functions involved**:
- `get_cached_block()` - Lookup block by hash
- `cache_full_blocks()` - Insert blocks into cache map

**Only called when**: `enable_prefix_caching=True`

### 5. `kv_cache_utils.py`
**Functions involved**:
- Block hash computation
- Cache key generation

**Only called when**: `enable_prefix_caching=True`

## What Happens Differently

### WITHOUT Prefix Caching
```
Request → Scheduler → Allocate blocks → Forward pass → Output
```

### WITH Prefix Caching
```
Request → Scheduler 
  ↓
  → kv_cache_manager.get_computed_blocks()
      ↓
      → kv_cache_coordinator.find_longest_cache_hit()
          ↓
          → single_type_kv_cache_manager.find_longest_cache_hit()
              ↓
              → block_pool.get_cached_block()  [CACHE HIT/MISS]
  ↓
  → Allocate remaining blocks
  ↓
  → Forward pass (may use cached KV)
  ↓
  → kv_cache_manager.cache_blocks()
      ↓
      → single_type_kv_cache_manager.cache_full_blocks()
          ↓
          → block_pool.cache_full_blocks()  [STORE IN CACHE]
  ↓
  → Output
```

## Key Difference
The prefix caching path has **additional cache lookup and storage operations** that could introduce non-determinism if not handled carefully.
