"""Set-associative DRAM cache simulator with optional address-range bypass.

The cache accepts a stream of ``[timestamp, address]`` accesses and models a
DRAM cache backed by an SSD with fixed latency ``L``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class CacheLine:
    """A single cache line in a set."""

    tag: int
    last_used: int


class DRAMCacheSimulator:
    """Configurable N-set, M-way DRAM cache with LRU replacement.

    Args:
        num_sets: Number of sets in the cache.
        num_ways: Associativity (ways per set).
        ssd_latency: Latency ``L`` of the SSD lower level.
        block_size: Cache block size in bytes.
        dram_latency: Hit latency of DRAM cache in the same time unit as input.
        bypass_enabled: Global bypass signal.
        bypass_max_address: Address threshold ``A``. When bypass is enabled,
            only addresses in ``[0, A]`` are allowed to access DRAM cache.
            Addresses outside this range bypass DRAM and go directly to SSD.
    """

    def __init__(
        self,
        num_sets: int,
        num_ways: int,
        ssd_latency: int,
        block_size: int = 64,
        dram_latency: int = 1,
        bypass_enabled: bool = False,
        bypass_max_address: Optional[int] = None,
    ) -> None:
        if num_sets <= 0:
            raise ValueError("num_sets must be > 0")
        if num_ways <= 0:
            raise ValueError("num_ways must be > 0")
        if ssd_latency < 0:
            raise ValueError("ssd_latency must be >= 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if dram_latency < 0:
            raise ValueError("dram_latency must be >= 0")
        if bypass_max_address is not None and bypass_max_address < 0:
            raise ValueError("bypass_max_address must be >= 0 when provided")
        if bypass_enabled and bypass_max_address is None:
            raise ValueError("bypass_max_address must be set when bypass_enabled=True")

        self.num_sets = num_sets
        self.num_ways = num_ways
        self.ssd_latency = ssd_latency
        self.block_size = block_size
        self.dram_latency = dram_latency
        self.bypass_enabled = bypass_enabled
        self.bypass_max_address = bypass_max_address

        self._sets: List[List[CacheLine]] = [[] for _ in range(self.num_sets)]
        self._access_counter = 0

        self.hits = 0
        self.misses = 0
        self.bypasses = 0

    def _decode(self, address: int) -> Tuple[int, int]:
        if address < 0:
            raise ValueError("address must be >= 0")
        block_addr = address // self.block_size
        set_idx = block_addr % self.num_sets
        tag = block_addr // self.num_sets
        return set_idx, tag

    def _touch_or_insert(self, set_idx: int, tag: int) -> bool:
        """Returns True on hit, False on miss (with insertion + eviction)."""
        lines = self._sets[set_idx]
        self._access_counter += 1

        for line in lines:
            if line.tag == tag:
                line.last_used = self._access_counter
                return True

        if len(lines) >= self.num_ways:
            lru_idx = min(range(len(lines)), key=lambda i: lines[i].last_used)
            lines.pop(lru_idx)

        lines.append(CacheLine(tag=tag, last_used=self._access_counter))
        return False

    def _can_use_dram_cache(self, address: int) -> bool:
        """Whether the access is permitted to use DRAM cache under bypass rules."""
        if not self.bypass_enabled:
            return True
        assert self.bypass_max_address is not None
        return 0 <= address <= self.bypass_max_address

    def access(self, timestamp: int, address: int) -> dict:
        """Process one ``[timestamp, address]`` access and return result metadata."""
        if timestamp < 0:
            raise ValueError("timestamp must be >= 0")
        if address < 0:
            raise ValueError("address must be >= 0")

        bypassed = not self._can_use_dram_cache(address)

        if bypassed:
            self.bypasses += 1
            latency = self.ssd_latency
            return {
                "timestamp": timestamp,
                "address": address,
                "set": None,
                "tag": None,
                "hit": False,
                "bypassed": True,
                "served_by": "SSD",
                "latency": latency,
                "complete_time": timestamp + latency,
            }

        set_idx, tag = self._decode(address)
        is_hit = self._touch_or_insert(set_idx, tag)

        if is_hit:
            self.hits += 1
            latency = self.dram_latency
            level = "DRAM"
        else:
            self.misses += 1
            latency = self.ssd_latency + self.dram_latency
            level = "SSD"

        return {
            "timestamp": timestamp,
            "address": address,
            "set": set_idx,
            "tag": tag,
            "hit": is_hit,
            "bypassed": False,
            "served_by": level,
            "latency": latency,
            "complete_time": timestamp + latency,
        }

    def process_stream(self, stream: Iterable[Sequence[int]]) -> List[dict]:
        """Process an access stream of ``[timestamp, address]`` items."""
        results: List[dict] = []
        for item in stream:
            if len(item) != 2:
                raise ValueError(f"each stream item must be [timestamp, address], got: {item}")
            timestamp, address = int(item[0]), int(item[1])
            results.append(self.access(timestamp, address))
        return results

    def stats(self) -> dict:
        cache_total = self.hits + self.misses
        hit_rate = (self.hits / cache_total) if cache_total else 0.0
        return {
            "total": cache_total + self.bypasses,
            "cache_accesses": cache_total,
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "hit_rate": hit_rate,
            "num_sets": self.num_sets,
            "num_ways": self.num_ways,
            "ssd_latency": self.ssd_latency,
            "block_size": self.block_size,
            "dram_latency": self.dram_latency,
            "bypass_enabled": self.bypass_enabled,
            "bypass_max_address": self.bypass_max_address,
        }


if __name__ == "__main__":
    # Example: stream entries are [timestamp, address]
    sample_stream = [
        [0, 0x1000],
        [5, 0x1040],
        [8, 0x1000],
        [20, 0x2000],
        [30, 0x3000],
        [35, 0x1000],
    ]

    sim = DRAMCacheSimulator(
        num_sets=4,
        num_ways=2,
        ssd_latency=100,
        block_size=64,
        bypass_enabled=True,
        bypass_max_address=0x1FFF,
    )
    trace = sim.process_stream(sample_stream)

    for event in trace:
        print(event)
    print("stats:", sim.stats())
