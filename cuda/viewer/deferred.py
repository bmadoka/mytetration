"""Immutable map archival. Submission is a post-present operation, never a draw dependency."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time

import numpy as np


class MapArchive:
    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-archive")
        self.entries = {}
        self.frames = []

    def register(self, key, m, record):
        # Producers return fresh arrays and relinquish write access permanently.
        m.setflags(write=False)
        self.entries[key] = {"map": m, "record": record, "future": None}
        record["sha256"] = None

    @staticmethod
    def _pack(m, record):
        assert not m.flags.writeable
        thread = threading.current_thread()
        record["sha_thread"] = {"name": thread.name, "ident": thread.ident}
        record["sha_t_start"] = time.perf_counter()
        record["sha256"] = hashlib.sha256(m.tobytes()).hexdigest()
        record["sha_t_end"] = time.perf_counter()
        record["pack_thread"] = dict(record["sha_thread"])
        record["pack_t_start"] = time.perf_counter()
        bits = np.packbits(m, bitorder="big")
        record["pack_t_end"] = time.perf_counter()
        return bits

    def shown(self, key, t_present):
        entry = self.entries[key]
        entry["record"].setdefault("first_shown_t", t_present)
        if entry["future"] is None:
            entry["record"]["archive_t_enqueue"] = time.perf_counter()
            entry["future"] = self.worker.submit(self._pack, entry.pop("map"), entry["record"])

    def bind(self, frame, key):
        self.frames.append((frame, key))

    def finish(self, last_present):
        for key, entry in self.entries.items():
            if entry["future"] is None:
                self.shown(key, last_present)
                entry["record"].pop("first_shown_t", None)
        for entry in self.entries.values():
            entry["bits"] = entry["future"].result()  # propagate every worker failure
        for frame, key in self.frames:
            record = self.entries[key]["record"]
            if "map_sha256" in frame:
                frame["map_sha256"] = record["sha256"]
            if frame["kind"] != "fast":
                for name in ("sha_t_start", "sha_t_end", "sha_thread", "pack_t_start", "pack_t_end", "pack_thread"):
                    frame[name] = record[name]

    def bits(self, key):
        return self.entries[key]["bits"]

    def close(self, last_present):
        try:
            self.finish(last_present)
        finally:
            self.worker.shutdown(wait=True)
