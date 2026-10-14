"""Bounded concurrent classification and provider-isolated answer caching."""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from .contracts import DecisionError, cache_key


class DecisionClassifier:
    def __init__(self, labels, cache, *, backend, model=None, workers=8, log=print):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer')
        self.labels, self.cache, self.backend = labels, cache, backend
        self.model, self.workers, self._log = model, workers, log

    def classify(self, scripts):
        results, todo = {}, []
        for timestamp, script in scripts.items():
            key = cache_key(self.model, self.labels, script, provider=self.backend.provider,
                            endpoint=self.backend.endpoint)
            hit = self.cache.get(key)
            if hit is not None:
                results[timestamp] = hit
            else:
                todo.append((timestamp, key, script))
        if todo:
            self._log(f'  {self.backend.provider}: {len(todo)} requests ({len(results)} cached)')
            asyncio.run(self._run(todo, results))
        return results

    async def _run(self, todo, results):
        semaphore = asyncio.Semaphore(self.workers)

        async def one(timestamp, key, script):
            async with semaphore:
                answer = await self.backend.predict(script, self.labels, self.model)
            return timestamp, key, replace(answer, provider=self.backend.provider)

        tasks = [asyncio.create_task(one(*item)) for item in todo]
        start, done = time.monotonic(), 0
        try:
            for task in asyncio.as_completed(tasks):
                timestamp, key, answer = await task
                self.cache.put(key, answer)
                results[timestamp] = answer
                done += 1
                if done % 250 == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - start, 1e-9)
                    self._log(f'  {self.backend.provider}: {done}/{len(todo)} done ({rate:.1f} requests/s)')
        except Exception as exc:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            detail = str(exc) if isinstance(exc, DecisionError) else type(exc).__name__
            raise DecisionError(f'{self.backend.provider} classification failed '
                                f'({done} answers saved to cache): {detail}') from None
        finally:
            await self.backend.aclose()
