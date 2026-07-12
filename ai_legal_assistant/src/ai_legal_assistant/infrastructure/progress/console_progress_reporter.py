from __future__ import annotations


class ConsoleProgressReporter:
    def report(self, message: str) -> None:
        print(message, flush=True)
