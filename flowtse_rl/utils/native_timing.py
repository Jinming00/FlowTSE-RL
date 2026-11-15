"""Low-frequency CPU wall-time accounting for an explicit training loop."""
import math
import time
from datetime import datetime, timedelta, timezone


class NativeTiming:
    def __init__(self, previous=None, *, known=True):
        previous = previous or {}
        self.history = float(previous.get('elapsed_seconds', 0))
        self.origin_known = previous.get('origin_known', known)
        self.started = self.cursor = time.perf_counter()
        self.stats = dict(train_seconds=0., train_batches=0, validation_seconds=0.,
                          validation_passes=0, scoring_seconds=0., overhead_seconds=0.)
        # Backend migration and restart can change compile/cache costs. Remeasure
        # rates in each process, but retain the previously recorded elapsed time.
        self.seen = self.pending = 0
        self.last_report = None

    def elapsed(self):
        return self.history + time.perf_counter() - self.started

    def batch_end(self):
        self.seen += 1
        self.pending += 1
        if self.seen == 20 or self.pending >= 20:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        now = time.perf_counter()
        if self.seen > 20:
            self.stats['train_seconds'] += now - self.cursor
            self.stats['train_batches'] += self.pending
        self.pending = 0
        self.cursor = now

    def validation_end(self, seconds, scoring):
        self.stats['validation_seconds'] += seconds
        self.stats['validation_passes'] += 1
        self.stats['scoring_seconds'] += scoring
        self.cursor = time.perf_counter()

    def overhead_end(self, started):
        self.stats['overhead_seconds'] += time.perf_counter() - started
        self.cursor = time.perf_counter()

    def state_dict(self):
        self.flush()
        return dict(elapsed_seconds=self.elapsed(), origin_known=self.origin_known, stats=dict(self.stats))

    def report(self, config, epoch, batch, batches, step, *, force=False):
        now = time.perf_counter()
        if not force and self.last_report is not None and now - self.last_report < 60:
            return {}
        self.last_report = now
        remaining_batches = max(0, (config['train']['num_epochs'] - epoch) * batches - batch)
        cap = config['train']['max_steps']
        if cap > 0:
            remaining_batches = min(remaining_batches, max(0, cap - step) * config['train']['accumulation_steps'])
        elapsed = self.elapsed()
        metrics = {'timing/elapsed_seconds': elapsed}
        measured = self.stats['train_batches']
        if measured:
            rate = self.stats['train_seconds'] / measured
            remaining = rate * remaining_batches
            if self.seen:
                remaining += self.stats['overhead_seconds'] / self.seen * remaining_batches
            passes = self.stats['validation_passes']
            if passes:
                # Estimate only scheduled validations within the remaining work.
                remaining_epochs = math.ceil(remaining_batches / max(1, batches))
                every = config['validation'].get('every_n_epochs', 1)
                validations = sum((epoch + i + 1) % every == 0 for i in range(remaining_epochs))
                remaining += validations * self.stats['validation_seconds'] / passes
            total = elapsed + remaining
            finish = time.time() + remaining
            metrics.update({'timing/remaining_seconds': remaining,
                            'timing/estimated_finish_unix': finish, 'timing/train_seconds_per_microbatch': rate})
            if self.origin_known:
                metrics['timing/estimated_total_seconds'] = total
            label = 'Training ETA' if passes else 'Training-only ETA (validation/scoring not measured yet)'
            duration = lambda value: str(timedelta(seconds=int(value)))
            stamp = datetime.fromtimestamp(finish, timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S SGT')
            history = '' if self.origin_known else ' | earlier elapsed unknown'
            total_text = duration(total) if self.origin_known else 'unknown (earlier elapsed unavailable)'
            print(f'{label} | epoch {min(epoch+1,config["train"]["num_epochs"])}/{config["train"]["num_epochs"]}, step {step} | '
                  f'active elapsed {duration(elapsed)} | estimated total {total_text} | '
                  f'remaining {duration(remaining)} | finish around {stamp}{history}', flush=True)
        return metrics
