import numpy as np

class StepwiseLog:
    def __init__(self, diffusion_steps, max_keep=None):
        self.diffusion_steps = diffusion_steps
        self.max_keep = max_keep
        self.reset()

    def reset(self):
        self.metric_per_t = {t: [] for t in range(1, self.diffusion_steps)}
        self.avg_per_step = np.zeros(self.diffusion_steps)
        self.avg_sq_per_step = np.zeros(self.diffusion_steps)
        self.n_per_step = np.zeros(self.diffusion_steps)

    def update(self, t, metric):
        self.metric_per_t[t].append(metric)
        if self.max_keep is not None and len(self.metric_per_t) > self.max_keep:
            self.metric_per_t[t] = self.metric_per_t[t][self.max_keep:]

        self.avg_per_step[t - 1] = np.mean(self.metric_per_t[t])
        self.avg_sq_per_step[t - 1] = np.sqrt(np.mean(np.power(self.metric_per_t[t], 2)))
        self.n_per_step[t - 1] += 1

    def update_multiple(self, ts, metrics):
        for t, m in zip(ts, metrics):
            self.update(t, m)

    def get_avg_in_range(self, t0, t1):
        # TODO: averaging within steps??
        return np.concatenate([
            self.metric_per_t[t] for t in range(t0, t1)
        ]).mean()

    def __getitem__(self, t):
        return self.metric_per_t[t]

