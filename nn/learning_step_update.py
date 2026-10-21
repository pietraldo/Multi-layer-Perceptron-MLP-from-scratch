from abc import ABC, abstractmethod

# update(learning_step, epoch, loss) is called after every epoch and returns the learning step for the next one
# epoch is the number of epochs trained so far (counting from 1), loss is the summed loss of the finished epoch
# params() returns the constructor arguments, it is what gets saved in the model file,
# so a strategy with state (like ReduceOnPlateau) has to take that state in its constructor too

class LearningStepUpdate(ABC):
    @abstractmethod
    def update(self, learning_step, epoch, loss):
        pass
    def params(self):
        return {}

class Constant(LearningStepUpdate):
    def update(self, learning_step, epoch, loss):
        return learning_step

class StepDecay(LearningStepUpdate):
    # multiplies the step by factor every `every` epochs
    def __init__(self, factor=0.5, every=10, min_step=0.0):
        self.factor = factor
        self.every = every
        self.min_step = min_step
    def update(self, learning_step, epoch, loss):
        if epoch % self.every == 0:
            learning_step *= self.factor
        return max(learning_step, self.min_step)
    def params(self):
        return {"factor": self.factor, "every": self.every, "min_step": self.min_step}

class ExponentialDecay(LearningStepUpdate):
    # multiplies the step by rate after every epoch, step after n epochs is step0 * rate^n
    def __init__(self, rate=0.95, min_step=0.0):
        self.rate = rate
        self.min_step = min_step
    def update(self, learning_step, epoch, loss):
        return max(learning_step * self.rate, self.min_step)
    def params(self):
        return {"rate": self.rate, "min_step": self.min_step}

class InverseTimeDecay(LearningStepUpdate):
    # step after n epochs is step0 / (1 + decay * n), written as a ratio so it only needs the current step
    def __init__(self, decay=0.1):
        self.decay = decay
    def update(self, learning_step, epoch, loss):
        return learning_step * (1 + self.decay * (epoch - 1)) / (1 + self.decay * epoch)
    def params(self):
        return {"decay": self.decay}

class ReduceOnPlateau(LearningStepUpdate):
    # multiplies the step by factor when the loss has not improved for `patience` epochs
    # best and wait are training state, saved with the model so training can be continued after loading
    def __init__(self, factor=0.5, patience=5, min_step=0.0, threshold=1e-4, best=None, wait=0):
        self.factor = factor
        self.patience = patience
        self.min_step = min_step
        self.threshold = threshold
        self.best = best
        self.wait = wait
    def update(self, learning_step, epoch, loss):
        # improvement is relative, so it does not depend on the scale of the loss
        if self.best is None or loss < self.best * (1 - self.threshold):
            self.best = float(loss)
            self.wait = 0
            return learning_step
        self.wait += 1
        if self.wait >= self.patience:
            self.wait = 0
            learning_step = max(learning_step * self.factor, self.min_step)
        return learning_step
    def params(self):
        return {"factor": self.factor, "patience": self.patience, "min_step": self.min_step,
                "threshold": self.threshold, "best": self.best, "wait": self.wait}
