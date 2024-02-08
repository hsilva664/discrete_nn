from losses.base_loss import BaseLoss
import torch


class DeceivingSquaredLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        # Combines two second order polynomials to induce following the direction of the gradient to lead to the
        # wrong solution. The parameter d divides the [0,1] interval in 2 parts. If the model probability is initialized
        # on the part closer to the wrong solution, following the gradient will converge to this wrong solution
        # Here we set the values for the minimization problem
        self.v0 = 0.0  # Value of z=0
        self.v1 = 1.0  # Value of z=1
        # Value of the stationary point z=d
        self.vd = 2.0
        # Position of the stationary point z=d, in the (0,1) interval
        self.d = 0.4
        assert 0 < self.d < 1

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            p1 = self.vd - ((self.vd - self.v0)/(self.d**2))*((sample-self.d)**2)
            p2 = self.vd - ((self.vd - self.v1) / ((1-self.d) ** 2)) * ((sample - self.d) ** 2)
            o = torch.where(sample <= self.d, p1, p2)
            return o.mean(1)

class DeceivingPiecewiseLinearLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        # This function behaves differently near the corners and inside the line (for 1d) making the model often choose
        # a wrong solution (choosing the right one depends on the learning rate)
        xf1 = 0.125
        yf1 = -1.25
        xf2 = 0.13

        # Infered from above
        m = (yf1 - (-1))/xf1
        yf2 = m * xf2 - 0.5 * m
        M = (yf2 - yf1) / (xf2 - xf1)
        C1 = lambda x: m * x - 1.
        C2 = lambda x: m * x - m/2.
        C3 = lambda x: m * x - m + 1.
        D1 = lambda x: M * x - M * xf2 + yf2
        D2 = lambda x: M * x - (M - m) * (1 - xf2) - m/2.

        self.f = lambda x: torch.where(x < xf1, C1(x),
                           torch.where(x < xf2, D1(x),
                           torch.where(x < 1-xf2, C2(x),
                           torch.where(x < 1-xf1, D2(x),
                           C3(x)))))

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            o = self.f(sample)
            return o.mean(1)
