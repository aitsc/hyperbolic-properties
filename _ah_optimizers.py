"""
Modified from
Oleg Smirnov. "TensorFlow ManOpt: a library for manifold-constrained optimization in TensorFlow"
https://github.com/master/tensorflow-manopt
"""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.optimizers import Adam, SGD

from _ai_manifold import *


@generic_utils.register_keras_serializable(name="RiemannianAdam")
class RiemannianAdam(optimizer_v2.OptimizerV2):
    """
    Optimizer that implements the Riemannian Adam algorithm.

    Becigneul, Gary, and Octavian-Eugen Ganea. "Riemannian Adaptive Optimization
    Methods." International Conference on Learning Representations. 2018.
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            name="RiemannianAdam",
            **kwargs,
    ):
        """Construct a new Riemannian Adam optimizer.

        Becigneul, Gary, and Octavian-Eugen Ganea. "Riemannian Adaptive
        Optimization Methods." International Conference on Learning
        Representations. 2018.

        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
            takes no arguments and returns the actual value to use, The learning
            rate. Defaults to 0.001.
          beta_1: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use. The exponential decay
            rate for the 1st moment estimates. Defaults to 0.9.
          beta_2: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use, The exponential decay
            rate for the 2nd moment estimates. Defaults to 0.999.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
          amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond". Defaults to `False`.
          name: Optional name for the operations created when applying gradients.
            Defaults to "RiemannianAdam".
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        """

        super(RiemannianAdam, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(RiemannianAdam, self)._prepare_local(
            var_device, var_dtype, apply_state
        )

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]["lr_t"] * (
                math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(RiemannianAdam, self).set_weights(weights)

    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)

        alpha = (
                coefficients["lr_t"]
                * math_ops.sqrt(1 - coefficients["beta_2_power"])
                / (1 - coefficients["beta_1_power"])
        )
        m.assign_add((grad - m) * (1 - coefficients["beta_1_t"]))
        g2 = grad ** 2 if manifold.s == 0 else manifold.inner(z=var, x=grad, y=grad)  # 欧式是直接平方
        v.assign_add(
            (g2 - v)
            * (1 - coefficients["beta_2_t"])
        )

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat.assign(math_ops.maximum(vhat, v))
            v = vhat
        var_t = manifold.expmap(
            y=var, x=-(m * alpha) / (math_ops.sqrt(v) + coefficients["epsilon"])
        )
        m.assign(manifold.ptransp(x=var, y=var_t, z=m))
        var.assign(var_t)

    @def_function.function(experimental_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t_values = (
                array_ops.gather(m, indices) * coefficients["beta_1_t"]
                + m_scaled_g_values
        )

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        g2 = grad ** 2 if manifold.s == 0 else manifold.inner(z=var, x=grad, y=grad)  # 欧式是直接平方
        v_scaled_g_values = (
                g2
                * coefficients["one_minus_beta_2_t"]
        )
        v_t_values = (
                array_ops.gather(v, indices) * coefficients["beta_2_t"]
                + v_scaled_g_values
        )

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat.scatter_max(ops.IndexedSlices(v_t_values, indices))
            v_t_values = array_ops.gather(vhat, indices)

        var_values = array_ops.gather(var, indices)
        var_t_values = manifold.expmap(
            y=var_values,
            x=-(m_t_values * coefficients["lr"])
              / (math_ops.sqrt(v_t_values) + coefficients["epsilon"]),
        )
        m_t_transp = manifold.ptransp(x=var_values, y=var_t_values, z=m_t_values)

        m.scatter_update(ops.IndexedSlices(m_t_transp, indices))
        v.scatter_update(ops.IndexedSlices(v_t_values, indices))
        var.scatter_update(ops.IndexedSlices(var_t_values, indices))

    def get_config(self):
        config = super(RiemannianAdam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


@generic_utils.register_keras_serializable(name="RiemannianSGD")
class RiemannianSGD(optimizer_v2.OptimizerV2):
    """
    Optimizer that implements the Riemannian SGD algorithm.

    Bonnabel, Silvere. "Stochastic gradient descent on Riemannian manifolds."
    IEEE Transactions on Automatic Control 58.9 (2013): 2217-2229.
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.0,
            nesterov=False,
            name="RiemannianSGD",
            **kwargs,
    ):
        """Construct a new Riemannian SGD optimizer.

        Bonnabel, Silvere. "Stochastic gradient descent on Riemannian
        manifolds." IEEE Transactions on Automatic Control 58.9 (2013):
        2217-2229.

        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
            takes no arguments and returns the actual value to use, The learning
            rate. Defaults to 0.001.
          momentum: A float hyperparameter >= 0 that accelerates gradient descent
            in the relevant direction and dampens oscillations. Defaults to 0, i.e.,
            vanilla gradient descent.
          nesterov: boolean. Whether to apply Nesterov momentum. Defaults to `False`.
          name: Optional name for the operations created when applying gradients.
            Defaults to "RiemannianSGD".
          **kwargs: Keyword arguments. Allowed to be one of `"clipnorm"` or
            `"clipvalue"`. `"clipnorm"` (float) clips gradients by norm; `"clipvalue"`
            (float) clips gradients by value.

        """

        super(RiemannianSGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._momentum = False
        if (
                isinstance(momentum, ops.Tensor)
                or callable(momentum)
                or momentum > 0
        ):
            self._momentum = True
        if isinstance(momentum, (int, float)) and (
                momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)
        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(RiemannianSGD, self)._prepare_local(
            var_device, var_dtype, apply_state
        )
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype)
        )

    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)

        if self._momentum:
            momentum = self.get_slot(var, "momentum")
            momentum_t = momentum * self._momentum - grad * coefficients["lr_t"]
            if self.nesterov:
                var_t = manifold.expmap(
                    y=var,
                    x=momentum_t * self._momentum - grad * coefficients["lr_t"],
                )
            else:
                var_t = manifold.expmap(y=var, x=momentum_t)
            momentum.assign(manifold.ptransp(x=var, y=var_t, z=momentum_t))
            var.assign(var_t)
        else:
            var.assign(manifold.expmap(y=var, x=-grad * coefficients["lr_t"]))

    @def_function.function(experimental_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)

        var_values = array_ops.gather(var, indices)

        if self._momentum:
            momentum = self.get_slot(var, "momentum")
            momentum_t_values = (
                    array_ops.gather(momentum, indices) * self._momentum
                    - grad * coefficients["lr_t"]
            )
            if self.nesterov:
                var_t_values = manifold.expmap(
                    y=var_values,
                    x=momentum_t_values * self._momentum
                      - grad * coefficients["lr_t"],
                )
            else:
                var_t_values = manifold.expmap(y=var_values, x=momentum_t_values)
            momentum_transp_values = manifold.ptransp(
                x=var_values, y=var_t_values, z=momentum_t_values
            )
            momentum.scatter_update(
                ops.IndexedSlices(momentum_transp_values, indices)
            )
        else:
            var_t_values = manifold.expmap(
                y=var_values, x=-grad * coefficients["lr_t"]
            )

        var.scatter_update(ops.IndexedSlices(var_t_values, indices))

    def get_config(self):
        config = super(RiemannianSGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config


class Test:
    @staticmethod
    def radam():
        """
        最后一次结果:
        dtype: <class 'numpy.float64'> amsgrad: True
        0 var0_ref: tf.Tensor([0.999 1.999], shape=(2,), dtype=float64) var0: tf.Tensor([0.999 1.999], shape=(2,), dtype=float64)
        0 var1_ref: tf.Tensor([2.99900003 3.99900003], shape=(2,), dtype=float64) var1: tf.Tensor([2.99900447 3.99900447], shape=(2,), dtype=float64)
        1 var0_ref: tf.Tensor([0.99800001 1.99800001], shape=(2,), dtype=float64) var0: tf.Tensor([0.99800225 1.99800225], shape=(2,), dtype=float64)
        1 var1_ref: tf.Tensor([2.99800005 3.99800005], shape=(2,), dtype=float64) var1: tf.Tensor([2.99805586 3.99805586], shape=(2,), dtype=float64)
        2 var0_ref: tf.Tensor([0.99700001 1.99700001], shape=(2,), dtype=float64) var0: tf.Tensor([0.99703487 1.99703487], shape=(2,), dtype=float64)
        2 var1_ref: tf.Tensor([2.99700007 3.99700007], shape=(2,), dtype=float64) var1: tf.Tensor([2.99714948 3.99714948], shape=(2,), dtype=float64)
        :return:
        """
        print('测试:', sys._getframe().f_code.co_name)
        for i, dtype in enumerate([np.half, np.float32, np.float64]):
            for amsgrad in [False, True]:
                print('dtype:', dtype, 'amsgrad:', amsgrad)

                var0_np = np.array([1.0, 2.0], dtype=dtype)
                grads0_np = np.array([0.1, 0.1], dtype=dtype)
                var1_np = np.array([3.0, 4.0], dtype=dtype)
                grads1_np = np.array([0.01, 0.01], dtype=dtype)

                var0 = tf.Variable(var0_np, name="var0_%d" % i)
                var1 = tf.Variable(var1_np, name="var1_%d" % i)
                var0_ref = tf.Variable(var0_np, name="var0_ref_%d" % i)
                var1_ref = tf.Variable(var1_np, name="var1_ref_%d" % i)
                grads0 = tf.constant(grads0_np)
                grads1 = tf.constant(grads1_np)

                learning_rate = 0.001
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8

                opt = RiemannianAdam(
                    learning_rate=learning_rate,
                    beta_1=beta1,
                    beta_2=beta2,
                    epsilon=epsilon,
                    amsgrad=amsgrad,
                )
                opt_ref = Adam(
                    learning_rate=learning_rate,
                    beta_1=beta1,
                    beta_2=beta2,
                    epsilon=epsilon,
                    amsgrad=amsgrad,
                )

                # Run 3 steps
                for t in range(3):
                    opt.apply_gradients(
                        zip([grads0, grads1], [var0, var1])
                    )
                    opt_ref.apply_gradients(
                        zip([grads0, grads1], [var0_ref, var1_ref])
                    )

                    print(t, 'var0_ref:', var0_ref.value(), 'var0:', var0.value())
                    print(t, 'var1_ref:', var1_ref.value(), 'var1:', var1.value())
                print()

    @staticmethod
    def rsgd():
        """
        最后一次结果:
        dtype: <class 'numpy.float64'> momentum: 0.9 nesterov: True
        0 var0_ref: tf.Tensor([0.99981 1.99981], shape=(2,), dtype=float64) var0: tf.Tensor([0.9998 1.9998], shape=(2,), dtype=float64)
        0 var1_ref: tf.Tensor([2.999981 3.999981], shape=(2,), dtype=float64) var1: tf.Tensor([2.99988 3.99988], shape=(2,), dtype=float64)
        1 var0_ref: tf.Tensor([0.999539 1.999539], shape=(2,), dtype=float64) var0: tf.Tensor([0.99949 1.99949], shape=(2,), dtype=float64)
        1 var1_ref: tf.Tensor([2.9999539 3.9999539], shape=(2,), dtype=float64) var1: tf.Tensor([2.99965 3.99965], shape=(2,), dtype=float64)
        2 var0_ref: tf.Tensor([0.9991951 1.9991951], shape=(2,), dtype=float64) var0: tf.Tensor([0.99907 1.99907], shape=(2,), dtype=float64)
        2 var1_ref: tf.Tensor([2.99991951 3.99991951], shape=(2,), dtype=float64) var1: tf.Tensor([2.99931 3.99931], shape=(2,), dtype=float64)
        :return:
        """
        print('测试:', sys._getframe().f_code.co_name)
        for i, dtype in enumerate([np.half, np.float32, np.float64]):
            for momentum in [0.0, 0.9]:  # SGD使用momentum似乎和原始tf版本不一致
                for nesterov in [False, True]:
                    print('dtype:', dtype, 'momentum:', momentum, 'nesterov:', nesterov)

                    var0_np = np.array([1.0, 2.0], dtype=dtype)
                    grads0_np = np.array([0.1, 0.1], dtype=dtype)
                    var1_np = np.array([3.0, 4.0], dtype=dtype)
                    grads1_np = np.array([0.01, 0.01], dtype=dtype)

                    var0 = tf.Variable(var0_np, name="var0_%d" % i)
                    var1 = tf.Variable(var1_np, name="var1_%d" % i)
                    var0_ref = tf.Variable(var0_np, name="var0_ref_%d" % i)
                    var1_ref = tf.Variable(var1_np, name="var1_ref_%d" % i)
                    grads0 = tf.constant(grads0_np)
                    grads1 = tf.constant(grads1_np)

                    learning_rate = 0.001
                    opt = RiemannianSGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                    )
                    opt_ref = SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                    )

                    # Run 3 steps
                    for t in range(3):
                        opt.apply_gradients(
                            zip([grads0, grads1], [var0, var1])
                        )
                        opt_ref.apply_gradients(
                            zip([grads0, grads1], [var0_ref, var1_ref])
                        )

                        print(t, 'var0_ref:', var0_ref.value(), 'var0:', var0.value())
                        print(t, 'var1_ref:', var1_ref.value(), 'var1:', var1.value())
                    print()


if __name__ == '__main__':
    Test.radam()
    Test.rsgd()
