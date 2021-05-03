from _ae_utils import *


class Manifold:
    def __init__(self, c=1., s=0):
        self.c = c  # 曲率绝对值
        self.s = s  # 0=euclidean, 1=hyperboloid, 2=poincare
        self.v_min = 1e-6  # 一般数或范数最小值
        self.v_max = 1e+6  # 一般数或范数最大值
        # norm_rate 最小为1. 若出现nan或loss震荡问题可尝试调大该值
        self.norm_rate1 = 1.1  # hyperboloid, 切空间最大范数缩小倍率. 若直接对双曲值softmax_coss操作产生0值无法优化, 尝试调高该值, 防止最大最小值差过大
        self.norm_rate2 = 1.1  # poincare, 切空间最大范数缩小倍率. 过度接近边界无穷远可能产生问题

    @staticmethod
    def s_to_tex(s):
        if s == 0:
            tex = '$\\mathbb{R}$'
        if s == 1:
            tex = '$\\mathbb{H}$'
        if s == 2:
            tex = '$\\mathbb{D}$'
        return tex

    def bound(self, x, min=None, max=None, use_norm=-1):  # 非负数范围约束
        if min is None:
            min = self.v_min
        if max is None:
            max = self.v_max
        min, max = min * 1., max * 1.
        if use_norm >= 0:  # 范数最大值约束
            x_norm2 = self.inner(x, s=use_norm)
            x_norm = (tf.abs(x_norm2) + self.v_min) ** 0.5  # 不能约束这里内积最大值
            max = tf.ones_like(x) * max  # tf1
            x = x * tf.where(x_norm > max, max / x_norm, tf.ones_like(x))  # tf1
        else:
            x = tf.abs(x) + min
            x = tf.clip_by_value(x, 0., max)  # 可能占用较大显存, 比如在4头2层HGAT4017节点LP任务中, 不去这行不能跑
        return x

    def inner(self, x, y=None, z=None, s=None, cross_join=False):
        """
        点x和y在双曲点z处的内积
        :param x:
        :param y:
        :param z:
        :param s:
        :param cross_join: bool; 是否用笛卡尔积的方式计算x和y的内积. 如果x和y维度不同则也使用这个方法(最后一维不管如何都是要相同的)
            如果s=2则z是比inner最后多一个向量维度,或者就一个向量.
            如果 x.shape=(5,2,4) x.shape=(5,3,4) 则 ret.shape=(5,2,3)
        :return:
        """
        if s is None:
            s = self.s
        if y is None:
            y = x
        if len(x.shape) == len(y.shape) > 1 and (cross_join or x.shape != y.shape):
            cross_join = True
            y_ = tf.transpose(y, perm=(lambda t: t[:-2] + t[-2:][::-1])(list(range(len(y.shape)))))
            inner = tf.matmul(x, y_)
        else:
            cross_join = False
            inner = tf.reduce_sum(x * y, axis=-1)
        if s == 0:
            ret = inner
        elif s == 1:
            if cross_join:
                ret = tf.matmul(tf.expand_dims(x[..., 0], -1), tf.expand_dims(y[..., 0], -2))
            else:
                ret = x[..., 0] * y[..., 0]
            ret = inner - 2. * ret
        elif s == 2:
            if z is None:
                z = tf.zeros_like(tf.expand_dims(inner, -1))
            ret = 1. - self.c * tf.reduce_sum(z ** 2, axis=-1)
            ret = self.bound(ret, max=1 - self.v_min)
            ret = inner * (2. / ret) ** 2.
        if len(x.shape) - len(ret.shape) == 1:
            ret = tf.expand_dims(ret, -1)
        return ret

    def norm(self, x, y=None, z=None, s=None, min=None, max=None):  # 点x和y在双曲点z处的范数
        inner = self.inner(x, y, z, s)
        if min is None or 0 < min ** 2 < self.v_min:
            min = self.v_min
        else:
            min **= 2
        if max is None:
            max = self.v_max ** 2
        else:
            max **= 2
        ret = self.bound(inner, min, max) ** 0.5
        return ret

    def sqdist(self, x, y, s=None, bound=True, cross_join=False):
        """
        双曲点x和双曲点y的距离平方
        :param x: Tensor; 空间中的点或点组成的张量, 最后一维是向量值
        :param y: Tensor; 空间中的点或点组成的张量, 最后一维是向量值
        :param s: None or int; 双曲空间类型, None则使用self.s
        :param bound: bool; 是否限制边界范围
        :param cross_join: bool; 是否用笛卡尔积的方式计算x和y的距离平方, 可用于计算出任意两向量之间距离
            如果 x.shape=(5,2,4) x.shape=(5,3,4) 则 ret.shape=(5,2,3)
        :return:
        """
        if s is None:
            s = self.s
        if s == 0:
            inner = lambda x, y=None, cross_join=False: self.inner(x, y, s=0, cross_join=cross_join)
            if cross_join:
                y_ = inner(y)
                y_ = tf.transpose(y_, perm=(lambda t: t[:-2] + t[-2:][::-1])(list(range(len(y_.shape)))))
                ret = inner(x) - 2. * inner(x, y, True) + y_
            else:
                ret = inner(x - y)
        elif s == 1:
            K = 1. / self.c
            x = tf.concat([(K + self.inner(x, s=0)) ** 0.5, x], -1)
            y = tf.concat([(K + self.inner(y, s=0)) ** 0.5, y], -1)
            ret = - self.c * self.inner(x, y, s=1, cross_join=cross_join)
            ret = self.bound(ret - 1, max=self.v_max ** 2) + 1
            ret = K ** 0.5 * tf.acosh(ret)
            ret = ret ** 2
        elif s == 2:
            if cross_join:
                y_ = (1. - self.c * self.inner(y, s=0))
                y_ = tf.transpose(y_, perm=(lambda t: t[:-2] + t[-2:][::-1])(list(range(len(y_.shape)))))
                ret = tf.matmul(1. - self.c * self.inner(x, s=0), y_)
                ret = self.bound(ret)
                ret = 1. / self.c ** 0.5 * tf.acosh(1 + 2 * self.c * self.sqdist(x, y, s=0, cross_join=True) / ret)
            else:
                inner = self.inner(self.mobius_add(-x, y, s=2), s=0)
                inner = self.bound(inner)  # 防止不可导点 √x, x=0
                ret = 2. / self.c ** 0.5 * tf.atanh(self.c ** 0.5 * inner ** 0.5)
            ret = ret ** 2
        if bound:
            ret = self.bound(ret)
        return ret

    def expmap(self, x, y=None, s=None):  # 双曲点y所在切空间上的点x映射到双曲空间
        if s is None:
            s = self.s
        if s == 0:
            ret = x + (0. if y is None else y)
        elif s == 1:
            if y is None:
                y = tf.concat([1. / self.c ** 0.5 * tf.ones_like(x[..., 0:1]), tf.zeros_like(x)], -1)
            else:
                y = tf.concat([(1 / self.c + self.inner(y, s=0)) ** 0.5, y], -1)
            x = tf.concat([self.inner(x, y[..., 1:], s=0) / y[..., 0:1], x], -1)
            x = self.bound(x, max=np.arccosh(self.v_max) / self.c ** 0.5 / self.norm_rate1, use_norm=1)
            u = self.norm(x, s=1) * self.c ** 0.5
            ret = tf.cosh(u) * y + tf.sinh(u) / u * x
            ret = ret[..., 1:]
        elif s == 2:
            if y is None:
                y = tf.zeros_like(x)
            x = self.bound(x, max=np.arctanh(1 - self.v_min) / self.c ** 0.5 / self.norm_rate2, use_norm=0)
            u = self.norm(x, s=0) * self.c ** 0.5
            y = self.bound(y, max=1 / self.c ** 0.5 - self.v_min, use_norm=0)
            ret = 1. / (1. - self.c * self.inner(y, s=0)) * u
            ret = self.mobius_add(y, tf.tanh(ret) / u * x)
        return ret

    def logmap(self, y, x=None, s=None):  # 双曲点y映射到双曲点x所在的切空间上
        if s is None:
            s = self.s
        if s == 0:
            ret = y - (0. if x is None else x)
        elif s == 1:
            K = 1. / self.c
            y = tf.concat([(K + self.inner(y, s=0)) ** 0.5, y], -1)
            if x is None:
                x = tf.concat([K ** 0.5 * tf.ones_like(y[..., 0:1]), tf.zeros_like(y[..., 1:])], -1)
            else:
                x = tf.concat([(K + self.inner(x, s=0)) ** 0.5, x], -1)
            u = y + self.c * self.inner(x, y, s=1) * x
            u_norm = self.norm(u, s=1)
            ret = self.sqdist(x[..., 1:], y[..., 1:], s=1) ** 0.5 / u_norm * u
            # 原来 self.norm_rate1 过低导致网络(尤其深层)双曲值最大最小值差过大, 进而softmax_coss操作会产生0值无法优化
            # 除了exp以外这里约束的目的就是能够使 self.norm_rate1 值更小, 减少返回欧式空间后的大小, 减轻优化问题
            ret = self.bound(ret, max=np.arccosh(self.v_max) / self.c ** 0.5 / self.norm_rate1, use_norm=1)
            ret = ret[..., 1:]
        elif s == 2:
            if x is None:
                x = tf.zeros_like(y)
            u = self.mobius_add(-x, y, s=2)
            u_norm = self.norm(u, s=0)
            x_norm2 = self.bound(self.inner(x, s=0), max=1. / self.c - self.v_min)
            ret = (1. - self.c * x_norm2) / self.c ** 0.5 * tf.atanh(self.c ** 0.5 * u_norm) / u_norm * u
        return ret

    def ptransp(self, z, x=None, y=None, s=None):  # 双曲点x切空间上的点z，在x切空间移动到双曲点y后的切空间点
        if s is None:
            s = self.s
        if s == 0:
            # ret = z + (0. if x is None else x) - (0. if y is None else y)
            ret = z  # 兼容欧式空间下的 黎曼优化
        elif s == 1:
            if x is None:
                x = tf.zeros_like(y)
            logyx = self.logmap(y, x, s=1)
            logxy = self.logmap(x, y, s=1)
            x0 = 1 / self.c + self.inner(x, s=0)
            x0 = self.bound(x0, max=self.v_max ** 2) ** 0.5
            z_ = tf.concat([self.inner(x, z, s=0) / x0, tf.ones_like(x) * z], -1)  # ones_like 防止z为单向量无批次
            logyx_ = tf.concat([self.inner(x, logyx, s=0) / x0, logyx], -1)
            ret = z - self.inner(logyx_, z_, s=1) / self.sqdist(x, y, s=1) * (logyx + logxy)
        elif s == 2:
            if x is None:
                x = tf.zeros_like(y)
            ret = self.logmap(self.mobius_add(y, self.expmap(z, x, s=2), s=2), y, s=2)
        return ret

    def mobius_add(self, x, y, s=None):  # 双曲点x和双曲点y的莫比乌斯加法
        if s is None:
            s = self.s
        if s == 0:
            ret = x + y
        elif s == 1:
            ret = self.expmap(self.ptransp(self.logmap(y, s=1), y=x, s=1), x, s=1)
        elif s == 2:
            x = self.bound(x, max=1 / self.c ** 0.5 - self.v_min, use_norm=0)
            y = self.bound(y, max=1 / self.c ** 0.5 - self.v_min, use_norm=0)
            a = 1. + 2. * self.c * self.inner(x, y, s=0) + self.c * self.inner(y, s=0)
            b = 1. - self.c * self.inner(x, s=0)
            ret = 1. + 2. * self.c * self.inner(x, y, s=0) + self.c ** 2. * self.inner(x, s=0) * self.inner(y, s=0)
            ret = self.bound(ret)  # 防止正负向量导致ret为0, 尤其计算距离中
            ret = (a * x + b * y) / ret
            ret = self.bound(ret, max=1 / self.c ** 0.5 - self.v_min, use_norm=0)
        return ret

    def mobius_matvec(self, x, M, s=None, exchange=False):  # 双曲点x和欧式矩阵M的莫比乌斯乘法
        if s is None:
            s = self.s
        if exchange:  # 矩阵乘法交换位置
            matmul = lambda x1, x2: tf.matmul(x2, x1)
        else:
            matmul = lambda x1, x2: tf.matmul(x1, x2)
        if s == 0:
            ret = matmul(x, M)
        elif s == 1:
            ret = self.expmap(matmul(self.logmap(x, s=1), M), s=1)
        elif s == 2:
            ret = self.expmap(matmul(self.logmap(x, s=2), M), s=2)
        return ret

    def to_other_manifold(self, x, om=None, tm=None):  # 从om流形到tm流形的转换, 半径不变
        if om is None:
            om = self.s
        if tm is None:
            tm = 0
        K = 1. / self.c
        if om == tm:
            return x
        elif om == 1 and tm == 2:  # hyperboloid to poincare
            r = K ** 0.5
            ret = r * x / (r + (K + self.inner(x, s=0)) ** 0.5)
        elif om == 2 and tm == 1:  # poincare to hyperboloid
            sqnorm = self.inner(x, s=0)
            ret = 2 * K * x / (K - sqnorm)
        elif tm == 0:
            ret = self.logmap(y=x, s=om)
        elif om == 0:
            ret = self.expmap(x, s=tm)
        return ret

    def egrad2rgrad(self, x, v, s=None):
        """
        将欧式梯度v转换为双曲点x切平面上的黎曼梯度
        :param x: Tensor; d维向量构成的张量, 用于表示d维双曲点
        :param v: Tensor; 同x, d维欧式点
        :param s: 黎曼空间类型
        :return:
        """
        if s is None:
            s = self.s
        if s == 0:
            ret = v
        elif s == 1:
            ret = v  # 没有时间维. 无需约束
        elif s == 2:
            ret = v * (1. - self.c * self.inner(x, s=0)) ** 2 / 4.
        return ret

    @staticmethod
    def test():
        c = 0.5
        K = 1. / c

        print('-' * 5 + 'range_limit')
        x = tf.Variable([[(np.float('-Nan')), -np.nan, np.inf, -np.inf, 1.]])
        print(x)
        a = Manifold().bound(x)
        print(a)

        print('-' * 5 + 'space：0')
        x = tf.Variable([[0.1, 0.2, 0.3, 0.4]])
        y = tf.Variable([[0.2, 0.11, 0.11, 0.11]])
        z = tf.Variable([[0.2, 0.25, 0.35, 0.15]])
        print(x, y, z)
        self = Manifold(c, 0)
        print('inner:\t', self.inner(x, y))
        print('sqdist-cross_join:\t',
              self.sqdist(tf.broadcast_to(x, [2, 4]), tf.broadcast_to(y, [3, 4]), cross_join=True))
        print('sqdist:\t', self.sqdist(x, y))
        print('log:\t', self.logmap(x, y))
        print('log0:\t', self.logmap(x))
        print('exp:\t', self.expmap(x))
        print('ptransp0:\t', self.ptransp(z, y=y))
        print('add:\t', self.mobius_add(x, y))
        print('matvec:\t', self.mobius_matvec(tf.concat([x, y], 0), tf.transpose(tf.concat([z, z], 0))))
        '''
        inner:	 tf.Tensor([[0.119]], shape=(1, 1), dtype=float32)
        sqdist:	 tf.Tensor([[0.13830101]], shape=(1, 1), dtype=float32)
        log:	 tf.Tensor([[-0.1         0.09        0.19000001  0.29000002]], shape=(1, 4), dtype=float32)
        log0:	 tf.Tensor([[0.1 0.2 0.3 0.4]], shape=(1, 4), dtype=float32)
        exp:	 tf.Tensor([[0.1 0.2 0.3 0.4]], shape=(1, 4), dtype=float32)
        ptransp0:	 <tf.Variable 'Variable:0' shape=(1, 4) dtype=float32, numpy=array([[0.2 , 0.25, 0.35, 0.15]], dtype=float32)>
        add:	 tf.Tensor([[0.3        0.31       0.41000003 0.51      ]], shape=(1, 4), dtype=float32)
        matvec:	 tf.Tensor(
        [[0.235  0.235 ]
         [0.1225 0.1225]], shape=(2, 2), dtype=float32)
        '''

        print('-' * 5 + 'space：1')
        x1 = tf.Variable([[(2 ** 2 + 3 ** 2 + 4 ** 2 + K) ** 0.5, 2, 3, 4.]])
        y1 = tf.Variable([[(21 ** 2 + 31 ** 2 + 41 ** 2 + K) ** 0.5, 21, 31, 41.]])
        z1 = tf.Variable([[(0.25 ** 2 + 0.35 ** 2 + 0.45 ** 2 + K) ** 0.5, 0.25, 0.35, -0.45]])
        x = x1[..., 1:]
        y = y1[..., 1:]
        z = z1[..., 1:]
        print(x, y, z)
        self = Manifold(c, 1)
        print('inner-cross_join:\t', self.inner(tf.broadcast_to(x1, [2, 4]), tf.broadcast_to(y1, [3, 4])))
        print('inner:\t', self.inner(x1, y1))
        print('sqdist-cross_join:\t',
              self.sqdist(tf.broadcast_to(x, [2, 3]), tf.broadcast_to(y, [3, 3]), cross_join=True))
        print('sqdist:\t', self.sqdist(x, y))
        print('log:\t', self.logmap(x, y))
        print('log0:\t', self.logmap(x))
        print('exp:\t', self.expmap(x))
        print('ptransp0:\t', self.ptransp(z, y=y))
        print('add:\t', self.mobius_add(x, y))
        print('matvec:\t', self.mobius_matvec(tf.concat([x, y], 0), tf.transpose(tf.concat([z], 0))))
        '''
        inner:	 tf.Tensor([[-10.249084]], shape=(1, 1), dtype=float32)
        sqdist:	 tf.Tensor([[10.741887]], shape=(1, 1), dtype=float32)
        log:	 tf.Tensor([[-48.702877 -71.872765 -95.04265 ]], shape=(1, 3), dtype=float32)
        log0:	 tf.Tensor([[1.0751545 1.6127319 2.150309 ]], shape=(1, 3), dtype=float32)
        exp:	 tf.Tensor([[11.826225 17.739338 23.65245 ]], shape=(1, 3), dtype=float32)
        ptransp0:	 tf.Tensor([[-0.36266768 -0.5544143  -1.6461608 ]], shape=(1, 3), dtype=float32)
        add:	 tf.Tensor([[160.1116  239.66739 319.2232 ]], shape=(1, 3), dtype=float32)
        matvec:	 tf.Tensor(
        [[-0.13459666]
         [-0.26266572]], shape=(2, 1), dtype=float32)
        '''

        print('-' * 5 + 'space：2')
        x = tf.Variable([[0.1, 0.2, 0.3, 0.4]])
        y = tf.Variable([[0.2, 0.11, 0.11, 0.11]])
        z = tf.Variable([[0.2, 0.25, 0.35, 0.15]])
        print(x, y, z)
        self = Manifold(c, 2)
        print('inner-cross_join:\t', self.inner(tf.broadcast_to(x, [2, 4]), tf.broadcast_to(y, [3, 4])))
        print('inner:\t', self.inner(x, y))
        print('inner-z-cross_join:\t', self.inner(tf.broadcast_to(x, [2, 4]), tf.broadcast_to(y, [3, 4]), z))
        print('inner-z:\t', self.inner(x, y, z))
        print('sqdist-cross_join:\t',
              self.sqdist(tf.broadcast_to(x, [2, 4]), tf.broadcast_to(y, [3, 4]), cross_join=True))
        print('sqdist:\t', self.sqdist(x, y))
        print('log:\t', self.logmap(x, y))
        print('log0:\t', self.logmap(x))
        print('exp:\t', self.expmap(x))
        print('ptransp0:\t', self.ptransp(z, y=y))
        print('add:\t', self.mobius_add(x, y))
        print('matvec:\t', self.mobius_matvec(tf.concat([x, y], 0), tf.transpose(tf.concat([z, z], 0))))
        '''
        inner-z:	 tf.Tensor([[0.6199417]], shape=(1, 1), dtype=float32)
        sqdist:	 tf.Tensor([[0.65837723]], shape=(1, 1), dtype=float32)
        log:	 tf.Tensor([[-0.12259186  0.08798665  0.19516747  0.3023483 ]], shape=(1, 4), dtype=float32)
        log0:	 tf.Tensor([[0.10550462 0.21100925 0.31651387 0.4220185 ]], shape=(1, 4), dtype=float32)
        exp:	 tf.Tensor([[0.09528284 0.19056568 0.2858485  0.38113135]], shape=(1, 4), dtype=float32)
        ptransp0:	 tf.Tensor([[0.19237003 0.2404625  0.3366475  0.14427751]], shape=(1, 4), dtype=float32)
        add:	 tf.Tensor([[0.25403154 0.28889793 0.39178106 0.4946642 ]], shape=(1, 4), dtype=float32)
        matvec:	 tf.Tensor(
        [[0.24297735 0.24297735]
         [0.12346134 0.12346134]], shape=(2, 2), dtype=float32)
        '''


def assign_to_manifold(var, manifold):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    setattr(var, "manifold", manifold)


def get_manifold(var, default_manifold=Manifold()):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if hasattr(var, "manifold"):
        return var.manifold
    else:
        return default_manifold


class Test:
    @staticmethod
    def manifold():
        x = tf.Variable([[0., 0.1], [0.4, 0.5]])
        y = tf.Variable([[0.1, 0], [0.5, 0.4]])

        for i in [1, 2]:
            print('-' * 10 + 'space：%d' % i)
            self = Manifold(s=i)
            xi = self.to_other_manifold(x, om=0, tm=i)
            yi = self.to_other_manifold(y, om=0, tm=i)
            xi1 = self.to_other_manifold(x, om=i, tm=0)
            yi1 = self.to_other_manifold(y, om=i, tm=0)
            xi2 = self.to_other_manifold(x, om=2, tm=1)
            yi2 = self.to_other_manifold(y, om=2, tm=1)
            # print('xi:\t', xi.numpy())
            # print('yi:\t', yi.numpy())
            # print('yi-:\t', self.to_other_manifold(y, om=2, tm=0).numpy())
            print('视觉距离:\t', self.sqdist(x, y, s=0).numpy()[:, 0] ** 0.5)
            print('作为切空间后的测地线距离:\t', self.sqdist(xi, yi).numpy()[:, 0] ** 0.5)
            print('作为双曲空间后的切空间距离:\t', self.sqdist(xi1, yi1, s=0).numpy()[:, 0] ** 0.5)
            print('作为poincare空间后的hyperboloid距离:\t', self.sqdist(xi2, yi2, s=1).numpy()[:, 0] ** 0.5)
            print('测地线距离:\t', self.sqdist(x, y).numpy()[:, 0] ** 0.5)

        print('-' * 10)
        print('x:', x)
        print('x EtoH:', self.to_other_manifold(x, om=0, tm=1).numpy())
        print('x EtoP:', self.to_other_manifold(x, om=0, tm=2).numpy())
        print('x EtoHtoP:', self.to_other_manifold(self.to_other_manifold(x, om=0, tm=1), om=1, tm=2).numpy())
        print('x PtoHtoE:', self.to_other_manifold(self.to_other_manifold(x, om=2, tm=1), om=1, tm=0).numpy())
        print('x PtoE:', self.to_other_manifold(x, om=2, tm=0).numpy())


if __name__ == '__main__':
    Manifold.test()
    Test.manifold()
