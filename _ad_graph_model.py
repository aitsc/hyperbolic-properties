from tensorflow.keras import layers, callbacks, activations, initializers, constraints, regularizers
import tensorflow.keras.backend as K

from _ac_data_helper import *
from _ah_optimizers import *
from _ai_manifold import *


class GraphConvolution(layers.Layer):
    def __init__(self, units, support=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 manifold=Manifold(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.support = support
        self.m = manifold
        assert support >= 1.0

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]
        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            # assign_to_manifold(self.bias, self.m)  # 对b黎曼优化
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):  # 输入欧式, 输出流形
        features = inputs[0]  # 特征向量
        basis = inputs[1:]  # list, 标准化邻接矩阵
        supports = list()
        for i in range(self.support):
            # A * X
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        # AX * W
        output = K.dot(supports, self.kernel)
        output = self.m.expmap(output)
        # + b
        if tf.is_tensor(self.bias):
            # output += self.bias
            b = self.bias
            b = self.m.expmap(b)  # 对b黎曼优化应去除此行
            output = self.m.mobius_add(output, b)
        return output

    def get_config(self):
        base_config = super(GraphConvolution, self).get_config()
        base_config['units'] = self.units
        base_config['support'] = self.support
        base_config['use_bias'] = self.use_bias
        base_config['kernel_initializer'] = self.kernel_initializer
        base_config['bias_initializer'] = self.bias_initializer
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['bias_regularizer'] = self.bias_regularizer
        base_config['kernel_constraint'] = self.kernel_constraint
        base_config['bias_constraint'] = self.bias_constraint
        base_config['manifold'] = self.m
        return base_config


class GAL(layers.Layer):  # GraphAttentionLayer
    def __init__(self,
                 units,  # 输出维度
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 manifold=Manifold(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GAL, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.m = manifold

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            # assign_to_manifold(self.bias, self.m)  # 对b黎曼优化
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):  # 输入欧式, 输出流形
        features = inputs[0]  # 特征向量
        adj = inputs[1]  # 标准化邻接矩阵
        # x * W
        x = tf.matmul(features, self.kernel)
        x = self.m.expmap(x)
        # a, 距离作为注意力权重, 原版欧式GAT是weight求和+relu激活
        a = self.m.sqdist(x, x, cross_join=True) ** 0.5
        a = a * adj
        a = tf.where(a != 0, a, tf.float64.min)  # 防止softmax计算无边0值
        a = tf.keras.activations.softmax(a)
        # a * xW
        x = self.m.mobius_matvec(x, a, exchange=True)
        # + b
        if tf.is_tensor(self.bias):
            # x += self.bias
            b = self.bias
            b = self.m.expmap(b)  # 对b黎曼优化应去除此行
            x = self.m.mobius_add(x, b)
        return x


class MLP(layers.Layer):
    def __init__(self, output_dim, manifold=Manifold(), use_kernel=True, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.manifold = manifold
        self.use_bias = use_bias
        self.use_kernel = use_kernel
        super(MLP, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[-1], self.output_dim))
        if self.use_kernel:
            w = tf.random.normal(shape, mean=0, stddev=0.1)
            self.kernel = self.add_weight(name='kernel',
                                          shape=shape,
                                          initializer=tf.keras.initializers.constant(w.numpy()),
                                          trainable=True,
                                          )
        if self.use_bias:
            b = tf.random.normal([self.output_dim], mean=0, stddev=0.1)
            self.bias = self.add_weight(name='bias',
                                        shape=[self.output_dim],
                                        initializer=tf.keras.initializers.constant(b.numpy()),
                                        trainable=True,
                                        )
            # assign_to_manifold(self.bias, self.m)  # 对b黎曼优化
        super(MLP, self).build(input_shape)

    def call(self, inputs, **kwargs):  # 输入流形, 输出流形
        x = inputs
        if self.use_kernel:
            w = self.kernel
            x = self.manifold.mobius_matvec(x, w)
        if self.use_bias:
            b = self.bias
            b = self.manifold.expmap(b)  # 对b黎曼优化应去除此行
            x = self.manifold.mobius_add(x, b)
        return x


class ActLayer(layers.Layer):
    def __init__(self, manifold=Manifold(), activation='relu', actM=None, name_pre=None, **kwargs):
        """
        激活层
        :param manifold: Manifold(); 指层输入向量所在的流形
        :param activation: None or str; 激活函数类型, None表示不使用激活
        :param actM: None or int; 激活函数转到哪个空间上做, 与Manifold.s含义一致, None表示与manifold流形一致
        :param name_pre: None or str; 不为空则自动替换name, 后缀自动补充 .relu.1-2-1 这种流形转换描述, 表示relu在流形2上执行
        :param kwargs:
        """
        self.activation = activations.get(activation) if isinstance(activation, str) else None
        self.m = manifold
        self.actM = manifold.s if actM is None else actM
        if name_pre:
            kwargs['name'] = f'{name_pre}.{activation}.{manifold.s}-{self.actM}-{manifold.s}'
            if self.activation is None:
                kwargs['name'] = f'{name_pre}.no-act'
        super(ActLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):  # 输入流形, 输出流形
        x = inputs
        if self.activation is None:
            ...
        elif self.m.s != self.actM:
            x = self.m.to_other_manifold(x, tm=self.actM)
            x = self.activation(x)
            x = self.m.to_other_manifold(x, om=self.actM, tm=self.m.s)
        else:
            x = self.activation(x)
        return x


class HyperLayer:  # 用于 Encoder
    @staticmethod
    def gcn(fea_input, adj_input, dim_out: list, manifold=Manifold(), act_L=None, inM=0, outM=None, actM_L=None,
            **kwargs):  # 输入欧式, 输出流形
        '''
        多流形GCN
        :param fea_input: Tensor, 特征向量
        :param adj_input: Tensor, 标准化邻接矩阵
        :param dim_out: list, 每一层 gcn 的输出维度
        :param manifold: 流形
        :param act_L: list or None, 每一层使用什么激活函数, 为None表示只有最后一层不使用relu激活函数(最后一层relu导致没有负数)
            list中一个为空表示那一层不使用激活函数
        :param actM_L: list or None or int, 每一层激活函数在什么流形空间上转换, None表示不转换, int与Manifold.s含义一致
            int表示所有层都转到int对应的流形上运算,list就是分别做
        :param inM: None or int; fea_input 所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param outM: None or int; 输出所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :return: Tensor, 计算出的向量
        '''
        assert act_L is None or len(act_L) == len(dim_out), 'len(act_L) != len(dim_out), %d != %d' % (
            len(act_L), len(dim_out))
        if act_L is None:
            act_L = ['relu'] * (len(dim_out) - 1) + [None]
        if actM_L is None or isinstance(actM_L, int):
            actM_L = [actM_L] * len(dim_out)
        if inM is None:
            inM = manifold.s
        if outM is None:
            outM = manifold.s
        # 流形转换需要满足 GraphConvolution 要求, 在欧式上
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, om=inM, tm=0),
                            name=f'Encoder.m.in.{inM}to0')(fea_input)
        # 网络
        for i, (dim, act, actM) in enumerate(zip(dim_out, act_L, actM_L)):
            net = GraphConvolution(dim, manifold=manifold, name='Encoder.GCN-%d' % i)([net, adj_input])
            net = ActLayer(manifold=manifold, activation=act, actM=actM, name_pre='Encoder.GCN-%d' % i)(net)
            if i < len(dim_out) - 1:  # 最后一层无需转欧式
                net = layers.Lambda(lambda x: manifold.logmap(x), name='Encoder.GCN-%d.log' % i)(net)
        # 流形输出要求
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, tm=outM),
                            name=f'Encoder.m.out.{manifold.s}to{outM}')(net)
        return net

    @staticmethod
    def gat(fea_input, adj_input, dim_out: list, manifold=Manifold(), act_L=None, inM=0, outM=None, head_L=None,
            actM_L=None, **kwargs):  # 输入欧式, 输出流形
        '''
        多流形GCN
        :param fea_input: Tensor, 特征向量
        :param adj_input: Tensor, 标准化邻接矩阵
        :param dim_out: list, 每一层 gcn 的输出维度
        :param manifold: 流形
        :param act_L: list or None, 每一层使用什么激活函数, 为None表示只有最后一层不使用relu激活函数(最后一层relu导致没有负数)
            list中一个为空表示那一层不使用激活函数
        :param actM_L: list or None or int, 每一层激活函数在什么流形空间上转换, None表示不转换, int与Manifold.s含义一致
            int表示所有层都转到int对应的流形上运算,list就是分别做
        :param inM: None or int; fea_input 所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param outM: None or int; 输出所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param head_L: list or None or int, 每一层 gat 使用几个注意力头, 为None表示全部使用4个, 为int表示每层都是int
        :return: Tensor, 计算出的向量
        '''
        assert act_L is None or len(act_L) == len(dim_out), 'len(act_L) != len(dim_out), %d != %d' % (
            len(act_L), len(dim_out))
        if act_L is None:
            act_L = ['relu'] * (len(dim_out) - 1) + [None]
        if actM_L is None or isinstance(actM_L, int):
            actM_L = [actM_L] * len(dim_out)
        if inM is None:
            inM = manifold.s
        if outM is None:
            outM = manifold.s
        if head_L is None:
            head_L = [4] * len(dim_out)
        if isinstance(head_L, int):
            head_L = [head_L] * len(dim_out)

        def headGAT(units, act, actM, name, features, n_head):  # 多头GAT, 输入欧式, 输出欧式
            gat_L = []
            for i in range(n_head):
                gat = GAL(units, manifold=manifold, name=f'{name}.h{i + 1}')([features, adj_input])
                gat_L.append(gat)
            net = layers.Lambda(lambda x: tf.reduce_mean([manifold.logmap(j) for j in x], 0),
                                name=f'{name}.log-mean')(gat_L)
            net = ActLayer(activation=act, actM=actM, name_pre=name)(net)  # 激活函数也可以放在多头注意力平均之前
            return net

        # 流形转换需要满足 headGAT 要求, 在欧式上
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, om=inM, tm=0),
                            name=f'Encoder.m.in.{inM}to0')(fea_input)
        # 网络
        for i, (dim, act, n_head, actM) in enumerate(zip(dim_out, act_L, head_L, actM_L)):
            net = headGAT(units=dim, act=act, actM=actM, name=f'Encoder.GAT-{i}', features=net, n_head=n_head)
        # 流形输出要求
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, om=0, tm=outM),
                            name=f'Encoder.m.out.0to{outM}')(net)
        return net

    @staticmethod
    def mlp(fea_input, dim_out: list, manifold=Manifold(), inM=0, outM=None, act_L=None, actM_L=None, **kwargs):
        '''
        多流形MLP
        :param fea_input: Tensor, 输入特征向量, shape=(None, dim_in)
        :param dim_out: 每一层mlp的输出维度
        :param manifold: 流形
        :param inM: None or int; fea_input 所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param outM: None or int; 输出所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param act_L: list or None, 每一层使用什么激活函数, 为None表示只有最后一层不使用relu激活函数(最后一层relu导致没有负数)
            list中一个为空表示那一层不使用激活函数
        :param actM_L: list or None or int, 每一层激活函数在什么流形空间上转换, None表示不转换, int与Manifold.s含义一致
            int表示所有层都转到int对应的流形上运算,list就是分别做
        :return:
        '''

        def no(i):
            """
            不转到欧式上做损失, 使用方式:
                (layers.Lambda(lambda x: manifold.logmap(x), name='Encoder.log-%d' % i) if 转欧式 else no(i))(net)
                (layers.Lambda(lambda x: tf.keras.activations.relu(x), name='Encoder.act-%d' % i))(net)
                (layers.Lambda(lambda x: manifold.expmap(x), name='Encoder.exp-%d' % i) if acte else no(i))(net)
            :param i: 序号
            :return:
            """
            return layers.Lambda(lambda x: x, name='D.no-%d' % i)

        assert act_L is None or len(act_L) == len(dim_out), 'len(act_L) != len(dim_out), %d != %d' % (
            len(act_L), len(dim_out))
        if inM is None:
            inM = manifold.s
        if outM is None:
            outM = manifold.s
        if act_L is None:
            act_L = ['relu'] * (len(dim_out) - 1) + [None]
        if actM_L is None or isinstance(actM_L, int):
            actM_L = [actM_L] * len(dim_out)
        net = fea_input
        # 流形转换需要满足 MLP 要求, 在双曲上
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, om=inM, tm=manifold.s),
                            name=f'Encoder.m.in.{inM}to{manifold.s}')(net)
        for i, (dim, act, actM) in enumerate(zip(dim_out, act_L, actM_L)):
            net = MLP(dim, manifold=manifold, name='Encoder.MLP-%d' % i)(net)
            net = ActLayer(manifold=manifold, activation=act, actM=actM, name_pre='Encoder.MLP-%d' % i)(net)
        # 流形输出要求
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, tm=outM),
                            name=f'Encoder.m.out.{manifold.s}to{outM}')(net)
        return net

    @staticmethod
    def fixed(fea_input, manifold=Manifold(), inM=0, outM=None, **kwargs):
        '''
        无参数训练的空白 Encoder, 一般用于预训练encoder固定参数
        :param fea_input: Tensor, 输入特征向量, shape=(None, dim_in)
        :param manifold: 流形
        :param inM: None or int; fea_input 所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param outM: None or int; 输出所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :return:
        '''
        if inM is None:
            inM = manifold.s
        if outM is None:
            outM = manifold.s
        net = fea_input
        # 流形输出要求
        net = layers.Lambda(lambda x: manifold.to_other_manifold(x, om=inM, tm=outM),
                            name=f'Encoder.m.out.{inM}to{outM}')(net)
        return net


class Encoder:
    def __init__(self, nodes_num, feat_dim, manifold=Manifold(), encoderOutDim=(100, 2), layer=HyperLayer.gcn,
                 layerManifold=None, inM=0, layer_paras=None, compute_all_dist=True):
        '''
        输入 inM 流形的 features, 经过 layerManifold 流形的模型, 输出 manifold 流形的向量
        增加参数时需要考虑 new_encoder 方法
        :param nodes_num: int, 节点数量
        :param feat_dim: int, 初始特征维度
        :param manifold: Manifold(); encoder 输出的流形
        :param encoderOutDim: list or tuple, 编码器每一层的输出维度
        :param layer: HyperLayer; encoder 层的神经网络
        :param layerManifold: None or Manifold(); encoder 层的流形, None表示与manifold一致
        :param inM: None or int; Input.features 所在流形, 与Manifold.s含义一致, None表示与manifold流形一致
        :param layer_paras: None or dict; layer 的额外参数, 参考 HyperLayer 对应参数解释, 比如 act_L/actM_L
        :param compute_all_dist: _dist_model是否一次性计算好所有点之间距离, 对于大矩阵必须用这个, 否则空间复杂度太高?
        '''
        if layer_paras is None:
            layer_paras = {}
        if inM is None:
            inM = manifold.s
        if layerManifold is None:
            layerManifold = manifold
        # model
        features = layers.Input(batch_shape=(nodes_num, feat_dim), name='Input.features')  # 原始的节点特征向量矩阵
        adj_train = layers.Input(batch_shape=(nodes_num, nodes_num), name='Input.adj_train')  # 训练用的标准化邻接矩阵
        embedding = layer(fea_input=features, adj_input=adj_train, dim_out=encoderOutDim,
                          manifold=layerManifold, inM=inM, outM=manifold.s, **layer_paras)  # encoder 层
        model = Model(inputs=[features, adj_train], outputs=embedding, name='Encoder')  # 模型
        self.model = model
        # other
        self.manifold = manifold
        self.encoderOutDim = encoderOutDim
        self.compute_all_dist = compute_all_dist
        self.dist_model = self._dist_model()  # 距离模型
        # 主要用于copy/初始化新的Encoder
        self.nodes_num = nodes_num
        self.feat_dim = feat_dim
        self.layer = layer
        self.layerManifold = layerManifold
        self.inM = inM
        self.layer_paras = layer_paras
        self.compute_all_dist = compute_all_dist

    def getData(self, d, pre=''):
        """
        dataset 获取方法. 因不同数据划分, 标准化邻接矩阵可能因数据不同而不同, 但是特征需要一致
        :param d: dict; dataHelper.getDataset() 输出数据
        :param pre: str; 数据集前缀
        :return:
        """
        features = d[f'{pre}features']
        if not isinstance(features, np.ndarray):  # 可能是 <class 'scipy.sparse.dia.dia_matrix'> 类型
            features = features.toarray()
        adj_train = d[f'{pre}adj_train']
        if not isinstance(adj_train, np.ndarray):  # 可能是 <class 'scipy.sparse.dia.dia_matrix'> 类型
            adj_train = adj_train.toarray()
        return [features, adj_train]  # 顺序与模型输入一致

    def new_encoder(self, layer=None, useOriginalOutputAsInput=True):
        """
        根据本参数初始化一个新的Encoder, 一般用于混合非损失函数预训练encoder
        :param layer: HyperLayer or None; encoder 层的神经网络, None则使用本Encoder自带的
        :param useOriginalOutputAsInput: bool; 是否使用原始encoder的输出作为新建encoder的输入
        :return: Encoder()
        """
        if layer is None:
            layer = self.layer
        if useOriginalOutputAsInput:
            feat_dim = self.encoderOutDim[-1]
            inM = self.manifold.s
        else:
            feat_dim = self.feat_dim
            inM = self.inM
        return Encoder(nodes_num=self.nodes_num, feat_dim=feat_dim, manifold=self.manifold,
                       encoderOutDim=self.encoderOutDim, layer=layer, layerManifold=self.layerManifold, inM=inM,
                       layer_paras=self.layer_paras, compute_all_dist=self.compute_all_dist)

    def copy_encoder(self):
        """
        浅拷贝encoder, 除了model使用深拷贝权重, 一般用于train返回效果最好的那次encoder结果
        :return:
        """
        # encoder = copy.copy(self)
        # 这种方式需要大改Manifold, 否则会出现 TypeError: ('Not JSON Serializable:', <_ai_manifold.Manifold object at 0x7feebae17750>)
        # savepath = 'ad_encoder.model.tmp'
        # self.model.save(savepath)
        # encoder.model = tf.keras.models.load_model(savepath)
        # 这种方式需要增加完善的 get_config 方法保存权重
        # encoder.model = tf.keras.models.clone_model(self.model)
        encoder = self.new_encoder(useOriginalOutputAsInput=False)
        for a, b in zip(encoder.model.variables, self.model.variables):
            a.assign(b)  # copies the variables of model_b into model_a
        return encoder

    def _dist_model(self):
        """
        计算不同转流形的距离模型
        :return:
        """
        embedding = layers.Input(batch_shape=(None, self.encoderOutDim[-1]), name='Input.embedding')
        edges = layers.Input(batch_shape=(None, 2), name='Input.edges', dtype=tf.int32)
        # 获得不同流形的嵌入
        toM = self.manifold.to_other_manifold
        if self.manifold.s == 0:
            em0 = layers.Lambda(lambda x: x, name='M.0_to_0')(embedding)
            em1 = layers.Lambda(lambda x: toM(x, 0, 1), name='M.0_to_1')(embedding)
            em2 = layers.Lambda(lambda x: toM(x, 0, 2), name='M.0_to_2')(embedding)
        elif self.manifold.s == 1:
            em0 = layers.Lambda(lambda x: toM(x, 1, 0), name='M.1_to_0')(embedding)
            em1 = layers.Lambda(lambda x: x, name='M.1_to_1')(embedding)
            em2 = layers.Lambda(lambda x: toM(x, 1, 2), name='M.1_to_2')(embedding)
        elif self.manifold.s == 2:
            em0 = layers.Lambda(lambda x: toM(x, 2, 0), name='M.2_to_0')(embedding)
            em1 = layers.Lambda(lambda x: toM(x, 2, 1), name='M.2_to_1')(embedding)
            em2 = layers.Lambda(lambda x: x, name='M.2_to_2')(embedding)
        else:
            raise NameError('错误的流形!')
        em_all = [em0, em1, em2]  # 顺序与流形顺序一致
        dist_L_L = []  # 不同流形的距离计算层
        for i in range(len(em_all)):
            dist_L = layers.Lambda(lambda x, i=i: tf.squeeze(self.manifold.sqdist(x[0], x[1], s=i) ** 0.5, axis=1),
                                   name='dist-%d' % i)
            dist_L_L.append(dist_L)
        em_dist = []
        # 是否一次性计算好所有点之间距离
        if self.compute_all_dist:
            for i, em in enumerate(em_all):
                em_all_dist = layers.Lambda(lambda x, i=i: self.manifold.sqdist(x, x, s=i, cross_join=True) ** 0.5,
                                            name='dist_mat-%d' % i)(em)
                em_dist.append(layers.Lambda(lambda x, i=i: tf.gather_nd(x[0], x[1]),
                                             name='look_up-%d' % i)([em_all_dist, edges]))
        else:
            # 边的左右节点分离
            l_nodes = layers.Lambda(lambda x: x[..., 0], name='left_nodes')(edges)
            r_nodes = layers.Lambda(lambda x: x[..., 1], name='right_nodes')(edges)
            # 距离计算所需层
            lookup_L = layers.Lambda(lambda x: tf.gather(x[0], x[1], axis=0), name='look_up')
            # 开始距离计算
            # lookup_LR = []  # 测试
            for i, em in enumerate(em_all):
                # lookup_LR.append([lookup_L([em, l_nodes]), lookup_L([em, r_nodes])])  # 测试
                em_dist.append(dist_L_L[i]([lookup_L([em, l_nodes]), lookup_L([em, r_nodes])]))
        # 计算和原点的距离
        em_dist_0 = []  # 和原点的距离
        originNodes = layers.Lambda(lambda x: tf.zeros_like(x), name='originNodes')(embedding)  # 原点统一为0
        for i, em in enumerate(em_all):
            em_dist_0.append(dist_L_L[i]([em, originNodes]))
        # model
        model = Model(inputs=[embedding, edges], outputs=[em_all, em_dist, em_dist_0], name='dist_model')
        return model

    def get_mutilManifold_embeddingANDdist(self, embedding, edges=None):
        '''
        获得encoder嵌入向量的不同流形, 以及和原点和其他点之间的距离
        :param embedding: Tensor, 每个节点的嵌入(encoder输出的). 节点序号就是编号
        :param edges: [(点l,点r),..] or None, 如果为 None 则计算所有点之间距离. 点编号从0开始与嵌入对应.
        :return:
        '''
        # 注意: 不同数据集可能 adj_train 不一样
        # if embedding is None:
        #     embedding = self.model(self.getData(data), training=False)
        nodes_num = embedding.shape[0]  # 节点数量
        # 需要计算距离的边
        if edges is not None and len(edges) > 0:
            # 去除重复边/自环边/下三角边
            edges = list(set([tuple(sorted(i)) for i in edges if i[0] != i[1]]))
            edges = np.array(edges)
        else:
            ones = np.ones([nodes_num, nodes_num])  # 全1矩阵
            edges = np.where((ones - np.tril(ones)) == 1)  # 减去下三角矩阵, 然后提取索引就是穷举的边
            edges = np.dstack(edges)[0]  # 沿着纵轴方向组合, 会升维1度, 再降1维度
        em_all, em_dist, em_dist_0 = self.dist_model([embedding, edges], training=False)
        out = []
        for i, (em, dist, dist_0) in enumerate(zip(em_all, em_dist, em_dist_0)):
            em, dist, dist_0 = em.numpy(), dist.numpy(), dist_0.numpy()
            out.append({
                'embedding': em,
                'dist': [],  # [[点l,点r,距离],..], 点之间距离, np.array 格式
                'dist_nan': [],  # [(点l,点l嵌入,点r,点r嵌入),..], 记录距离为nan的点
                'dist_0': [],  # [(点,离原点距离),..], 点与原点距离
                'dist_0_nan': [],  # [(点,点嵌入),..], 记录距离为nan的点
            })
            # 点之间距离
            out[-1]['dist'] = np.hstack((edges, np.expand_dims(dist, axis=1)))
            # dist_nan 值统计
            edges_nan = edges[np.where(np.isnan(dist))[0]]
            out[-1]['dist_nan'] = [(l, em[l].tolist(), r, em[r].tolist()) for l, r in edges_nan]
            # 对于4017节点, 以下距离和nan值统计代码耗时可能超过15秒
            # for (l, r), d in zip(edges, dist):
            #     out[-1]['dist'].append((l, r, d))
            #     if math.isnan(d):
            #         out[-1]['dist_nan'].append((l, em[l].tolist(), r, em[r].tolist()))
            # 点与原点距离
            for j, d in enumerate(dist_0):
                out[-1]['dist_0'].append((j, d))
                if math.isnan(d):
                    out[-1]['dist_0_nan'].append((j, em[j].tolist()))
        return out

    def summary(self, outPath):
        '''
        :param outPath: str or None, 模型结构的输出路径, 会自动加后缀输出多个模型
        :return:
        '''
        print('=' * 10 + 'encoder 模型')
        self.model.summary()
        print('=' * 10 + '距离计算模型')
        self.dist_model.summary()
        if outPath:
            # 整体模型
            printModelG(self.model, outPath)
            # 距离计算模型
            fname, fextension = os.path.splitext(outPath)
            path = fname + '-dist' + fextension  # 小心重名覆盖
            printModelG(self.dist_model, path)


class Decoder:
    def __init__(self, learning_rate=0.01, **kwargs):
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self._optimizer = None

        self.set_optimizer(learning_rate=learning_rate)

    def set_optimizer(self, learning_rate=0.01, optimizer=RiemannianAdam):
        self._optimizer = optimizer(learning_rate=learning_rate)

    def reDataSetting(self, *args, **kwargs):
        raise NotImplementedError

    def decoder(self, *args, **kwargs):
        raise NotImplementedError

    def loss_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def loss_name(self, *args, **kwargs):
        raise NotImplementedError

    def metric_name(self, *args, **kwargs):
        raise NotImplementedError

    def all_model(self, *args, **kwargs):
        raise NotImplementedError

    def summary(self, outPath):
        '''
        :param outPath: str or None, 模型结构的输出路径, 会自动加后缀输出多个模型
        :return:
        '''
        print('=' * 10 + 'decoder 模型')
        self.model.summary()
        if outPath:
            printModelG(self.model, outPath)

    def check_data(self, dataType='train'):
        """
        验证数据集是否存在, 防止空数据集 compute 错误
        :param dataType: str, 数据类型, train/dev/test
        :return:
        """
        if dataType == 'train':
            data = self.train_data
        elif dataType == 'dev':
            data = self.dev_data
        elif dataType == 'test':
            data = self.test_data
        else:
            raise NameError('错误的类型数据!')
        if data is not None:
            for i in data:
                if i is None or len(i) == 0:
                    return None
        return data

    def compute(self, training=True, dataType='train', check=False):
        '''
        训练方法, 需要保证子类模型有相同的模型输出
        :param training: bool, 是否训练
        :param dataType: str, 数据类型, train/dev/test
        :param check: bool, 是否直接终止并输出梯度的情况
        :return:
        '''
        data = self.check_data(dataType)
        if data is None:
            raise NameError(f'该数据类型没有数据! dataType={dataType}')

        out = {}
        if training:
            with tf.GradientTape() as tape:
                embedding, loss, metric = self.model(data)[:3]
            grads = tape.gradient(loss, self.model.trainable_variables)
            grad_names = [i.name for i in self.model.trainable_variables]

            # 检查: 计算梯度为nan/inf的情况
            grads_nan_num = 0
            grads_num = 0
            gradss_vaild_num = 0
            for i in grads:
                if isinstance(i, tf.IndexedSlices):
                    i = i.values
                if i is None:
                    continue
                grads_nan_num += np.count_nonzero(np.isnan(i) | np.isinf(i))
                grads_num += np.prod(i.shape)
                gradss_vaild_num += 1
            if check or grads_nan_num > 0:
                print('%d个有效梯度张量%.2f%%为nan/inf, 当前平均loss：' %
                      (gradss_vaild_num, grads_nan_num / grads_num * 100))
                print('-' * 5 + '本轮训练得到的权重梯度')
                for i, j in enumerate(grads):
                    print(get_stat(j, '%d-%s' % (i + 1, grad_names[i]), get_out=2))
                print('-' * 5 + '重新计算当前网络所有层所有张量的输出和梯度')
                get_out_grad(self.model, data, 1)
                if grads_nan_num > 0:
                    raise ValueError('梯度或loss出现异常!')
                else:
                    raise ValueError('check=True !')

            grads = [tf.clip_by_value(g, -1e+14, 1e+14) for g in grads]
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        else:
            embedding, loss, metric = self.model(data, training=training)[:3]

        out['embedding'] = embedding
        out['loss'] = loss
        out['metric'] = metric
        return out

    @staticmethod
    def dhToDecoder(dh):
        """
        返回 DataHelper 对应的 Decoder
        :param dh: DataHelper or DataHelper的实例化 or DataHelper.__name__
        :return: Decoder, DataHelper.__name__, DataHelper
        """
        try:
            if isinstance(dh, str):
                name = dh
            else:
                name = dh.__name__  # DataHelper
        except:
            name = dh.__class__.__name__  # DataHelper的实例化
        if name == Classification.__name__:
            decoder = Decoder_classification
            dh = Classification
        elif name == LinkPred.__name__:
            decoder = Decoder_linkPred
            dh = LinkPred
        elif name == GraphDistor.__name__:
            decoder = Decoder_graphDistor
            dh = GraphDistor
        elif name == HypernymyRel.__name__:
            decoder = Decoder_hypernymyRel
            dh = HypernymyRel
        elif name == MixedDataset.__name__:
            decoder = Decoder_mixedLoss
            dh = MixedDataset
        else:
            raise NameError('错误的模型:', name)
        return decoder, name, dh


class Decoder_linkPred(Decoder):
    def __init__(self, dataset, encoder: Encoder, r=2., t=1., pre='', complile=True, **kwargs):
        '''
        :param dataset: data helper 的输出数据
        :param encoder: Encoder, 编码器模型
        :param r: 费米-狄拉克分布 超参数半径
        :param t: 费米-狄拉克分布 超参数比例
        :param pre: str; 数据集名称的前缀, 一般用于混合数据/多损失函数
        :param complile: bool; 是否编译, 一般用于多损失函数结合, 直接训练不能为False
        '''
        super(Decoder_linkPred, self).__init__(**kwargs)
        self.sh = 'LR'  # 类名的简写
        # dataset
        self.pre = pre
        self.encoder = encoder
        self.reDataSetting(dataset)
        # model
        self.complile = complile
        self.model = self.all_model(r=r, t=t)

    def reDataSetting(self, dataset):
        '''
        :param dataset: data helper 的输出数据
        :return:
        '''
        pre = self.pre
        self.train_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}train_edges_pos'], dataset[f'{pre}train_edges_neg'])
        self.test_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}test_edges_pos'], dataset[f'{pre}test_edges_neg'])
        self.dev_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}dev_edges_pos'], dataset[f'{pre}dev_edges_neg'])

    def decoder(self, edgesPos, edgesNeg, embedding, manifold=Manifold(), r=2., t=1., **kwargs):
        '''
        链路预测的解码器
        :param edgesPos: 正例边二元组
        :param edgesNeg: 负例边二元组
        :param embedding: 编码器输出的特征矩阵
        :param manifold: 流形
        :param r: 费米-狄拉克分布 超参数半径
        :param t: 费米-狄拉克分布 超参数比例
        :return: (Tensor, Tensor), (正例发生概率, 负例发生概率)
        '''
        # 层
        lookup_L = layers.Lambda(lambda x: tf.gather(x[0], x[1], axis=0), name=f'{self.sh}.D.look_up')
        dist_L = layers.Lambda(lambda x: tf.squeeze(manifold.sqdist(x[0], x[1]) ** 0.5, axis=-1),
                               name=f'{self.sh}.D.dist')
        probs = layers.Lambda(lambda x: 1. / (tf.exp((x - r) / t) + 1.0), name=f'{self.sh}.D.probs')
        l_nodes = layers.Lambda(lambda x: x[..., 0], name=f'{self.sh}.D.left_nodes')
        r_nodes = layers.Lambda(lambda x: x[..., 1], name=f'{self.sh}.D.right_nodes')
        # 正例距离计算
        edgesPosLeftNodes = lookup_L([embedding, l_nodes(edgesPos)])
        edgesPosRightNodes = lookup_L([embedding, r_nodes(edgesPos)])
        pos_dist = dist_L([edgesPosLeftNodes, edgesPosRightNodes])

        # 测试 poincare 为 nan 的原因, 不可导点 √x,x=0
        # pos_dist = [edgesPosLeftNodes, edgesPosRightNodes]
        # test_L = [
        #     layers.Lambda(lambda x: manifold.mobius_add(-x[0], x[1], s=2), name='Decoder.test.mobius_add'),
        #     layers.Lambda(lambda x: manifold.inner(x, s=0), name='Decoder.test.inner'),
        #     layers.Lambda(lambda x: tf.atanh(x), name='Decoder.test.atanh'),
        #     layers.Lambda(lambda x: x ** 0.5, name='Decoder.test.sqt'),
        #     layers.Lambda(lambda x: tf.squeeze(x, axis=1), name='Decoder.test.squeeze'),
        # ]
        # for i in test_L:
        #     pos_dist = i(pos_dist)

        # 负例距离计算
        edgesNegLeftNodes = lookup_L([embedding, l_nodes(edgesNeg)])
        edgesNegRightNodes = lookup_L([embedding, r_nodes(edgesNeg)])
        neg_dist = dist_L([edgesNegLeftNodes, edgesNegRightNodes])
        # 概率值计算
        pos_probs = probs(pos_dist)
        neg_probs = probs(neg_dist)
        return pos_probs, neg_probs

    def loss_metrics(self, pos_probs, neg_probs):
        # 损失
        loss = layers.Lambda(lambda x: - tf.math.log(x[0]) - tf.math.log(1 - x[1]),
                             name=f'{self.sh}.L.crossentropy')([pos_probs, neg_probs])
        loss = layers.Lambda(lambda x: tf.reduce_sum(x), name=f'{self.sh}.L.sum')(loss)
        # 评估
        metric = layers.Lambda(lambda x: tf.cast(tf.greater(x[0], x[1]), tf.float32),
                               name=f'{self.sh}.M.acc')([pos_probs, neg_probs])
        metric = layers.Lambda(lambda x: tf.reduce_mean(x), name=f'{self.sh}.M.acc.mean')(metric)
        return loss, metric

    def loss_name(self, latex=True):
        if latex:
            return '$\\mathcal{L}_{3}$'
        else:
            return 'L3'

    def metric_name(self):
        return 'acc'

    def all_model(self, r=2., t=1.):
        '''
        链路预测的整体模型
        :param r: 费米-狄拉克分布 超参数半径
        :param t: 费米-狄拉克分布 超参数比例
        :return:
        '''
        # 输入
        edgesPos = layers.Input(batch_shape=(None, 2), name=f'{self.sh}.Input.edgesPos',
                                dtype=tf.int32)  # train/dev/test 正例边
        edgesNeg = layers.Input(batch_shape=(None, 2), name=f'{self.sh}.Input.edgesNeg',
                                dtype=tf.int32)  # train/dev/test 负例边
        # 编码器
        embedding = self.encoder.model.output
        # 解码器
        pos_probs, neg_probs = self.decoder(edgesPos, edgesNeg, embedding, self.encoder.manifold, r, t)
        # 损失函数
        loss, metric = self.loss_metrics(pos_probs, neg_probs)
        # 模型
        model = Model(inputs=self.encoder.model.inputs + [edgesPos, edgesNeg], outputs=[embedding, loss, metric],
                      name=f'{self.sh}.All_model')
        if self.complile:
            model.compile(optimizer=self._optimizer, loss=lambda y_true, y_pred: y_pred[1])
        return model


class Decoder_classification(Decoder):
    def __init__(self, dataset, encoder: Encoder, pre='', complile=True, **kwargs):
        '''
        :param dataset: data helper 的输出数据
        :param encoder: Encoder, 编码器模型
        :param pre: str; 数据集名称的前缀, 一般用于混合数据
        :param complile: bool; 是否编译, 一般用于多损失函数结合, 直接训练不能为False
        '''
        super(Decoder_classification, self).__init__(**kwargs)
        self.sh = 'NC'  # 类名的简写
        # dataset
        self.pre = pre
        self.encoder = encoder
        self.reDataSetting(dataset)
        # model
        self.complile = complile
        self.model = self.all_model(dataset[f'{pre}classNum'], inEuropean=False)

    def reDataSetting(self, dataset):
        '''
        :param dataset: data helper 的输出数据
        :return:
        '''
        pre = self.pre
        self.train_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}train_nodes'], dataset[f'{pre}train_nodes_class'])
        self.test_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}test_nodes'], dataset[f'{pre}test_nodes_class'])
        self.dev_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}dev_nodes'], dataset[f'{pre}dev_nodes_class'])

    def decoder(self, nodes, embedding, classNum, manifold=Manifold(), inEuropean=False, **kwargs):
        '''
        节点分类的解码器
        :param nodes: train/dev/test 点
        :param embedding: 编码器的输出向量
        :param classNum: int, 类别数量
        :param manifold: embedding所在流形
        :param inEuropean: 是否在 decoder 前将空间转到欧式. 这个可以不用, 变成由encoder的输出流形决定
        :return: Tensor, 解码之后的节点向量
        '''
        # 计算softmax
        nodes_embedding = layers.Lambda(lambda x: tf.gather(x[0], x[1], axis=0),
                                        name=f'{self.sh}.D.look_up')([embedding, nodes])
        if inEuropean:
            nodes_embedding = layers.Lambda(lambda x: manifold.logmap(x), name=f'{self.sh}.D.logmap')(
                nodes_embedding)
        # nodes_embedding = layers.Dense(classNum, name=f'{self.sh}.D.MLP')(nodes_embedding)  # 欧式MLP
        nodes_embedding = MLP(classNum, manifold=manifold, name=f'{self.sh}.D.MLP')(nodes_embedding)
        nodes_embedding = layers.Lambda(lambda x: tf.keras.activations.softmax(x),
                                        name=f'{self.sh}.D.softmax')(nodes_embedding)
        return nodes_embedding

    def loss_metrics(self, fea_input, labels_input):
        '''
        :param fea_input: decoder 后的特征向量
        :param labels_input: 对应特征向量的类别标签
        :return: 每个样本的loss, 每个样本的acc
        '''
        # 标签
        labels = layers.Lambda(lambda x: tf.one_hot(x, fea_input.shape[1]),
                               name=f'{self.sh}.L.label_one_hot')(labels_input)
        # 损失
        loss = layers.Lambda(
            lambda x: K.categorical_crossentropy(x[0], x[1], from_logits=False),
            name=f'{self.sh}.L.crossentropy'
        )([labels, fea_input])
        loss = layers.Lambda(lambda x: tf.reduce_sum(x), name=f'{self.sh}.L.sum')(loss)
        # 评估
        pred_class = layers.Lambda(lambda x: tf.argmax(x, 1, output_type=tf.int32),
                                   name=f'{self.sh}.M.pred_class')(
            fea_input)
        metric = layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32), name=f'{self.sh}.M.acc'
        )([pred_class, labels_input])
        metric = layers.Lambda(lambda x: tf.reduce_mean(x), name=f'{self.sh}.M.acc.mean')(metric)
        return loss, metric

    def loss_name(self, latex=True):
        if latex:
            return r'$\mathcal{L}_{4}$'
        else:
            return 'L4'

    def metric_name(self):
        return 'acc'

    def all_model(self, classNum, inEuropean=False):
        '''
        节点分类的整体模型
        :param classNum: 节点的类别数量
        :param inEuropean: 是否在 decoder 前将空间转到欧式
        :return:
        '''
        # 输入
        nodes_pred = layers.Input(batch_shape=(None,), name=f'{self.sh}.Input.nodes_pred',
                                  dtype=tf.int32)  # train/dev/test 点
        labels = layers.Input(batch_shape=(None,), name=f'{self.sh}.Input.labels', dtype=tf.int32)  # 标签, 与点对应
        # 编码器
        embedding = self.encoder.model.output
        # 解码器
        nodes_embedding = self.decoder(nodes_pred, embedding, classNum, self.encoder.manifold, inEuropean)
        # 损失函数 / 评估
        loss, metric = self.loss_metrics(nodes_embedding, labels)
        # 模型
        model = Model(inputs=self.encoder.model.inputs + [nodes_pred, labels], outputs=[embedding, loss, metric],
                      name=f'{self.sh}.All_model')
        if self.complile:
            model.compile(optimizer=self._optimizer, loss=lambda y_true, y_pred: y_pred[1])
        return model


class Decoder_graphDistor(Decoder):
    def __init__(self, dataset, encoder: Encoder, pre='', complile=True, compute_all_dist=True, **kwargs):
        '''
        :param dataset: data helper 的输出数据
        :param encoder: Encoder, 编码器模型
        :param pre: str; 数据集名称的前缀, 一般用于混合数据
        :param complile: bool; 是否编译, 一般用于多损失函数结合, 直接训练不能为False
        :param compute_all_dist: 是否一次性计算好所有点之间距离, 对于大矩阵必须用这个, 否则空间复杂度太高?
        '''
        super(Decoder_graphDistor, self).__init__(**kwargs)
        self.sh = 'GD'  # 类名的简写
        # dataset
        self.pre = pre
        self.encoder = encoder
        self.reDataSetting(dataset)
        # model
        self.complile = complile
        self.compute_all_dist = compute_all_dist
        self.model = self.all_model()

    def reDataSetting(self, dataset):
        '''
        :param dataset: data helper 的输出数据
        :return:
        '''
        pre = self.pre
        self.train_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}train_edges'], dataset[f'{pre}train_short_path'])
        self.test_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}test_edges'], dataset[f'{pre}test_short_path'])
        self.dev_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}dev_edges'], dataset[f'{pre}dev_short_path'])

    def decoder(self, edges, embedding, manifold=Manifold(), **kwargs):
        '''
        解码器
        :param edges: 边二元组
        :param embedding: 编码器输出的特征矩阵
        :param manifold: 流形
        :return: Tensor, 边距离
        '''
        if self.compute_all_dist:
            em_all_dist = layers.Lambda(lambda x: manifold.sqdist(x, x, cross_join=True) ** 0.5,
                                        name=f'{self.sh}.D.dist_mat')(embedding)
            dist = layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]), name=f'{self.sh}.D.look_up')([em_all_dist, edges])
        else:
            # 层
            lookup_L = layers.Lambda(lambda x: tf.gather(x[0], x[1], axis=0), name=f'{self.sh}.D.look_up')
            dist_L = layers.Lambda(lambda x: tf.squeeze(manifold.sqdist(x[0], x[1]) ** 0.5, axis=-1),
                                   name=f'{self.sh}.D.dist')
            l_nodes = layers.Lambda(lambda x: x[..., 0], name=f'{self.sh}.D.left_nodes')
            r_nodes = layers.Lambda(lambda x: x[..., 1], name=f'{self.sh}.D.right_nodes')
            # 边距离计算
            edgesLeftNodes = lookup_L([embedding, l_nodes(edges)])
            edgesRightNodes = lookup_L([embedding, r_nodes(edges)])
            dist = dist_L([edgesLeftNodes, edgesRightNodes])
        return dist

    def loss_metrics(self, dist, short_path):
        # 损失
        loss_ = layers.Lambda(lambda x: tf.abs((x[0] / tf.cast(x[1], tf.float32)) ** 2 - 1),
                              name='Loss')([dist, short_path])
        loss = layers.Lambda(lambda x: tf.reduce_sum(x), name=f'{self.sh}.L.sum')(loss_)
        # 评估
        # argsort = layers.Lambda(lambda x: tf.concat([[tf.argsort(x[0]) + 1], [tf.argsort(x[1]) + 1]], 0),
        #                         name=f'{self.sh}.M.argsort')([dist, short_path])  # 索引排序后拼接
        # metric = layers.Lambda(lambda x: tf.reduce_mean(tf.reduce_min(x, 0) / tf.reduce_max(x, 0)),
        #                     name=f'{self.sh}.M.ap')(argsort)  # 平均精度, 这指标一直 0.5
        metric = layers.Lambda(lambda x: -tf.reduce_mean(x), name=f'{self.sh}.M.loss.mean')(loss_)
        return loss, metric

    def loss_name(self, latex=True):
        if latex:
            return '$\\mathcal{L}_{1}$'
        else:
            return 'L1'

    def metric_name(self):
        return 'negdistort'

    def all_model(self):
        '''
        整体模型
        :return:
        '''
        # 输入
        edges = layers.Input(batch_shape=(None, 2), name=f'{self.sh}.Input.edges', dtype=tf.int32)  # train/dev/test 边
        short_path = layers.Input(batch_shape=(None,), name=f'{self.sh}.Input.short_path',
                                  dtype=tf.int32)  # train/dev/test 最短路径
        # 编码器
        embedding = self.encoder.model.output
        # 解码器
        dist = self.decoder(edges, embedding, self.encoder.manifold)
        # 损失函数
        loss, metric = self.loss_metrics(dist, short_path)
        # 模型
        model = Model(inputs=self.encoder.model.inputs + [edges, short_path], outputs=[embedding, loss, metric],
                      name=f'{self.sh}.All_model')
        if self.complile:
            model.compile(optimizer=self._optimizer, loss=lambda y_true, y_pred: y_pred[1])
        return model


class Decoder_hypernymyRel(Decoder):
    def __init__(self, dataset, encoder: Encoder, pre='', complile=True, **kwargs):
        '''
        :param dataset: data helper 的输出数据
        :param encoder: Encoder, 编码器模型
        :param pre: str; 数据集名称的前缀, 一般用于混合数据
        :param complile: bool; 是否编译, 一般用于多损失函数结合, 直接训练不能为False
        '''
        super(Decoder_hypernymyRel, self).__init__(**kwargs)
        self.sh = 'HR'  # 类名的简写
        # dataset
        self.pre = pre
        self.encoder = encoder
        self.reDataSetting(dataset)
        # model
        self.complile = complile
        self.model = self.all_model()

    def reDataSetting(self, dataset):
        '''
        :param dataset: data helper 的输出数据
        :return:
        '''
        pre = self.pre
        self.train_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}train_edges_pos'], dataset[f'{pre}train_edges_neg'])
        self.test_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}test_edges_pos'], dataset[f'{pre}test_edges_neg'])
        self.dev_data = (
            *self.encoder.getData(dataset, pre), dataset[f'{pre}dev_edges_pos'], dataset[f'{pre}dev_edges_neg'])

    def decoder(self, edgesPos, edgesNeg, embedding, manifold=Manifold(), **kwargs):
        '''
        解码器
        :param edgesPos: 正例边二元组
        :param edgesNeg: 负例边二元组
        :param embedding: 编码器输出的特征矩阵
        :param manifold: 流形
        :return: (Tensor, Tensor), (正例发生概率, 负例发生概率|高一维度), ([-1], [-1,negNum])
        '''
        # 层
        lookup_L = layers.Lambda(lambda x: tf.gather(x[0], x[1], axis=0), name=f'{self.sh}.D.look_up')
        dist_L = layers.Lambda(lambda x: tf.squeeze(manifold.sqdist(x[0], x[1]) ** 0.5, axis=-1),
                               name=f'{self.sh}.D.dist')
        probs = layers.Lambda(lambda x: tf.exp(-x), name=f'{self.sh}.D.exp-dist')
        l_nodes = layers.Lambda(lambda x: x[..., 0], name=f'{self.sh}.D.left_nodes')
        r_nodes = layers.Lambda(lambda x: x[..., 1], name=f'{self.sh}.D.right_nodes')
        # 正例距离计算
        edgesPosLeftNodes = lookup_L([embedding, l_nodes(edgesPos)])
        edgesPosRightNodes = lookup_L([embedding, r_nodes(edgesPos)])
        pos_dist = dist_L([edgesPosLeftNodes, edgesPosRightNodes])
        # 负例距离计算
        edgesNegLeftNodes = lookup_L([embedding, l_nodes(edgesNeg)])
        edgesNegRightNodes = lookup_L([embedding, r_nodes(edgesNeg)])
        neg_dist = dist_L([edgesNegLeftNodes, edgesNegRightNodes])
        # 概率值计算
        pos_probs = probs(pos_dist)
        neg_probs = probs(neg_dist)
        return pos_probs, neg_probs

    def loss_metrics(self, pos_probs, neg_probs):
        '''
        :param pos_probs: [-1], Tensor
        :param neg_probs: [-1,negNum], Tensor
        :return:
        '''
        # 损失
        loss_div = layers.Lambda(
            lambda x: x[0] / (tf.abs(x[0] + tf.reduce_sum(x[1], -1)) + self.encoder.manifold.v_min),
            name=f'{self.sh}.L.div')([pos_probs, neg_probs])
        loss = layers.Lambda(lambda x: - tf.reduce_sum(tf.math.log(x)), name=f'{self.sh}.L.-sum_log')(loss_div)
        # 评估
        metric = layers.Lambda(lambda x: tf.cast(tf.greater(x[0], tf.reduce_max(x[1], -1)), tf.float32),
                               name=f'{self.sh}.M.acc')([pos_probs, neg_probs])
        metric = layers.Lambda(lambda x: tf.reduce_mean(x), name=f'{self.sh}.M.acc.mean')(metric)
        return loss, metric

    def loss_name(self, latex=True):
        if latex:
            return '$\\mathcal{L}_{2}$'
        else:
            return 'L2'

    def metric_name(self):
        return 'acc'

    def all_model(self):
        '''
        整体模型
        :return:
        '''
        # 输入
        edgesPos = layers.Input(batch_shape=(None, 2), name=f'{self.sh}.Input.edgesPos',
                                dtype=tf.int32)  # train/dev/test 正例边
        edgesNeg = layers.Input(batch_shape=(None, None, 2), name=f'{self.sh}.Input.edgesNeg',
                                dtype=tf.int32)  # train/dev/test 负例边
        # 编码器
        embedding = self.encoder.model.output
        # 解码器
        pos_probs, neg_probs = self.decoder(edgesPos, edgesNeg, embedding, self.encoder.manifold)
        # 损失函数
        loss, metric = self.loss_metrics(pos_probs, neg_probs)
        # 模型
        model = Model(inputs=self.encoder.model.inputs + [edgesPos, edgesNeg], outputs=[embedding, loss, metric],
                      name=f'{self.sh}.All_model')
        if self.complile:
            model.compile(optimizer=self._optimizer, loss=lambda y_true, y_pred: y_pred[1])
        return model


class Decoder_mixedLoss(Decoder):  # 多损失函数混合
    def __init__(self, dataset, encoder: Encoder, task_weight=None, **kwargs):
        '''
        :param dataset: MixedDataset.getDataset(); data helper 的输出数据
        :param encoder: Encoder, 编码器模型
        :param task_weight: {'data helper name':float,..} or None; 不同任务损失函数的权重, 要与 dataset 一致, 主任务由dataset确定
            None 默认权重为 [n-1,1,1,..]; n表示任务数量, 主任务权重占一半. 权重最后会归一化
        '''
        super(Decoder_mixedLoss, self).__init__(**kwargs)
        self.sh = 'ML'  # 类名的简写
        # 初始化 task_weight
        if task_weight is None or len(task_weight) == 0:
            task_weight = {i: 1 for i in dataset['all_dh_name']}
            task_weight[dataset['all_dh_name'][0]] = len(dataset['all_dh_name']) - 1
        s = sum(list(task_weight.values()))
        task_weight = {k: v / s for k, v in task_weight.items()}
        print('Decoder_mixedLoss.task_weight:', task_weight)
        # 初始化 dh_weight_L
        all_dh_name_L = dataset['all_dh_name']  # ['data helper name',..]; 第一个是主任务
        assert len(set(task_weight) - set(all_dh_name_L)) == 0, \
            f'dataset 和 task_weight 数据不匹配!\nall_dh_name_L: {all_dh_name_L};\ntask_weight: {task_weight}'
        self.dh_weight_L = [task_weight[i] for i in all_dh_name_L]  # [权重,..]; 与 dataset['all_dh_name'] 对应
        self.all_dh_name_L = all_dh_name_L
        # 构建 decoder
        self.decoder_L = []  # 第一个是主任务 decoder
        for i in all_dh_name_L:
            self.decoder_L.append(Decoder.dhToDecoder(i)[0](dataset, encoder, pre=f'{i}_', complile=False, **kwargs))
        # dataset
        self.encoder = encoder
        self.reDataSetting(dataset)
        # model
        self.model = self.all_model()

    def reDataSetting(self, dataset):
        '''
        :param dataset: data helper 的输出数据
        :return:
        '''
        encoderDataset = self.encoder.getData(dataset, pre=self.all_dh_name_L[0] + '_')
        self.train_data = [*encoderDataset]  # 需要保证所有兄弟类前面都是encoder的数据
        self.test_data = [*encoderDataset]
        self.dev_data = [*encoderDataset]
        for decoder in self.decoder_L:
            decoder.reDataSetting(dataset)
            self.train_data += list(decoder.train_data[len(encoderDataset):])
            self.test_data += list(decoder.test_data[len(encoderDataset):])
            self.dev_data += list(decoder.dev_data[len(encoderDataset):])

    def loss_name(self, latex=True):
        if latex:
            return self.decoder_L[0].loss_name(True) + '^{' + ''.join(
                [i.loss_name(True) for i in self.decoder_L[1:]]) + '}'
        else:
            return ''.join([i.loss_name(False) for i in self.decoder_L])

    def metric_name(self):
        return self.decoder_L[0].metric_name()

    def all_model(self):
        '''
        整体模型
        :return:
        '''
        # 输入
        encoderInputs = self.encoder.model.inputs
        modelInputs = [*encoderInputs]
        for decoder in self.decoder_L:
            modelInputs.append(decoder.model.inputs[len(encoderInputs):])
        # 编码器
        embedding = self.encoder.model.output
        # 损失函数
        loss_L = [i.model.outputs[1] for i in self.decoder_L]  # 所有损失函数
        loss = layers.Lambda(lambda x: tf.concat([x], 0), name=f'{self.sh}.L.concat')(loss_L)
        dh_weight = tf.constant(self.dh_weight_L)  # 损失函数权重转 tensor
        loss = layers.Lambda(lambda x: tf.reduce_sum(x * dh_weight), name=f'{self.sh}.L.weighted_sum')(loss)
        # 指标(主任务)
        metric = self.decoder_L[0].model.outputs[2]
        # 模型
        model = Model(inputs=modelInputs, outputs=[embedding, loss, metric], name=f'{self.sh}.All_model')
        model.compile(optimizer=self._optimizer, loss=lambda y_true, y_pred: y_pred[1])
        return model
