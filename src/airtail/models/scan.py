import torch


# 这份文件为 selective scan 提供两种执行后端：
# 1. reference：完全使用 PyTorch 编写，便于阅读、调试和小张量验证；
# 2. cuda：调用已经编译好的 selective_scan_cuda 扩展，提高实际运行速度。
#
# 两种后端遵循相同的输入、输出约定。这里的 batch 不一定等于数据加载器中的
# 原始批次大小 B。在上游 quantile_ssm.py 中，站点、分位区域和时间尺度等维度
# 会被展平到 scan 的批次维，因此这里的 batch 更准确地说是“扫描实例数”。


def reference_selective_scan(u, delta, A, B, C, D):
    """Small-tensor reference scan with CUDA-kernel-compatible contracts."""

    # u 是进入状态空间模型的输入序列，形状为：
    # [batch, channels, length]
    #
    # delta 是每个扫描实例、每个隐藏通道、每个时间步对应的离散时间步长，
    # 必须与 u 具有完全相同的形状。
    #
    # 注意：这个函数不会在内部对 delta 做 Sigmoid、Softplus 或上下界裁剪。
    # QC-MSSM 中 delta 的有界性由上游 quantile_ssm.py 的 _delta() 保证。
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must have shape [batch, channels, length]")

    # batch：扫描实例数
    # channels：隐藏通道数，即上游的 d_model
    # length：历史时间序列长度 L
    batch, channels, length = u.shape

    # A 是连续时间状态矩阵的对角参数，形状为：
    # [channels, state]
    #
    # 每个隐藏通道有一组独立的 state 个衰减系数。
    # 在上游 QuantileConditionedMultiscaleSSM 中，
    # A = -exp(A_log)，因此 A 的所有元素都严格小于 0。
    if A.ndim != 2 or A.shape[0] != channels:
        raise ValueError("A must have shape [channels, state]")

    # state_size 对应上游配置中的 d_state。
    state_size = A.shape[1]

    # B 是随输入动态生成的状态写入系数，形状为：
    # [batch, state, length]
    #
    # B 没有显式的 channels 维。计算时会在该位置增加一个长度为 1 的维度，
    # 然后广播到所有隐藏通道。
    if B.shape != (batch, state_size, length):
        raise ValueError("B must have shape [batch, state, length]")

    # C 是随输入动态生成的状态读取系数，形状同样为：
    # [batch, state, length]
    #
    # 它决定每个时间步如何把 state 维的内部状态投影为各隐藏通道的输出。
    if C.shape != (batch, state_size, length):
        raise ValueError("C must have shape [batch, state, length]")

    # D 是跳跃连接（skip connection）系数，每个隐藏通道一个值。
    # 它让当前输入 u_t 可以不经过递归状态，直接参与当前输出 y_t。
    if D.shape != (channels,):
        raise ValueError("D must have shape [channels]")

    # 所有扫描实例的初始隐状态均设为 0：
    # [batch, channels, state]
    #
    # new_zeros 会继承 u 的 device 和 dtype，避免 CPU/GPU 或精度不一致。
    state = u.new_zeros(batch, channels, state_size)

    # 逐时间步保存输出，循环结束后再沿时间维堆叠。
    outputs = []

    # 按 step = 0, 1, ..., length-1 从历史序列前端向后递推。
    # 当前 step 只读取上一步 state 和当前时刻输入，因此这里不访问未来时间步。
    for step in range(length):

        # 取当前时间步的离散步长：
        # [batch, channels]
        dt = delta[:, :, step]

        # 将连续时间衰减项 A 离散化：
        #
        # transition = exp(delta_t * A)
        #
        # dt.unsqueeze(-1)：[batch, channels, 1]
        # A.unsqueeze(0)：  [1, channels, state]
        # 广播后 transition：[batch, channels, state]
        #
        # 当上游保证 delta > 0 且 A < 0 时，transition 位于 (0, 1)，
        # 表示旧状态随时间衰减，而不是无界放大。
        transition = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))

        # 计算当前输入写入内部状态的驱动项：
        #
        # drive = delta_t * B_t * u_t
        #
        # dt.unsqueeze(-1)：       [batch, channels, 1]
        # B[:, None, :, step]：    [batch, 1, state]
        # u[:, :, step, None]：    [batch, channels, 1]
        # 广播后 drive：           [batch, channels, state]
        drive = dt.unsqueeze(-1) * B[:, None, :, step] * u[:, :, step, None]

        # selective scan 的核心状态递推：
        #
        # h_t = exp(delta_t A) ⊙ h_{t-1} + delta_t B_t u_t
        #
        # 第一项保留并衰减历史状态，第二项写入当前输入。
        state = transition * state + drive

        # 从当前状态生成输出：
        #
        # y_t = C_t h_t + D ⊙ u_t
        #
        # C[:, None, :, step] 的形状为 [batch, 1, state]，
        # 与 state 相乘后在 state 维求和，得到 [batch, channels]。
        # D * u_t 是逐隐藏通道的直接输入支路。
        y = (state * C[:, None, :, step]).sum(dim=-1) + D * u[:, :, step]

        # 保存当前时间步输出 [batch, channels]。
        outputs.append(y)

    # length 个 [batch, channels] 沿最后一维堆叠，恢复为：
    # [batch, channels, length]
    #
    # 输出布局与输入 u 保持一致。
    return torch.stack(outputs, dim=-1)


# torch.autograd.Function 用来手工定义 CUDA 扩展的前向与反向过程。
# Python/PyTorch 本身无法自动追踪外部 CUDA 函数内部的运算，因此需要显式说明：
# - forward() 如何调用 CUDA 前向；
# - backward() 如何根据上游梯度计算六个输入张量的梯度。
class _CudaSelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):

        # 仅在用户真正选择 cuda 后端时才导入编译扩展。
        # 因此使用 reference 后端时，即使环境未安装该扩展，也能正常运行。
        try:
            import selective_scan_cuda
        except ImportError as error:
            raise RuntimeError("cuda scan backend requested but selective_scan_cuda is unavailable") from error

        # CUDA 扩展要求所有参与 scan 的张量都位于 GPU。
        # 这里只做明确报错，不会自动把 CPU 张量搬到 GPU。
        if not all(tensor.is_cuda for tensor in (u, delta, A, B, C, D)):
            raise RuntimeError("cuda scan backend requires every scan tensor on CUDA")

        # CUDA 内核通常要求输入在内存中连续排列。
        # contiguous() 不改变数值和逻辑形状，只在必要时重新整理底层内存布局。
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()

        # Python 参考实现接收的 B/C 形状为：
        # [batch, state, length]
        #
        # selective_scan_cuda 接口允许按 group 提供动态 B/C，所需布局为：
        # [batch, groups, state, length]
        #
        # 这里增加长度为 1 的 group 维：
        # [batch, state, length] → [batch, 1, state, length]
        #
        # 这就是“适配 scan 调用”的一个具体位置。它没有改变 B/C 的数值，
        # 只是把项目上游的三维接口转换为 CUDA 内核要求的四维接口。
        B = B[:, None].contiguous()
        C = C[:, None].contiguous()
        D = D.contiguous()

        # 调用已经编译好的 CUDA selective scan 前向函数。
        #
        # 这里最后三个实参依次表示：
        # - None：不使用额外的 z 门控输入；
        # - None：不在内核内部添加 delta_bias；
        # - False：不在内核内部对 delta 执行 Softplus。
        #
        # 原因是 QC-MSSM 已在上游显式生成并约束 delta，
        # 这里应直接使用该 delta，避免再次变换其数值与上下界语义。
        #
        # out 是最终序列输出；scan_state 是反向传播所需的中间状态；
        # *rest 兼容 CUDA 扩展可能返回的其他结果，但当前代码不使用它们。
        out, scan_state, *rest = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, None, False
        )

        # 把已导入的 CUDA 模块暂存在 ctx 中，供 backward() 调用其 bwd 接口。
        ctx.scan_module = selective_scan_cuda

        # 保存反向传播计算梯度所需的输入和中间状态。
        # PyTorch 会管理这些张量的生命周期。
        ctx.save_for_backward(u, delta, A, B, C, D, scan_state)

        # 返回 CUDA 内核计算的 [batch, channels, length] 输出。
        return out

    @staticmethod
    def backward(ctx, dout):

        # 取回 forward() 中保存的张量。
        u, delta, A, B, C, D, scan_state = ctx.saved_tensors

        # dout 是损失函数相对于 scan 输出的上游梯度。
        # CUDA 反向内核同样要求其内存布局连续。
        dout = dout.contiguous()

        # 调用 CUDA 扩展提供的反向函数，计算：
        # du、ddelta、dA、dB、dC、dD。
        #
        # 参数中的 None 对应前向未启用的 z、delta_bias 等可选分支；
        # False 表示 delta_softplus=False；
        # 最后的 False 表示不要求反向过程额外重算带 z 的输出。
        du, ddelta, dA, dB, dC, dD, *_ = ctx.scan_module.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            None,
            None,
            dout,
            scan_state,
            None,
            None,
            False,
            False,
        )

        # autograd.Function.backward() 返回的梯度数量和顺序必须与
        # forward(ctx, u, delta, A, B, C, D) 的六个张量输入完全一致。
        #
        # CUDA 内核返回的 dB/dC 仍包含前向适配时加入的 group 维：
        # [batch, 1, state, length]
        #
        # squeeze(1) 将其还原为上游期望的：
        # [batch, state, length]
        return du, ddelta, dA, dB.squeeze(1), dC.squeeze(1), dD


# 对自定义 autograd.Function 的普通函数封装。
# 必须通过 .apply(...) 调用，PyTorch 才会把 _CudaSelectiveScan 作为
# 一个可求导节点接入计算图，并在反向传播时自动调用上面的 backward()。
def cuda_selective_scan(u, delta, A, B, C, D):
    return _CudaSelectiveScan.apply(u, delta, A, B, C, D)


# 统一的后端分发入口。quantile_ssm.py 不需要了解两种后端内部的实现差异，
# 只需传入相同的 u、delta、A、B、C、D，并通过 backend 选择执行路径。
#
# 需要特别注意：函数名中的 bounded 表示它服务于项目的“有界 delta scan”
# 设计，但这个函数本身不会检查或裁剪 delta。真正的上下界映射在上游完成。
def bounded_selective_scan(u, delta, A, B, C, D, backend):

    # PyTorch 参考实现：可在 CPU 或 CUDA 张量上运行，适合小张量核对。
    if backend == "reference":
        return reference_selective_scan(u, delta, A, B, C, D)

    # 编译 CUDA 扩展：用于更高效的正式计算。
    if backend == "cuda":
        return cuda_selective_scan(u, delta, A, B, C, D)

    # 不静默回退到其他后端，避免用户以为正在运行 CUDA，
    # 实际却在较慢的 reference 实现上执行。
    raise ValueError(f"unknown selective scan backend: {backend}")
