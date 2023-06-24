import torch
scheduler = DDIMScheduler.from_pretrained(pipeline_name)
scheduler.set_timesteps(num_inference_steps=40)
# 定义需要进行梯度计算的 tensor
x = torch.randn(3, 3, requires_grad=True)
w = torch.randn(3, 3, requires_grad=True)

# 计算 x 和 w 的点积
y = torch.matmul(x, w)

# 对 y 进行一些操作，但不需要计算梯度
z = y.detach().requires_grad_()

# 计算 z 的平均值，并反向传播梯度
loss = z.mean()
loss.backward()

# 冻结 w 的梯度，只更新 x 的梯度
w.requires_grad_(False)
x.grad.zero_()
x2 = torch.randn(3, 3, requires_grad=True)
y2 = torch.matmul(x2, w)
loss2 = y2.mean()
loss2.backward()

# 打印梯度信息
print("x 的梯度：", x.grad)
print("w 的梯度：", w.grad)
print("x2 的梯度：", x2.grad)
