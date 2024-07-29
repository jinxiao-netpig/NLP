import os
import torch
import argparse
import torch.nn as nn
from module import PMF, SCoR, SCoR2
from my_utils import DataLoader, Batchify, mean_absolute_error, root_mean_square_error
from utils.logger import Logger

###############################################################################
# 1. Set parameter
###############################################################################
parser = argparse.ArgumentParser(description='MF & SCoR')
# 输入（加载数据集）
parser.add_argument('--data_path', type=str, default='../Data/TripAdvisor/reviews.pickle',
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default='../Data/TripAdvisor/1',
                    help='load indexes')
# 输出（保存实验结果、日志、模型）
parser.add_argument('--checkpoint', type=str, default='../Model/mf',
                    help='directory to save the final model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--log_dir', type=str, default='../Logs/tmp',
                    help='path for saving the log file')
parser.add_argument('--log_name', type=str, default=None,
                    help='path for saving the log file')
# 构建模型
parser.add_argument('--model', type=str, default='scor',
                    help='model to use')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of embeddings')
# 训练模型
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
# 设备
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(),
                    help='use CUDA')
args = parser.parse_args()

logger = Logger(log_dir=args.log_dir, log_name=args.log_name)
# 打印全部参数
logger.log_message('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    logger.log_message('{:40} {}'.format(arg, getattr(args, arg)))
logger.log_message('-' * 40 + 'ARGUMENTS' + '-' * 40)

# 检查参数
if args.data_path is None:
    parser.error('--data_path should be provided for loading Data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading Data splits')

# 设置参数
torch.manual_seed(args.seed)  # 设置随机种子以确保可重复性
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')  # model 保存路径

###############################################################################
# 2. Load Data
###############################################################################
logger.log_message('Loading Data')

corpus = DataLoader(args.data_path, args.index_dir)  # 数据集对象
nuser  = len(corpus.user_dict)
nitem  = len(corpus.item_dict)

# 划分数据集（批量化）
train_data = Batchify(corpus.train, args.batch_size, shuffle=True)  # 训练集
val_data   = Batchify(corpus.valid, args.batch_size)                # 验证集
test_data  = Batchify(corpus.test,  args.batch_size)                # 测试集

###############################################################################
# 3. Build the Model
###############################################################################
logger.log_message('Builde the Model')
# 创建模型实例
if args.model.lower() == 'scor':
    model = SCoR(nuser, nitem, args.emsize).to(device)
    logger.log_message('Model: SCoR')
elif args.model.lower() == 'mf+scor':
    model = SCoR2(nuser, nitem, args.emsize).to(device)
    logger.log_message('Model: MF+SCoR')
else:
    model = PMF(nuser, nitem, args.emsize).to(device)
    logger.log_message('Model: PMF')

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.25)

###############################################################################
# 4. Training code
###############################################################################
def train(data):
    model.train()  # Turn on training mode which enables dropout.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating = data.next_batch()  # (batch_size), Data.step += 1
        model.predict(user, item)

        batch_size = user.size(0)               # 获取批次大小
        user   = user.to(device)                # (batch_size,)
        item   = item.to(device)
        rating = rating.to(device)

        optimizer.zero_grad()         # 清除（重置）优化器的梯度，以确保梯度不会在多个迭代中累积。
        rating_p = model(user, item)  # (batch_size,)
        loss   = loss_fn(rating_p, rating)
        loss.backward()  # 反向传播

        # 梯度裁剪：通过限制梯度到一个特定范围，来防止梯度爆炸问题。
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()  # 更新模型参数

        rating_loss  += batch_size * loss.item()
        total_sample += batch_size

        # 打印损失
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_r_loss = rating_loss / total_sample
            logger.log_message('rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                 cur_r_loss, data.step, data.total_step))
            rating_loss  = 0.
            total_sample = 0

        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rating_loss  = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating = data.next_batch()  # (batch_size), Data.step += 1
            batch_size = user.size(0)
            user   = user.to(device)  # (batch_size,)
            item   = item.to(device)
            rating = rating.to(device)

            rating_p = model(user, item)  # (batch_size,)
            r_loss = loss_fn(rating_p, rating)

            rating_loss  += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return rating_loss / total_sample


# train model
best_val_loss = float('inf')             # 初始化为无穷大
endure_count = 0                         # 用于记录多少个周期损失没有改善
for epoch in range(1, args.epochs + 1):  # 循环遍历每个训练周期（epoch）
    logger.log_message('epoch {}'.format(epoch))
    train(train_data)                                        # 训练模型
    val_loss = evaluate(val_data)  # 评估模型/计算损失（上下文、解释、评分）
    logger.log_message('valid rating loss {:4.4f} on validation'.format(val_loss))

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        logger.log_message('Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:  # 达到阈值，停止训练
            logger.log_message('Cannot endure it anymore | Exiting from early stop')
            break
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        scheduler.step()
        logger.log_message('Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))



###############################################################################
# 5. Test model
###############################################################################

def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rating_predict  = []
    with torch.no_grad():
        while True:
            user, item, rating = data.next_batch()
            user    = user.to(device)  # (batch_size,)
            item    = item.to(device)

            rating_p = model(user, item)  # (batch_size,)
            rating_predict.extend(rating_p.tolist())

            if data.step == data.total_step:
                break

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.log_message('RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.log_message('MAE {:7.4f}'.format(MAE))


# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test Data.
test_r_loss = evaluate(test_data)
logger.log_message('=' * 89)
logger.log_message('rating loss {:4.4f} on test | End of training'.format(test_r_loss))
generate(test_data)
