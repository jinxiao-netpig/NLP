import os
import math
import torch
import argparse
import torch.nn as nn
# from module import FVG
from module2 import PETER_MF, PETER_SCoR
from my_utils import rouge_score, bleu_score, DataLoader, Batchify, ids2tokens, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from utils.logger import Logger

###############################################################################
# 1. Set parameter
###############################################################################
parser = argparse.ArgumentParser(description='FVG')
# 输入（加载数据集）
parser.add_argument('--data_path', type=str, default='../Data/Amazon/MoviesAndTV/reviews.pickle',
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default='../Data/Amazon/MoviesAndTV/1',
                    help='load indexes')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
# 输出（保存实验结果、日志、模型）
parser.add_argument('--checkpoint', type=str, default='../Model/tmp',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--log_dir', type=str, default='../Logs/tmp',
                    help='path for saving the log file')
parser.add_argument('--log_name', type=str, default=None,
                    help='path for saving the log file')
# 构建模型
parser.add_argument('--emsize', type=int, default=512,
                    help='size of embeddings')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--rating_model', type=str, default='scor',
                    help='Recommendation Model')
parser.add_argument('--fixed_params', action='store_true',
                    help='fixed params of recommender')
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
parser.add_argument('--rating_reg', type=float, default=0.01,
                    help='regularization on recommendation task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
# 设备
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(),
                    help='use CUDA')
args = parser.parse_args()

logger = Logger(args.log_dir, args.log_name)
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
model_path      = os.path.join(args.checkpoint, 'model.pt')  # model 保存路径
prediction_path = os.path.join(args.checkpoint, args.outf)   # 预测结果输出路径

###############################################################################
# 2. Load Data
###############################################################################
logger.log_message('Loading Data')

corpus   = DataLoader(args.data_path, args.index_dir, args.vocab_size)  # 数据集对象
word2idx = corpus.word_dict.word2idx  # 词典
idx2word = corpus.word_dict.idx2word  # 词典
feature_set = corpus.feature_set

# 划分数据集（批量化）
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)  # 训练集
val_data   = Batchify(corpus.valid, word2idx, args.words, args.batch_size)                # 验证集
test_data  = Batchify(corpus.test,  word2idx, args.words, args.batch_size)                # 测试集

###############################################################################
# 3. Build the model
###############################################################################
logger.log_message('Build the model')

tgt_len = args.words + 1  # added <bos> or <eos>
src_len = 2
ntokens = len(corpus.word_dict)
nuser   = len(corpus.user_dict)
nitem   = len(corpus.item_dict)
pad_idx = word2idx['<pad>']  # 结束标志

# model = FVG(
#     tgt_len, pad_idx,
#     nuser, nitem,
#     ntokens, args.emsize, args.nhead, args.nlayers, dropout=args.dropout,
# ).to(device)
# logger.log_message('Model: FVG-PMF')

if args.rating_model == 'scor':
    model = PETER_SCoR(
        tgt_len, pad_idx,
        nuser, nitem, ntokens,
        args.emsize, args.nhead, args.nlayers, args.dropout,
    ).to(device)
    if args.fixed_params:
        logger.log_message('Model: PETER-SCoR (Fixed Params)')
    else:
        logger.log_message('Model: PETER-SCoR')
else:
    model = PETER_MF(
        tgt_len, pad_idx,
        nuser, nitem, ntokens,
        args.emsize, args.nhead, args.nlayers, args.dropout,
    ).to(device)
    if args.fixed_params:
        logger.log_message('Model: PETER-MF (Fixed Params)')
    else:
        logger.log_message('Model: PETER-MF')

###############################################################################
# 4. Training code (rating)
###############################################################################
# r_params    = model.get_rating_params()                                              # 参数
if args.rating_model == 'scor':
    r_optimizer = torch.optim.SGD([
        {'params': model.user_embeddings.parameters()},
        {'params': model.item_embeddings.parameters()},
        {'params': model.hidden2rating.parameters()},
    ], lr=args.lr)                                  # 优化器
else:
    r_optimizer = torch.optim.SGD([
        {'params': model.user_embeddings.parameters()},
        {'params': model.item_embeddings.parameters()},
    ], lr=args.lr)  # 优化器
r_scheduler = torch.optim.lr_scheduler.StepLR(r_optimizer, step_size=1, gamma=0.25)  # 学习率调度器
r_criterion = nn.MSELoss()                                                           # 损失函数-评分预测

t_criterion = nn.NLLLoss(ignore_index=pad_idx)                                     # 损失函数-文本预测# 损失函数-评分预测
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)                        # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.25)  # 学习率调度器

def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rating_loss  = 0.
    text_loss    = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq, _, _ = data.next_batch()  # (batch_size, seq_len), Data.step += 1
            batch_size = user.size(0)
            user   = user.to(device)    # (batch_size,)
            item   = item.to(device)
            rating = rating.to(device)
            seq    = seq.to(device)     # (batch_size, tgt_len + 1)
            # model.predict_seq(user, item, seq[:, :-1])
            rating_p, log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, tgt_len, ntoken)

            r_loss = r_criterion(rating_p, rating)
            t_loss = t_criterion(
                log_word_prob.view(-1, ntokens),  # (batch_size * tgt_len, ntoken)
                seq[:, 1:].reshape((-1,))         # (batch_size * tgt_len)
            )

            text_loss    += batch_size * t_loss.item()
            rating_loss  += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return  rating_loss / total_sample, text_loss / total_sample


def r_train(data):
    model.train()  # Turn on training mode which enables dropout.
    rating_loss  = 0.
    total_sample = 0
    while True:
        user, item, rating, _, _, _ = data.next_batch()  # (batch_size, seq_len), Data.step += 1
        batch_size = user.size(0)   # 获取批次大小
        user   = user.to(device)    # (batch_size,)
        item   = item.to(device)
        rating = rating.to(device)

        r_optimizer.zero_grad()     # 清除（重置）优化器的梯度，以确保梯度不会在多个迭代中累积。
        rating_p, _ = model(user, item, _, seq_pred=False)  # (batch_size,) vs. (batch_size, tgt_len, ntoken)

        r_loss = r_criterion(rating_p, rating)
        r_loss.backward()  # 反向传播

        # 梯度裁剪：通过限制梯度到一个特定范围，来防止梯度爆炸问题。
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        r_optimizer.step()  # 更新模型参数

        rating_loss  += batch_size * r_loss.item()
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


# train model
best_val_loss = float('inf')             # 初始化为无穷大
endure_count = 0                         # 用于记录多少个周期损失没有改善
for epoch in range(1, args.epochs + 1):  # 循环遍历每个训练周期（epoch）
    logger.log_message('epoch {}'.format(epoch))
    r_train(train_data)                            # 训练模型
    val_r_loss, _ = evaluate(val_data)  # 评估模型/计算损失（上下文、解释、评分）
    val_loss = val_r_loss
    logger.log_message('rating loss {:4.4f} on validation'.format(
        val_r_loss))

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
        r_scheduler.step()
        logger.log_message('Learning rate set to {:2.8f}'.format(r_scheduler.get_last_lr()[0]))

if args.fixed_params:
    model.set_parameter_requires_grad(rating=False)

###############################################################################
# 5. Training code (text)
###############################################################################
def train(data):
    model.train()  # Turn on training mode which enables dropout.
    rating_loss  = 0.
    text_loss    = 0.
    total_sample = 0
    while True:
        user, item, rating, seq, _, _ = data.next_batch()  # (batch_size, seq_len), Data.step += 1
        batch_size = user.size(0)   # 获取批次大小
        user   = user.to(device)    # (batch_size,)
        item   = item.to(device)
        rating = rating.to(device)
        seq    = seq.to(device)

        optimizer.zero_grad()  # 清除（重置）优化器的梯度，以确保梯度不会在多个迭代中累积。
        rating_p, log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, tgt_len, ntoken)

        r_loss = r_criterion(rating_p, rating)
        t_loss = t_criterion(
            log_word_prob.view(-1, ntokens),  # (batch_size * tgt_len, ntoken)
            seq[:, 1:].reshape((-1,))         # (batch_size * tgt_len)
        )
        loss = args.rating_reg * r_loss + args.text_reg * t_loss
        loss.backward()  # 反向传播

        # 梯度裁剪：通过限制梯度到一个特定范围，来防止梯度爆炸问题。
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()  # 更新模型参数

        text_loss    += batch_size * t_loss.item()
        rating_loss  += batch_size * r_loss.item()
        total_sample += batch_size

        # 打印损失
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss   / total_sample
            cur_r_loss = rating_loss / total_sample
            logger.log_message('rating loss {:4.4f} | text ppl {:4.4f} |  {:5d}/{:5d} batches'.format(
                cur_r_loss, math.exp(cur_t_loss), data.step, data.total_step))
            text_loss    = 0.
            rating_loss  = 0.
            total_sample = 0

        if data.step == data.total_step:
            break


# train model
best_val_loss = float('inf')             # 初始化为无穷大
endure_count = 0                         # 用于记录多少个周期损失没有改善
for epoch in range(1, args.epochs + 1):  # 循环遍历每个训练周期（epoch）
    logger.log_message('epoch {}'.format(epoch))
    train(train_data)                            # 训练模型
    val_r_loss, val_t_loss = evaluate(val_data)  # 评估模型/计算损失（上下文、解释、评分）
    val_loss = args.rating_reg * val_r_loss + args.text_reg * val_t_loss
    logger.log_message('rating loss {:4.4f} | text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(
        val_r_loss, math.exp(val_t_loss), val_loss))

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
# Load the best saved model.

def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rating_predict = []
    idss_predict   = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, _, _ = data.next_batch()
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, 0].unsqueeze(1).to(device)  # (batch_size, 1)
            for idx in range(args.words):
                # produce a word at each step
                if idx == 0:
                    rating_p, log_word_prob = model(user, item, text)  # (batch_size,) vs. (batch_size, 1, ntoken)
                    rating_predict.extend(rating_p.tolist())
                else:
                    log_word_prob = model.predict_seq(user, item, text)  # (batch_size, text_len, ntoken)
                word_prob = log_word_prob.exp()             # (batch_size, text_len, ntoken)
                word_idx = torch.argmax(word_prob, dim=-1)  # (batch_size, text_len), pick the one with the largest probability
                text = torch.cat(tensors=(text, word_idx[:, idx].unsqueeze(1)), dim=1)  # (batch_size, len++)
            idss_predict.extend(text.tolist())

            if data.step == data.total_step:
                break

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.log_message('RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.log_message('MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    logger.log_message('BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    logger.log_message('BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    logger.log_message('USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    logger.log_message('DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    logger.log_message('FCR {:7.4f}'.format(FCR))
    feature_test = [idx2word[i] for i in data.feature.squeeze(1).tolist()]  # ids to words
    FMR = feature_matching_ratio(feature_batch, feature_test)
    logger.log_message('FMR {:7.4f}'.format(FMR))
    text_test      = [' '.join(tokens) for tokens in tokens_test]
    text_predict   = [' '.join(tokens) for tokens in tokens_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        logger.log_message('{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, fake) in zip(text_test, text_predict):
        text_out += '{}\n{}\n\n'.format(real, fake)
    return text_out


with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test Data.
test_r_loss, test_t_loss = evaluate(test_data)
logger.log_message('=' * 89)
logger.log_message('rating loss {:4.4f} on test | text ppl {:4.4f} | End of training'.format(
    test_r_loss, math.exp(test_t_loss)))

logger.log_message('Generating text')
text_o = generate(test_data)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
logger.log_message('Generated text saved to ({})'.format(prediction_path))
