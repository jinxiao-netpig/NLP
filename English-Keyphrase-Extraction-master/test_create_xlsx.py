import pandas as pd

if __name__ == '__main__':
    # 创建数据
    data = {
        ('', '算法名称'): "tf-idf",
        ('Inspec', 'P'): 6,
        ('Inspec', 'R'): 7,
        ('Inspec', 'F1'): 8
    }

    # 创建具有多级列索引的 DataFrame
    df = pd.DataFrame(data, index=[2])

    # 设置列索引
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # 保存到 Excel 文件
    df.to_excel('example.xlsx', index=True, header=True)
