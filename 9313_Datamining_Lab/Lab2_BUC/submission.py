## import modules here 
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################

def st_opt(input, output, pre):
    formar = [[input.iat[0, 0]], ["ALL"]]
    for i in range(1, input.shape[1]-1):
        latter = []
        for p in range(0, len(formar)):
            latter.append(formar[p] + [input.iat[0, i]])
        for q in range(0, len(formar)):
            latter.append(formar[q] + ["ALL"])
        formar = latter
    for x in formar:
        x.append(input.iat[0, -1])
        new = pre + x
        output.loc[len(output)] = new


def buc(df, df_out, pre=[]):
    dims = df.shape[1]
    if dims == 1:  # one column left
        all = [] # sum result on column
        for x in helper.project_data(df, 0):
            all.append(int(x))
        pre.append(sum(all))
        df_out.loc[len(df_out)] = pre
        pre.pop()
        return 0
    elif df.shape[0] == 1:
        st_opt(df, df_out, pre)
    else:
        dim0_vals = set(helper.project_data(df, 0).values)
        for dim0_v in sorted(dim0_vals):
            sub_data = helper.slice_data_dim0(df, dim0_v)
            pre.append(dim0_v)
            buc(sub_data, df_out, pre)
            pre.pop()
        sub_data = helper.remove_first_dim(df)
        pre.append("ALL")
        buc(sub_data, df_out, pre)
        pre.pop()
    return 0


def buc_rec_optimized(df):# do not change the heading of the function
    final_result = []                                                           # create an empty list for a new dataframe
    df_out = pd.DataFrame(final_result, columns=df.columns.values.tolist())     # create an empty dataframe to store result
    buc(df, df_out)                                                             # updata output dataframe along with the buc recurision function
    # print(df_out)
    return df_out

# for test
# a = [['0','1','2','3','4','100'],['0','2','5','4','7','200'],['0','2','5','6','8','300']]
# df = pd.DataFrame(a, columns=['A','B','C','D','E','M'])
# print(buc_rec_optimized(df))








