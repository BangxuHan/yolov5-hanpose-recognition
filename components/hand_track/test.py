import torch


a1 = torch.Tensor([[11.42280, -4.68948,  1.16724, -5.80339,  0.33294,  2.90298, -4.76893, -2.89002,  1.84688, -3.91473,
                    2.48294,  4.13128, -1.14426, -1.14264]])


def normalization(outlist):
    out = outlist[0]
    out1 = torch.softmax(out, 0)
    print(out1)
    # print(out.max(), out.min())
    # # fin = (outlist - out.min())/(out.max() - out.min())
    # # print(fin)
    #
    # x = torch.clamp(out, min=0)
    # print(x)
    # fin = (x - x.min())/(x.max() - x.min())
    # print(fin)


print(normalization(a1))

