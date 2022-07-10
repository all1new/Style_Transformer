from PIL import Image

'''用来加载数据:
    1. 首先根据文件地址打开文件,并将图片数据转化成RGB的格式;
    2. 如果size存在,则修改文件的大小;
    3. 如果scale存在就按照scale比例缩放文件的大小;
    4. 返回处理过的图片;
'''
def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.reszie((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

'''用来保存图片:
    1.将数据图像克隆并转化为numpy()格式,每个像素转化到0-255之间的值;
    2.然后将通道数进行变换,格式类型转化为uint8的类型;
    3.将数据转化为Image格式,最后保存到filename地址下.
'''
def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

'''格拉姆矩阵:主要应用在风格迁移
    1.主要做向量内积,对两个向量执行内积运算,就是对这两个向量对应位一一相乘之后求和的操作,内积的结果是一个标量。
    Gram矩阵是两两向量的内积组成,所以Gram矩阵可以反映出该组向量中各个向量之间的某种关系;
    2.格拉姆矩阵用于度量各个维度自己的特性以及各个维度之间的关系。
    内积之后得到的多尺度矩阵中,对角线元素提供了不同特征图各自的信息,其余元素提供了不同特征图之间的相关信息。
    这样一个矩阵,既能体现出有哪些特征,又能体现出不同特征间的紧密程度。
    具体流程:
    ① 首先获取各个维度的值;
    ② 然后重置特征尺寸:[b, c, w, h] -> [b, c, w * h];
    ③ 获取feature的转置矩阵feature_t->[b, w * h, c];
    ④ 将feature与feature_t相乘获取gram矩阵.
'''
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std