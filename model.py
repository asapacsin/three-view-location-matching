import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim = 2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x




class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim = 512)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool=='gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, circle=False,):
        super(two_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle

        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)
        if VGG16:
            self.classifier = ClassBlock(512, class_num, droprate, return_f = circle)
            if pool =='avg+max':
                self.classifier = ClassBlock(1024, class_num, droprate, return_f = circle)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2

class ft_net_nas(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_nas, self).__init__()
        model_ft = models.efficientnet_b0(pretrained=True)
        
        # Adjust stride in the initial conv_stem
        if stride == 1:
            model_ft.features[0][0].stride = (1, 1)  # Default is (2, 2)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=1280)  # EfficientNet-B0 outputs 1280 channels

        self.model = model_ft
        # Map EfficientNet’s 1280 channels to 2048 to match ft_net and ClassBlock
        self.channel_mapper = nn.Linear(1280, 2048) if pool != 'avg+max' else nn.Linear(2560, 4096)
        self.channel_mapper.apply(weights_init_kaiming)

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        # print(f"ft_net_nas input shape: {x.shape}")
        
        # EfficientNet features (conv_stem, MBConv blocks)
        x = self.model.features(x)
        # print(f"After features: {x.shape}")  # Expect (batch_size, 1280, h, w)

        # Pooling
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            # print(f"avgpool2 shape: {x1.shape}, maxpool2 shape: {x2.shape}")
            x = torch.cat((x1, x2), dim=1)
            # print(f"After concat: {x.shape}")  # (batch_size, 2560, 1, 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            # print(f"After avgpool2: {x.shape}")  # (batch_size, 1280, 1, 1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            # print(f"After maxpool2: {x.shape}")  # (batch_size, 1280, 1, 1)
        elif self.pool == 'gem':
            x = self.model.gem2(x)
            # print(f"After gem2: {x.shape}")  # (batch_size, 1280)

        x = x.view(x.size(0), -1)
        # print(f"After view: {x.shape}")  # (batch_size, 1280) or (batch_size, 2560) with avg+max
        
        # Map to 2048 (or 4096 for avg+max)
        x = self.channel_mapper(x)
        # print(f"After channel_mapper: {x.shape}")  # (batch_size, 2048) or (batch_size, 4096)
        
        # print(f"ft_net_nas output shape: {x.shape}")
        return x

class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, circle=False,nas=False, swin=False):
        super(three_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        elif nas:
            self.model_1 =  ft_net_nas(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_nas(class_num, stride = stride, pool = pool)
        elif swin:
            self.model_1 =  ft_net_swin(class_num)
            self.model_2 =  ft_net_swin(class_num)
        else:
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            elif nas:
                self.model_3 =  ft_net_nas(class_num, stride = stride, pool = pool)
            elif swin:
                self.model_1 =  ft_net_swin(class_num)
                self.model_2 =  ft_net_swin(class_num)
            else:
                self.model_3 =  ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle

        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        if pool =='avg+max':

            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            x1 = x1.view(x1.size(0), x1.size(1))
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            x2 = x2.view(x2.size(0), x2.size(1))
            y2 = self.classifier(x2)

        if x3 is None:
            y3 = None
        else:
            x3 = self.model_3(x3)
            x3 = x3.view(x3.size(0), x3.size(1))
            y3 = self.classifier(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            x4 = x4.view(x4.size(0), x4.size(1))
            y4 = self.classifier(x4)
            return y1, y2, y3, y4

class ft_net_LPN(nn.Module):
    def __init__(self, num_classes=1000):
        super(ft_net_LPN, self).__init__()
        # Backbone: Simple CNN
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 56 * 56, num_classes)  # Adjust input size based on your data
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
        
class ft_net_swin(nn.Module):
    def __init__(self, num_classes=2048, pool='none'):
        super(ft_net_swin, self).__init__()
        self.backbone = models.swin_t(weights='IMAGENET1K_V1')
        in_features = self.backbone.head.in_features  # 768 for Swin-Tiny
        self.backbone.head = nn.Linear(in_features, 2048)  # Hardcode to 2048
        self.pool = pool
        if pool == 'gem':
            self.gem_pool = GeM(dim=2048)  # Match output dimension
    
    def forward(self, x):
        x = self.backbone(x)  # Shape: (batch_size, 2048)
        if self.pool == 'gem':
            x = x.view(x.size(0), x.size(1), 1, 1)  # Shape: (batch_size, 2048, 1, 1)
            x = self.gem_pool(x)  # Shape: (batch_size, 2048)
        return x
    
class GeM(nn.Module):
    def __init__(self, dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.dim = dim
        self.eps = eps
    
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        x = x.clamp(min=eps).pow(p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x
    
class ft_net_dense(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_dense, self).__init__()
        # Load pretrained DenseNet (using DenseNet121 as an example)
        model_ft = models.densenet121(pretrained=True)
        
        # Modify stride if needed (DenseNet uses stride in initial conv and pooling layers)
        if stride == 1:
            model_ft.features.conv0.stride = (1, 1)  # Initial conv stride
            model_ft.features.pool0 = nn.Identity()  # Remove initial maxpool

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=1024)  # DenseNet121 output channels before classifier

        # Remove the original classifier
        model_ft.classifier = nn.Identity()
        
        self.model = model_ft

        # If an initialized model is provided, override with its components
        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        # Extract features from DenseNet
        x = self.model.features(x)
        
        # Apply pooling based on the specified method
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        # Flatten the output
        x = x.view(x.size(0), x.size(1))
        
        return x
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(751, droprate=0.5, VGG16=True)
    #net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output,output = net(input,input)
    print('net output size:')
    print(output.shape)
