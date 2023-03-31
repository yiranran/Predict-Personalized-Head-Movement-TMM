import torch
import torch.nn as nn
# from pts3d import *
from ops import *
import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from convolutional_rnn import Conv2dGRU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
       
            )
        self.lstm = nn.LSTM(256*3,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,6),
            )

    def forward(self, example_landmark, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        example_landmark_f = self.lmark_encoder(example_landmark)
        #print 'example_landmark_f', example_landmark_f.shape # (1,512)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            features = torch.cat([example_landmark_f,  current_feature], 1)
            #print 'current_feature', current_feature.shape # (1,256)
            #print 'features', features.shape # (1,768)
            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)

class ATC_net(nn.Module):
    def __init__(self, para_dim):
        super(ATC_net, self).__init__()
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
       
            )
        self.lstm = nn.LSTM(256,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,para_dim),
            )

    def forward(self, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            lstm_input.append(current_feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden) # output, (hn,cn) = LSTM(input, (h0,c0))
        fc_out   = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)


class AT_single(nn.Module):
    def __init__(self):
        super(AT_single, self).__init__()
        # self.lmark_encoder = nn.Sequential(
        #     nn.Linear(6,256),
        #     nn.ReLU(True),
        #     nn.Linear(256,512),
        #     nn.ReLU(True),

        #     )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1,normalizer = None),
            conv2d(64,128,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1,normalizer = None),
            conv2d(256,256,3,1,1,normalizer = None),
            conv2d(256,512,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
            nn.Linear(256, 6)
            )
        # self.fusion = nn.Sequential(
        #     nn.Linear(256 *3, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 6)
        #     )

    def forward(self, audio):
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_eocder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)

        output = self.audio_eocder_fc(current_feature)
     
        return output


class GL_Discriminator(nn.Module):


    def __init__(self):
        super(GL_Discriminator, self).__init__()

        self.image_encoder_dis = nn.Sequential(
            conv2d(3,64,3,2, 1,normalizer=None),
            # conv2d(64, 64, 4, 2, 1),
            conv2d(64, 128, 3, 2, 1),

            conv2d(128, 256, 3, 2, 1),

            conv2d(256, 512, 3, 2, 1),
            )
        self.encoder = nn.Sequential(
            nn.Linear(136, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 136),
            nn.Tanh()
            )
        self.img_fc = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(True),
            )

        self.lstm = nn.LSTM(1024,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,136),
            nn.Tanh())
        self.decision = nn.Sequential(
            nn.Linear(256,1),
            )
        self.aggregator = nn.AvgPool1d(kernel_size = 16)
        self.activate = nn.Sigmoid()
    def forward(self, xs, example_landmark):
        hidden = ( torch.autograd.Variable(torch.zeros(3, example_landmark.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, example_landmark.size(0), 256).cuda()))
        lstm_input = list()
        lmark_feature= self.encoder(example_landmark)
        for step_t in range(xs.size(1)):
            x = xs[:,step_t,:,:, :]
            x.data = x.data.contiguous()
            x = self.image_encoder_dis(x)
            x = x.view(x.size(0), -1)
            x = self.img_fc(x)
            new_feature = torch.cat([lmark_feature, x], 1)
            lstm_input.append(new_feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        decision = []
        for step_t in range(xs.size(1)):
            fc_in = lstm_out[:,step_t,:]
            decision.append(self.decision(fc_in))
            fc_out.append(self.lstm_fc(fc_in)+ example_landmark)
        fc_out = torch.stack(fc_out, dim = 1)
        decision = torch.stack(decision, dim = 2)
        decision = self.aggregator(decision)
        decision = self.activate(decision)
        return decision.view(decision.size(0)), fc_out



class VG_net(nn.Module):
    def __init__(self,input_nc = 3, output_nc = 3,ngf = 64, use_dropout=True, use_bias=False,norm_layer=nn.BatchNorm2d,n_blocks = 9,padding_type='zero'):
        super(VG_net,self).__init__()
        dtype            = torch.FloatTensor


        self.image_encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3, 64, 7,1, 0),

            # conv2d(64,16,3,1,1),
            conv2d(64,64,3,2,1),
            # conv2d(32,64,3,1,1),
            conv2d(64,128,3,2,1)
            )

        self.image_encoder2 = nn.Sequential(
            conv2d(128,256,3,2,1),
            conv2d(256,512,3,2,1)
            )

        self.landmark_encoder  = nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(True)
            )

        self.landmark_encoder_stage2 = nn.Sequential(
            conv2d(1,256,3),
            
            )
        self.lmark_att = nn.Sequential(
            nn.ConvTranspose2d(512, 256,kernel_size=3, stride=(2),padding=(1), output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128,kernel_size=3, stride=(2),padding=(1), output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            conv2d(128, 1,3, activation=nn.Sigmoid, normalizer=None)
            )
        self.lmark_feature = nn.Sequential(
            conv2d(256,512,3)) 
     
        model = []
        n_downsampling = 4
        mult = 2**(n_downsampling -1  )
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling ):
            mult = 2**(n_downsampling-i-1 ) 
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            if i == n_downsampling-3:
                self.generator1 = nn.Sequential(*model)
                model = []

        self.base = nn.Sequential(*model)
        model = []
        model += [nn.Conv2d(int(ngf/2), output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.generator_color = nn.Sequential(*model)

        model = []
        model += [nn.Conv2d(int(ngf/2), 1, kernel_size=7, padding=3)]
        model += [nn.Sigmoid()]
        self.generator_attention = nn.Sequential(*model)

        self.bottle_neck = nn.Sequential(conv2d(1024,128,3,1,1))

        
        self.convGRU = Conv2dGRU(in_channels = 128, out_channels = 512, kernel_size = (3), num_layers = 1, bidirectional = False, dilation = 2, stride = 1, dropout = 0.5 )
        
    def forward(self,image, landmarks, example_landmark ):
        # ex_landmark1 = self.landmark_encoder(example_landmark.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128,128))
        image_feature1 = self.image_encoder1(image)
        image_feature = self.image_encoder2(image_feature1)
        ex_landmark1 = self.landmark_encoder(example_landmark.view(example_landmark.size(0), -1))
        ex_landmark1 = ex_landmark1.view(ex_landmark1.size(0), 1, image_feature.size(2), image_feature.size(3) )
        ex_landmark1 = self.landmark_encoder_stage2(ex_landmark1)
        ex_landmark = self.lmark_feature(ex_landmark1)
        
        lstm_input = list()
        lmark_atts = list()
        for step_t in range(landmarks.size(1)):
            landmark = landmarks[:,step_t,:]
            landmark.data = landmark.data.contiguous()
            landmark = self.landmark_encoder(landmark.view(landmark.size(0), -1))
            landmark = landmark.view(landmark.size(0), 1, image_feature.size(2), image_feature.size(3) )
            landmark = self.landmark_encoder_stage2(landmark)

            lmark_att = self.lmark_att( torch.cat([landmark, ex_landmark1], dim=1))
            landmark = self.lmark_feature(landmark)

            inputs =  self.bottle_neck(torch.cat([image_feature, landmark - ex_landmark], dim=1))
            lstm_input.append(inputs)
            lmark_atts.append(lmark_att)
        lmark_atts =torch.stack(lmark_atts, dim = 1)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_output, _ = self.convGRU(lstm_input)

        outputs = []
        atts = []
        colors = []
        for step_t in range(landmarks.size(1)):
            input_t = lstm_output[:,step_t,:,:,:]
            v_feature1 = self.generator1(input_t)
            v_feature1_f = image_feature1 * (1- lmark_atts[:,step_t,:,:,:] ) + v_feature1 * lmark_atts[:,step_t,:,:,:] 
            base = self.base(v_feature1_f)
            color = self.generator_color(base)
            att = self.generator_attention(base)
            atts.append(att)
            colors.append(color)
            output = att * color + (1 - att ) * image
            outputs.append(output)
        return torch.stack(outputs, dim = 1), torch.stack(atts, dim = 1), torch.stack(colors, dim = 1), lmark_atts


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, norm_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class VideoNet_Unet(nn.Module):
    def __init__(self):
        super(VideoNet_Unet, self).__init__()
        #Video
        vf_channels = [32, 64, 128 ,256, 512, 1024]
        norm_layer = nn.BatchNorm2d
        #norm_layer = nn.InstanceNorm2d
        
        
        self.inc = nn.Sequential(
            nn.Conv2d(6, vf_channels[0], 3, padding=1, bias=False),
            norm_layer(vf_channels[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down0 = Down(vf_channels[0], vf_channels[1], norm_layer)
        self.down1 = Down(vf_channels[1], vf_channels[2], norm_layer)
        self.down2 = Down(vf_channels[2], vf_channels[3], norm_layer)
        self.down3 = Down(vf_channels[3], vf_channels[4], norm_layer)
        self.down4 = Down(vf_channels[4], vf_channels[4], norm_layer)
        self.up1 = Up(vf_channels[5], vf_channels[3], norm_layer)
        self.up2 = Up(vf_channels[4], vf_channels[2], norm_layer)
        self.up3 = Up(vf_channels[3], vf_channels[1], norm_layer)
        self.up4 = Up(vf_channels[2], vf_channels[0], norm_layer)
        self.up5 = Up(vf_channels[1], vf_channels[0], norm_layer)

        self.out_color = nn.Sequential(
            nn.Conv2d(vf_channels[0], 3, 1),
            nn.Tanh()
        )
        

    def forward(self, img):
        down224 = self.inc(img)
        down112 = self.down0(down224)
        down56 = self.down1(down112)
        down28 = self.down2(down56)
        down14 = self.down3(down28)
        down7 = self.down4(down14)
        
        
        up = self.up1(down7, down14)
        up = self.up2(up, down28)
        up = self.up3(up, down56)
        up = self.up4(up, down112)
        up = self.up5(up, down224)
        color = self.out_color(up)
        '''
        mask = self.out_mask(up)
        img_out = img*(1-mask) + color*mask
        '''
        
        return color


class sequenceFC_layer(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, kernel_size=3, stride=1, padding=1, act_layer = nn.ReLU(inplace=True)):
        super(sequenceFC_layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding),
            norm_layer(out_ch),
            act_layer
        )

    def forward(self, x):
        return self.fc(x)

class FC_layer(nn.Module):
    def __init__(self, in_ch, out_ch, act_layer = nn.ReLU(inplace=True)):
        super(FC_layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            act_layer
        )

    def forward(self, x):
        return self.fc(x)

class iddEncoder_from_poseonly(nn.Module):
    def __init__(self, norm_layer = nn.BatchNorm1d, input_ch=6, idd1_ch=10, op=0):
        super(iddEncoder_from_poseonly, self).__init__()
        self.op = op

        self.pose_lstm = nn.LSTM(input_ch, 256, 3, batch_first = True)
        self.fc = nn.Sequential(
            FC_layer(1536, 256),
            FC_layer(256, 128),
            FC_layer(128, 64),
            FC_layer(64, 32),
            nn.Linear(32, idd1_ch)
        )

    def forward(self, pose):
        hidden0 = torch.autograd.Variable(torch.zeros(3, pose.size(0), 256).cuda(pose.device))
        cell0 = torch.autograd.Variable(torch.zeros(3, pose.size(0), 256).cuda(pose.device))
        pose_feature, (hidden, cell) = self.pose_lstm(pose, (hidden0, cell0)) # output, (hn,cn) = LSTM(input, (h0,c0))
        pose_feature_last = torch.cat([hidden, cell],0).permute(1,0,2).reshape([-1,1536])#[3,1,256],[3,1,256]->[1,1536]
        idd_feature = self.fc(pose_feature_last)

        return idd_feature, pose_feature, pose_feature_last

class contentMotionNet(nn.Module):
    def __init__(self, norm_layer = nn.BatchNorm1d, idd1_ch=10):
        super(contentMotionNet, self).__init__()

        af_channels = [256, 1024]

        self.audio_lstm = nn.LSTM(13, af_channels[0], 3, batch_first = True)
        self.audio_frame_gather = nn.Conv1d(af_channels[0], af_channels[1], 28, stride=4, padding=0)
        self.sfc = nn.Sequential(
            sequenceFC_layer(1024+idd1_ch, 512, norm_layer),
            sequenceFC_layer(512, 256, norm_layer),
            nn.Conv1d(256, 6, 1)
        )

    def forward(self, af, idd):
        hidden = ( Variable(torch.zeros(3, af.size(0), 256).cuda(af.device)),
                   Variable(torch.zeros(3, af.size(0), 256).cuda(af.device)) )
        audio_feature, _ = self.audio_lstm(af, hidden)
        audio_feature = self.audio_frame_gather(audio_feature.permute(0,2,1))
        idd = idd.unsqueeze(2)
        audio_feature = torch.cat((audio_feature, idd.expand(idd.shape[0], idd.shape[1], audio_feature.shape[2])), 1)
        res = self.sfc(audio_feature).permute(0,2,1)

        return res

class contentMotionNet2(nn.Module):
    def __init__(self, norm_layer = nn.BatchNorm1d, idd1_ch=10):
        super(contentMotionNet2, self).__init__()

        af_channels = [256, 1024]

        self.audio_lstm = nn.LSTM(13, af_channels[0], 3, batch_first = True)
        self.audio_frame_gather = nn.Conv1d(af_channels[0], af_channels[1], 28, stride=4, padding=0)
        self.sfc = nn.Sequential(
            sequenceFC_layer(1024+idd1_ch, 512, norm_layer),
            sequenceFC_layer(512, 256, norm_layer),
            nn.Conv1d(256, 3, 1)
        )


    def forward(self, af, idd):
        hidden = ( Variable(torch.zeros(3, af.size(0), 256).cuda(af.device)),
                   Variable(torch.zeros(3, af.size(0), 256).cuda(af.device)) )
        audio_feature, _ = self.audio_lstm(af, hidden)
        audio_feature = self.audio_frame_gather(audio_feature.permute(0,2,1))
        idd = idd.unsqueeze(2)
        audio_feature = torch.cat((audio_feature, idd.expand(idd.shape[0], idd.shape[1], audio_feature.shape[2])), 1)
        res = self.sfc(audio_feature).permute(0,2,1)

        return res