# architecture1  for  square-shape input 
import torch.nn as nn
from torchsummary import summary
from torchvision import models
import torch

#visualize the summary of model
def visualize_model_summary(net, input1_size, input2_size):

    summary(net,[input1_size, input2_size])

class loss_fn(nn.Module):
  def __init__(self):
    super(loss_fn,self).__init__()

  def forward(self, out_s, out_t, dis_label):
    batch_size = out_s.size()[0]
    
    # out_s, out_t are representation of images pairs thought the network
    # we hope that: with decreasing the distance of source image from tg image, the loss of the two images' output from network will decrese too. 
    diff =  torch.linalg.norm(out_s - out_t, dim = 1)

    loss = torch.nn.functional.mse_loss(diff, dis_label)

    return loss

def compute_loss_comp(out_s, conf_loss):
    # We want: d_k – d_{k-1} to be greater than 0, if it is >0 then the loss is 0 
    # Relu(x) = max(x, 0) 
    # dd_k = d_k – d_{k-1} 
    # loss = sum_{k=1,N} relu( -dd_k ) – alpha_k * relu( dd_k ) 
    # alpha_k – is in [0,1] and encourages positive distance differences, i.e. positive dd_k, 
    # alpha_k is large for small k, and then goes against 0 
    n = out_s.shape[0]
    dd = out_s[1:] - out_s[:-1]  
    #print(output.shape)
    #print("----dd----", dd)
    alphas = [0.3, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    alphas_tensor = torch.tensor(alphas, device=torch.cuda.current_device())
    alphas_tensor = alphas_tensor.reshape((9,1))
    #loss =  torch.sum(torch.nn.functional.relu((-1)*dd) - torch.mul(alphas_tensor,  torch.nn.functional.relu(dd)))

    #disable t1
    #t1 = torch.tensor(0).cuda()
    t1  = torch.abs(out_s[0])
    #disable t2
    t2 = torch.tensor(0).cuda()
    #t2 = torch.sum(torch.nn.functional.relu((-1)*dd))
    #print(torch.nn.functional.sigmoid(dd))
    #t3 = - torch.sum(torch.mul(alphas_tensor,dd))
    #print(torch.mul(alphas_tensor,  torch.nn.functional.sigmoid(dd)))
    #t3 =  - torch.sum(torch.mul(alphas_tensor,  torch.nn.functional.sigmoid(dd)))
    t3 = 1 - torch.sum(torch.mul(alphas_tensor,  torch.nn.functional.sigmoid(dd)))

    return  t1, t2, t3

class loss_seq_fn(nn.Module):
    def __init__(self, conf_loss):
      super(loss_seq_fn,self).__init__()
      self.conf_loss = conf_loss


    def forward(self, out_s):
        t_weight = [0.1, 0, 0.9]
        t1, t2, t3 = compute_loss_comp(out_s, self.conf_loss)
        loss = t_weight[0]* self.conf_loss.w_t1 + t_weight[1]* self.conf_loss.w_t2 + t_weight[2]*self.conf_loss.w_t3

        #print("loss terms : ", t1.item(),t2.item(), t3.item()  )
        #print("loss:", loss.item())
        
        # check the components of loss


        return loss, [t1.data.item(), t2.data.item(), t3.data.item() ]


class feature_op_model(nn.Module):
    def __init__(self):
        super(feature_op_model, self).__init__()
        
        self.conv_im1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 10),  
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),   

            nn.Conv2d(32, 64, 7), 
            nn.ReLU(inplace=True),   
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),        
        )
        
        self.conv_im2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 10),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),   

            nn.Conv2d(32, 64, 7), 
            nn.ReLU(inplace=True),   
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),        
        )

        self.ext = nn.Sequential(
            nn.Conv2d(64, 128, 3),  
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),        

            nn.Conv2d(128, 256, 3),  
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),     

            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True)  
        )
        self.linear = nn.Linear(18432, 1024)
        self.out = nn.Linear(1024, 1)
  
    def conv_shared(self, x):
      x = self.ext(x)
      
      x = x.view(x.size()[0], -1)
      #print('after shared conv--', x.shape)
      x = self.linear(x)
      return x

    
    def forward(self, input1, input2):
      # pass 2 images to sepearate conv layers, followed by shared conv layers, then linear layers finally
      x1 = self.conv_im1(input1)
      x1 = self.conv_shared(x1)
      x2 = self.conv_im2(input2)
      x2 = self.conv_shared(x2)
      # print('after  conv--', x1.shape)   ([4, 512, 4, 4])

      # x2 = self.conv_shared(x2)
      #dif = torch.abs(x1 - x2)
      dif = x1 - x2
      output = self.out(dif)

      return output


# architecture  for input with original size
import torch.nn as nn
class feature_op_model2(nn.Module):
    def __init__(self):
        super(feature_op_model2, self).__init__()
        
        self.conv_im1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 10),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   

            nn.Conv2d(16, 32, 7), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2),      

            nn.Conv2d(32, 64, 7), 
            nn.ReLU(inplace=True),   
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),      
        )
        
        self.conv_im2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 10),
            nn.BatchNorm2d(16),          
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   

            nn.Conv2d(16, 32, 7), 
            nn.BatchNorm2d(32),          
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2),        

            nn.Conv2d(32, 64, 7), 
            nn.BatchNorm2d(64),          
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2),  
        )

        self.ext = nn.Sequential(
            nn.Conv2d(64, 128, 3),  
            nn.BatchNorm2d(128),          
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2),
            #nn.Dropout2d(p=0.3),     

            nn.Conv2d(128, 256, 3), 
            nn.BatchNorm2d(256),           
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2),   
            #nn.Dropout2d(p=0.3),   

            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),  
            #nn.Dropout2d(p=0.3)
        )

        # self.linear = nn.Sequential(
        #   nn.Linear(7168, 1024),
        #   #nn.Dropout2d(p=0.3),          
        # )
        self.linear = nn.Linear(7168, 1024)
        self.dropout2 = nn.Dropout(p=0.2)

        self.out = nn.Linear(1024, 1)
  
    def conv_shared(self, x):
      x = self.ext(x)
      x = x.view(x.size()[0], -1)
      #print('after shared conv--', x.shape)
      x = self.linear(x)
      #x = self.dropout2(x)
      return x

    
    def forward(self, input1, input2):
      # pass 2 images to sepearate conv layers, followed by shared conv layers, then linear layers finally
      x1 = self.conv_im1(input1)
      x1 = self.conv_shared(x1)
      x2 = self.conv_im2(input2)
      x2 = self.conv_shared(x2)
      # print('after  conv--', x1.shape)   ([4, 512, 4, 4])

      
      #dif = x1 - x2
      #dif = torch.square(x1 - x2)
      #experiement 1  fully connected layer
      #output = self.out(dif)
      
      #experiment 2  without last linear layer
      output = torch.linalg.norm(x1-x2, dim =1, keepdim=True)

      #print("output ---shape, ", output.shape)
      #output = torch.sum(dif, dim = 1, keepdim=True)
      
      return output

# model 3: 
# Extract embedding using Resnet
class learn_dis_model(nn.Module):
    def __init__(self):
      super(learn_dis_model, self).__init__()

      resnet18_1 = models.resnet18(pretrained=True)
      resnet18_2 = models.resnet18(pretrained=True)

      res_fc_outsize = resnet18_1.fc.in_features

      self.resnet18_mod_1 = nn.Sequential(*list(resnet18_1.children())[:-1])
      self.resnet18_mod_2 = nn.Sequential(*list(resnet18_2.children())[:-1])
      #self.fc_1 = nn.Linear(res_fc_outsize, 1)
      self.fc = nn.Linear(res_fc_outsize, 1)

    def forward(self, x_t, x_s):
      x_t = self.resnet18_mod_1(x_t)
      x_t = torch.flatten(x_t, start_dim=1)
      #print(x_t.shape)
      
      x_s = self.resnet18_mod_2(x_s)
      x_s = torch.flatten(x_s, start_dim=1)
      #print(x_s.shape)

      diff = torch.square(x_t - x_s)

      out = self.fc(diff)

      return out

#prin params to learn
def print_model_params(net, feature_extract =True):
    #params_to_update = model_ft.parameters()
    params_to_update = net.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

def cal_pose_dis(t, s, alpha =10):
    ## size: (batch_size, 7)

    t_p, t_r = t[:,:3], t[:,3:]
    s_p, s_r = s[:,:3], s[:,3:]
    #print('--', t_p.shape, s_p.shape)
    # position distance: Euclidian Norm
    p_dis =  torch.linalg.norm( t_p - s_p, dim =1)
    # rotation distance: Euclidean Norm between wxyz
    r_dis =  torch.linalg.norm( t_r - s_r, dim =1)
    
    ##  euclidean_distance = F.pairwise_distance(output1, output2)

    # alpha: parameter to offset the two distans with different units
    dis = p_dis + alpha * r_dis

    return dis
