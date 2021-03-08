
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.autils import where
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape=(1, 32, 32), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std
        
    def forward(self, x):
        self.noise.data.normal_(0.2, std=self.std)
        return x + self.noise

class Attacks(object):
    """
    An abstract class representing attacks.

    Arguments:
        name (string): name of the attack.
        model (nn.Module): a model to attack.

    .. note:: device("cpu" or "cuda") will be automatically determined by a given model.
    
    """
    def __init__(self, name, model):
        self.attack = name
        self.model = model.eval()
        self.model_name = str(model).split("(")[0]
        self.device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
                
    # Whole structure of the model will be NOT displayed for pretty print.        
    def __str__(self):
        info = self.__dict__.copy()
        del info['model']
        del info['attack']
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"
    
    # Save image data as torch tensor from data_loader
    # If you want to reduce the space of dataset, set 'to_unit8' as True
    # If you don't want to know about accuaracy of the model, set accuracy as False
    def save(self, file_name, data_loader, to_uint8 = True, accuracy = True):
        image_list = []
        label_list = []
        
        correct = 0
        total = 0
        
        total_batch = len(data_loader)
        
        for step, data in enumerate(data_loader) :
            images, labels, index = data['image'], data['labels'], data['index']
            adv_images = self.__call__(images, labels)
            
            if accuracy :
                outputs = self.model(adv_images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

            if to_uint8 :
                image_list.append((adv_images*255).type(torch.uint8).cpu())
            else :
                image_list.append(adv_images.cpu())
                
            label_list.append(labels)
        
            print('- Save Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')
        
        if accuracy :
            acc = 100 * float(correct) / total
            print('\n- Accuracy of the model : %f %%' % (acc), end='')
        
        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        
        torch.save((x, y), file_name)
        
        print('\n- Save Complete!')
        
    # Load image data as torch dataset
    # When scale=True it automatically tansforms images to [0, 1]
    def load(self, file_name, scale = True) :
        adv_images, adv_labels = torch.load(file_name)
        
        if scale :
            adv_data = torch.utils.data.TensorDataset(adv_images.float() / adv_images.max(), adv_labels)
        else :
            adv_data = torch.utils.data.TensorDataset(adv_images.float(), adv_labels)
            
        return adv_data
    
    ########################################## DEPRECIATED ##########################################
    '''
    # Evaluate accuaracy of a model
    # With default 'model = None', it will return accuracy of white box attack
    # If not, it will return accuracy of black box attack with self.model as holdout model
    
    def eval(self, data_loader, model = None) :
        
        if model is None :
            model = self.model
        else :
            model = model.eval()

        correct = 0
        total = 0

        total_batch = len(data_loader)
        
        for step, (images, labels) in enumerate(data_loader) :

            adv_images = self.__call__(images, labels)
            outputs = model(adv_images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum()
            
            print('- Evaluation Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')

        accuracy = 100 * float(correct) / total
        print('\n- Accuracy of model : %f %%' % (accuracy))

        return accuracy
    '''

class FGSM(Attacks):
    """
    FGSM attack in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.007)
    
    """
    def __init__(self, model, eps=0.5):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps
        self.gn = GaussianNoise(100, input_shape=(1, 32, 32), std=1.0)
    
    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        images.requires_grad = True
        outputs,_ = self.model(images)

        self.model.zero_grad()
        cost = loss(outputs, labels).to(self.device)
        cost.backward()

        adv_images = images + self.eps*images.grad.sign()


        adv_images = torch.clamp(adv_images, min=0, max=1).detach_()

        return adv_images


    def eval(self, data_loader, model = None) :
        
        if model is None :
            model = self.model
        else :
            model = model.eval()

        correct = 0
        total = 0

        total_batch = len(data_loader)
        
        for step, data in enumerate(data_loader) :
            images, labels, index = data['image'], data['labels'], data['index']

            adv_images = self.__call__(images, labels)
            with torch.no_grad():
                # adv_images = self.gn(adv_images)
                outputs,_= model(adv_images)
            
            total += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(self.device)).sum()    

            print('- Evaluation Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')

        accuracy = 100 * float(correct) / total
        print('\n- attack accuracy: {}'.format(accuracy))

        return accuracy


class IFGSM(Attacks):
    """
    FGSM attack in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.007)
    
    """
    def __init__(self, model, eps=0.03, alpha=1.0, iterations=10):
        super(IFGSM, self).__init__("FGSM", model)
        self.eps = eps
        self.model = model
        self.alpha = alpha
        self.iterations = iterations
        self.gn = GaussianNoise(100, input_shape=(1, 32, 32), std=1.0)
    
    def __call__(self, x, y, targeted=False, x_val_min=0, x_val_max=1.0):
        x = x.to(self.device)
        y = y.to(self.device)
        loss = nn.CrossEntropyLoss()
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(self.iterations):
            h_adv = self.model(x_adv)
            if targeted:
                cost = loss(h_adv, y)
            else:
                cost = -loss(h_adv, y)

            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - self.alpha*x_adv.grad
            x_adv = where(x_adv > x+self.eps, x+self.eps, x_adv)
            x_adv = where(x_adv < x-self.eps, x-self.eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        return x_adv

    def eval(self, data_loader, model = None) :
        
        if model is None :
            model = self.model
        else :
            model = model.eval()

        correct = 0
        total = 0

        total_batch = len(data_loader)
        
        for step, (images, labels) in enumerate(data_loader) :

            adv_images = self.__call__(images, labels)
            with torch.no_grad():
                # adv_images = self.gn(adv_images)
                outputs = model(adv_images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum()
            
            print('- Evaluation Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')

        accuracy = 100 * float(correct) / total
        print('\n- Accuracy of model : %f %%' % (accuracy))

        return accuracy


class PGD(Attacks):
    """
    CW attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)
        
    """
    def __init__(self, model, eps=0.3, alpha=2/255, iters=40):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
    
    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.data
        
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs,_ = self.model(images)
        
            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        adv_images = images
        
        return adv_images


    def eval(self, data_loader, model = None) :
        
        if model is None :
            model = self.model
        else:
            model = model.eval()

        correct = 0
        total = 0

        total_batch = len(data_loader)
        
        for step, data in enumerate(data_loader) :
            images, labels, index = data['image'], data['labels'], data['index']

            adv_images = self.__call__(images, labels)
            with torch.no_grad():
                outputs,_ = model(adv_images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum()
            
            print('- Evaluation Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')

        accuracy = 100 * float(correct) / total
        print('\n- Accuracy of model : %f %%' % (accuracy))

        return accuracy

class CWL2(Attacks):
    def __init__(self, model, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
        super(CWL2, self).__init__("CWL2", model)
        self.model = model
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.lr = learning_rate
        

    def __call__(self, images, labels):
        model = self.model
        images = images.to(self.device)     
        labels = labels.to(self.device)

        # Define f-function
        def f(x):
            outputs = model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.byte())
            
            # If targeted, optimize for making the other class most likely 
            if self.targeted :
                return torch.clamp(i-j, min=-self.kappa)
            
            # If untargeted, optimize for making the other class most likely 
            else :
                return torch.clamp(j-i, min=-self.kappa)
        
        w = torch.zeros_like(images, requires_grad=True).to(self.device)

        optimizer = optim.Adam([w], lr=self.lr)

        prev = 1e10
        
        for step in range(self.max_iter):

            a = 1/2*(nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(self.c*f(a))


            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.max_iter//10) == 0 :
                if cost > prev :
                    print('Attack Stopped due to CONVERGENCE....')
                    return a
                prev = cost
            print('- Learning Progress : %2.2f %%        ' %((step+1)/self.max_iter*100), end='\r')

        attack_images = 1/2*(nn.Tanh()(w) + 1)
        
        return attack_images

    def eval(self, data_loader, model = None):
        
        if model is None :
            model = self.model
        else:
            model = model.eval()

        correct = 0
        total = 0

        for images, labels in data_loader:
            
            images = self.__call__(images, labels)
            labels = labels.to(self.device)
            
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)

            total += outputs.size(0)
            correct += (pre == labels).sum()
            
            # imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])

        accuracy = 100 * float(correct) / total
        print('\n- Accuracy of model : %f %%' % (accuracy))

        return accuracy