from turtle import shape
import torch
import os
import numpy as np

def merge_shape(m_shape,v):
    if m_shape == v.shape:
        return v
    elif len(m_shape) != len(v.shape):
        raise NotImplementedError
    else:
        print(v.shape,'to',m_shape)
        shape_list = []
        for i in range(len(v.shape)):
            m_dim = m_shape[i]
            v_dim = v.shape[i]
            if m_dim < v_dim:
                shape_list.append(m_dim)
            elif m_dim == v_dim:
                shape_list.append(v_dim)
            else:
                print('param extention not allowed!')
                return None
        if len(v.shape) ==1:
            return v[:shape_list[0]]
        elif len(v.shape) == 4:
            return v[:shape_list[0],:shape_list[1],:shape_list[2],:shape_list[3]]
        else:
            return None
def load_network_and_optimizer(net, opt, pretrained_dir, gpu, scaler=None):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    opt.load_state_dict(pretrained['optimizer'])
    if scaler is not None and 'scaler' in pretrained.keys():
        scaler.load_state_dict(pretrained['scaler'])
    del (pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove


def load_network_and_optimizer_v2(net, opt, pretrained_dir, gpu, scaler=None):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    # load model
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)

    # load optimizer
    opt_dict = opt.state_dict()
    all_params = {
        param_group['name']: param_group['params'][0]
        for param_group in opt_dict['param_groups']
    }
    pretrained_opt_dict = {'state': {}, 'param_groups': []}
    for idx in range(len(pretrained['optimizer']['param_groups'])):
        param_group = pretrained['optimizer']['param_groups'][idx]
        if param_group['name'] in all_params.keys():
            pretrained_opt_dict['state'][all_params[
                param_group['name']]] = pretrained['optimizer']['state'][
                    param_group['params'][0]]
            param_group['params'][0] = all_params[param_group['name']]
            pretrained_opt_dict['param_groups'].append(param_group)

    opt_dict.update(pretrained_opt_dict)
    opt.load_state_dict(opt_dict)

    # load scaler
    if scaler is not None and 'scaler' in pretrained.keys():
        scaler.load_state_dict(pretrained['scaler'])
    del (pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove


def load_network(net, pretrained_dir, gpu):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            _v = merge_shape(model_dict[k].shape,v)
            if _v != None:
                pretrained_dict_update[k] = _v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                _v = merge_shape(model_dict[k[7:]].shape,v)
                if _v != None:
                    pretrained_dict_update[k[7:]] = _v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del (pretrained)
    return net.cuda(gpu), pretrained_dict_remove


def save_network(net,
                 opt,
                 step,
                 save_path,
                 max_keep=8,
                 backup_dir='./saved_models',
                 scaler=None):
    ckpt = {'state_dict': net.state_dict(), 'optimizer': opt.state_dict()}
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save(ckpt, save_dir)
    except:
        save_path = backup_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save(ckpt, save_dir)

    all_ckpt = os.listdir(save_path)
    if max_keep is not None and len(all_ckpt) > max_keep:
        all_step = []
        for ckpt_name in all_ckpt:
            step = int(ckpt_name.split('_')[-1].split('.')[0])
            all_step.append(step)
        all_step = list(np.sort(all_step))[:-max_keep]
        for step in all_step:
            ckpt_path = os.path.join(save_path, 'save_step_%s.pth' % (step))
            os.system('rm {}'.format(ckpt_path))
