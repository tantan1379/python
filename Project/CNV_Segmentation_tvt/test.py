# import utils.progress_bar as pb
# import utils.utils as u
# import utils.loss as LS
# from config import DefaultConfig
# import os
# from model.cpfnet import CPFNet

# args = DefaultConfig()
# for c in os.listdir(os.path.join("./checkpoints",args.net_work)):
#         print(c)
#         pretrained_model_path = os.path.join("./checkpoints",args.net_work,c) # 最后一个模型(最好的)

# model = CPFNet()
# model.load_state_dict(checkpoint['state_dict'])
# print("Load best model "+'\"'+os.path.abspath(pretrained_model_path)+'\"')
# checkpoint = torch.load(pretrained_model_path)
# model.load_state_dict(checkpoint['state_dict'])
# with torch.no_grad():
#     model.eval()
#     test_progressor = pb.Test_ProgressBar(total=len(dataloader),model_name=args.net_work)

#     total_Dice = []
#     total_Acc = []
#     total_jaccard = []
#     total_Sensitivity = []
#     total_Specificity = []

#     for i, (data, label) in enumerate(dataloader):
#         test_progressor.current = i
#         if torch.cuda.is_available() and args.use_gpu:
#             data = data.cuda()
#             label = label.cuda()

#         # get RGB predict image
#         predict = model(data)
#         Dice, Acc, jaccard, Sensitivity, Specificity = u.eval_single_seg(predict, label)

#         total_Dice += Dice
#         total_Acc += Acc
#         total_jaccard += jaccard
#         total_Sensitivity += Sensitivity
#         total_Specificity += Specificity
    
#         dice = sum(total_Dice) / len(total_Dice)
#         acc = sum(total_Acc) / len(total_Acc)
#         jac = sum(total_jaccard) / len(total_jaccard)
#         sen = sum(total_Sensitivity) / len(total_Sensitivity)
#         spe = sum(total_Specificity) / len(total_Specificity)
#         test_progressor.val=[dice,acc,jac,sen,spe]
#         test_progressor()    
#     test_progressor.done()