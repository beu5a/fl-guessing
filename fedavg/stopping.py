def check_stopping_criteria_femnist(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.77 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False
  # if ((abs(loss_history[9] - loss_history[8]) < 0.0001) or 
  # (abs(loss_history[9] - loss_history[6]) > 2.0)):
  #   return True
  # else:
  #   return False

def check_stopping_criteria_cifar10(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.65 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False

def check_stopping_criteria_shakespeare(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.50 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False

def check_stopping_criteria_celeba(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.885 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False

def check_stopping_criteria_synthetic(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.81 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False

def check_stopping_criteria_sent140(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.70 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False

def check_stopping_criteria_reddit(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.12 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False 