import tensorflow as tf
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class StopEarly(tf.keras.callbacks.Callback):
    """
    Custom Early Stopping callback factoring in the validation chi-squared metric 
    """    
    def __init__(self, monitor = 'val_loss', patience = 5, start_from_epoch = 1, target_chisq = 2.0, epsilon_chisq = 0.05, verbose = True, restore_best_weights = True):
        super(StopEarly, self).__init__()
        self.monitor  = monitor
        self.patience = patience
        self.target_chisq  = target_chisq
        self.epsilon_chisq = epsilon_chisq
        self.verbose       = verbose
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch     = start_from_epoch
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value    = None
        self.best_epoch    = None
        self.best_weights  = None
        
        
    def on_train_begin(self, logs = None):
        self.best_epoch   = 0    # Note this is zero-indexed
        self.best_value   = float('inf') if 'loss' in self.monitor else -float('inf')
        self.best_weights = None
        # self.good_metrics = []
        
        
    def on_epoch_end(self, epoch, logs = None):
        # if epoch+1 < self.start_from_epoch:
        #     return 
        
        chisq    = logs.get('val_ChiSquaredError')
        current  = logs.get(self.monitor)
        # if abs(chisq - self.target_chisq) < self.epsilon_chisq:
        #     self.good_metrics.append(current)
            
        if 'loss' in self.monitor:
            if (current < self.best_value) and (abs(chisq - self.target_chisq) < self.epsilon_chisq):
                self.best_value = current
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                if epoch+1 > self.start_from_epoch:
                    self.wait = 0
            else:
                if epoch+1 > self.start_from_epoch:
                    self.wait += 1
        else:
            if (current > self.best_value) and (abs(chisq - self.target_chisq) < self.epsilon_chisq):
                self.best_value = current
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                if epoch+1 > self.start_from_epoch:
                    self.wait = 0
            else:
                if epoch+1 > self.start_from_epoch:
                    self.wait += 1
                    
        if epoch >= self.start_from_epoch and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            tf.keras.backend.clear_session()
            if self.verbose:
                print(f"Stopping early at epoch {self.stopped_epoch + 1}")
                
                
    def on_train_end(self, logs = None):
        if self.stopped_epoch > 0 and self.verbose:
            if self.restore_best_weights and self.best_weights is not None:
                print(f"Restoring model weights from the end of the best epoch {self.stopped_epoch - self.patience + 1}.")
                self.model.set_weights(self.best_weights)
            else:
                print(f"Training stopped at epoch {self.stopped_epoch + 1}, without restoring weights.")




class MyCheckPoint(tf.keras.callbacks.Callback):
    """
    Custom Checkpoint callback factoring in the validation chi-squared metric 
    """    
    def __init__(self, filepath, mode = 'min', save_weights_only = True, monitor = 'val_loss', target_chisq = 2.0, epsilon_chisq = 0.05, save_format = 'h5', save_best_only = True):
        super(MyCheckPoint, self).__init__()
        self.filepath = filepath
        self.monitor  = monitor
        self.mode     = mode
        self.target_chisq  = target_chisq
        self.epsilon_chisq = epsilon_chisq
        self.save_format   = save_format
        self.save_weights_only    = save_weights_only
        self.save_best_only       = save_best_only
        
        self.best_value    = None
        self.best_epoch    = None
        self.best_weights  = None
        
    def on_train_begin(self, logs = None):
        self.best_epoch     = 0    # Note this is zero-indexed
        self.best_weights   = None
        if self.mode == 'min':
            self.best_value = float('inf') 
        elif self.mode == 'max':
            self.best_value = -float('inf')
        else:
            raise ValueError("Invalid mode for checkpoint metric monitoring")
        
        if type(self.filepath) == str:
            Path(self.filepath).parent.mkdir(exist_ok = True, parents = True)
        else:
            self.filepath.parent.mkdir(exist_ok = True, parents = True)
        

    def on_epoch_end(self, epoch, logs = None):
        if not self.save_best_only:
            save_this_epoch = True
        else:
            chisq    = logs.get('val_ChiSquaredError')
            current  = logs.get(self.monitor)
                
            if self.mode == 'min':
                save_this_epoch = (current < self.best_value) and (abs(chisq - self.target_chisq) < self.epsilon_chisq)
            elif self.mode == 'max':
                save_this_epoch = (current > self.best_value) and (abs(chisq - self.target_chisq) < self.epsilon_chisq)
            else:
                raise ValueError("Invalid mode for checkpoint metric monitoring")

        if save_this_epoch:
            self.best_value   = current
            self.best_epoch   = epoch
            self.best_weights = self.model.get_weights()
            if self.save_weights_only:
                self.model.save_weights(str(self.filepath).format(epoch = epoch+1), save_format = self.save_format)
            else:
                self.model.save(str(self.filepath).format(epoch = epoch+1), save_format = self.save_format)
        