# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Written with <3 by Julien Romero
from numpy import *

from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from hoeffopttree import HoeffdingOptTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data import LEDGenerator, RandomTreeGenerator, RandomRBFGenerator, WaveformGenerator
import matplotlib as plt

plt.interactive(True)

#dataset = "elec"
dataset = "covtype"
#stream = FileStream("./data/"+dataset+".csv", n_targets=1, target_idx=-1)
#stream = LEDGenerator()
#stream = RandomTreeGenerator()
#stream = RandomRBFGenerator()
stream = WaveformGenerator(has_noise=False)
stream.prepare_for_use()
h = [
        HoeffdingOptTree(),
#       HoeffdingTree()
     ]

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=20000, show_plot=True, 
                                metrics=['accuracy'], output_file='result_'+dataset+'.csv', 
                                batch_size=1)
# 4. Run
evaluator.evaluate(stream=stream, model=h, model_names=["HOT"])

'''import pandas as pd
from matplotlib.pyplot import *

df = pd.read_csv('result_'+dataset+'.csv', comment='#')
ax = df.plot(x="id", y=["mean_acc_[HoeffdintOptTree]","mean_acc_[M1]","mean_acc_[M2]"], rot=45, linewidth=3, title=dataset)
#ax = df.plot(x="id", y=["current_acc_[M0]", "current_acc_[M1]", "current_acc_[M2]"], rot=30, linewidth=3, title=dataset)
#ax = df.plot(x="id", y=["mean_kappa_[M0]","mean_kappa_[M1]","mean_kappa_[M2]"], rot=45, linewidth=3, title=dataset)
#ax = df.plot(x="id", y=["current_kappa_[M0]", "current_kappa_[M1]", "current_kappa_[M2]"], rot=30, linewidth=3, title=dataset)
ax.set_xlabel("")
ax.set_title("Performance on the %s dataset" % dataset)
ax.legend([r"HT"], loc='best')
print("write out to %s ..." % dataset+".pdf")
#savefig("result_"+dataset+".pdf")
show()'''